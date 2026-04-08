import cv2
import numpy as np
import multiprocessing as mp
import logging

GLUESTICK_DEFAULT_CONF = {
    'name': 'two_view_pipeline',
    'use_lines': True,
    'extractor': {
        'name': 'wireframe',
        'sp_params': {'force_num_keypoints': False, 'max_num_keypoints': 1000},
        'wireframe_params': {'merge_points': True, 'merge_line_endpoints': True},
        'max_n_lines': 300,
    },
    'matcher': {'name': 'gluestick', 'weights': 'checkpoint_GlueStick_MD.tar'},
    'ground_truth': {'from_pose_depth': False},
}


def _gs_worker(conn, pair_images, max_keypoints, max_lines, use_lines, ransac_threshold, max_edge):
    try:
        import torch
        import copy
        from gluestick import batch_to_np
        from gluestick.models.two_view_pipeline import TwoViewPipeline

        conf = copy.deepcopy(GLUESTICK_DEFAULT_CONF)
        conf['use_lines'] = use_lines
        conf['extractor']['sp_params']['max_num_keypoints'] = max_keypoints
        conf['extractor']['max_n_lines'] = max_lines
        pipeline = TwoViewPipeline(conf).to('cpu').eval()

        batch_src, batch_tgt = [], []
        for img0, img1 in pair_images:
            pts0, pts1 = _gs_match_pair(pipeline, img0, img1, use_lines, ransac_threshold, max_edge)
            batch_src.append(pts0)
            batch_tgt.append(pts1)
        conn.send((batch_src, batch_tgt))
    except Exception as e:
        conn.send(e)
    finally:
        conn.close()


def _gs_match_pair(pipeline, img0, img1, use_lines, ransac_threshold, max_edge):
    import torch
    from gluestick import numpy_image_to_torch, batch_to_np

    gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY) if img0.ndim == 3 else img0
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1

    h0, w0 = gray0.shape[:2]
    h1, w1 = gray1.shape[:2]
    scale0 = min(max_edge / max(h0, w0), 1.0)
    scale1 = min(max_edge / max(h1, w1), 1.0)
    if scale0 < 1.0:
        gray0 = cv2.resize(gray0, (int(w0 * scale0), int(h0 * scale0)), interpolation=cv2.INTER_AREA)
    if scale1 < 1.0:
        gray1 = cv2.resize(gray1, (int(w1 * scale1), int(h1 * scale1)), interpolation=cv2.INTER_AREA)

    torch_gray0 = numpy_image_to_torch(gray0).to('cpu')[None]
    torch_gray1 = numpy_image_to_torch(gray1).to('cpu')[None]
    x = {'image0': torch_gray0, 'image1': torch_gray1}

    try:
        with torch.inference_mode():
            pred = batch_to_np(pipeline(x))
    except Exception:
        empty = np.array([], dtype=np.float32).reshape(0, 2)
        return empty, empty

    kp0, kp1 = pred['keypoints0'], pred['keypoints1']
    m0 = pred['matches0']
    valid = m0 != -1
    matched_kps0 = kp0[valid]
    matched_kps1 = kp1[m0[valid]]

    if use_lines:
        line_seg0 = pred.get('lines0')
        line_matches = pred.get('line_matches0')
        if line_seg0 is not None and line_matches is not None:
            valid_lines = line_matches != -1
            if valid_lines.any():
                ml0 = line_seg0[valid_lines].reshape(-1, 2)
                ml1 = pred['lines1'][line_matches[valid_lines]].reshape(-1, 2)
                matched_kps0 = np.concatenate([matched_kps0, ml0], axis=0)
                matched_kps1 = np.concatenate([matched_kps1, ml1], axis=0)

    pts0 = matched_kps0.astype(np.float32)
    pts1 = matched_kps1.astype(np.float32)
    if scale0 < 1.0:
        pts0 /= scale0
    if scale1 < 1.0:
        pts1 /= scale1

    if len(pts0) < 4:
        return pts0, pts1

    _, mask = cv2.findHomography(pts0, pts1, cv2.USAC_MAGSAC,
                                  ransacReprojThreshold=ransac_threshold,
                                  maxIters=5000, confidence=0.9999)
    if mask is None:
        return pts0, pts1
    inlier = mask.ravel().astype(bool)
    return pts0[inlier], pts1[inlier]


class GlueStickMatcher:
    def __init__(self, device='cpu', max_keypoints=1000, max_lines=300,
                 use_lines=True, ransac_threshold=5.0, max_edge=1024):
        self.max_keypoints = max_keypoints
        self.max_lines = max_lines
        self.use_lines = use_lines
        self.ransac_threshold = ransac_threshold
        self.max_edge = max_edge

    def inference(self, image_pairs):
        pair_images = []
        for i0, i1 in image_pairs:
            img0 = i0.image if hasattr(i0, 'image') else i0
            img1 = i1.image if hasattr(i1, 'image') else i1
            pair_images.append((img0, img1))

        ctx = mp.get_context('spawn')
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_gs_worker, args=(
            child_conn, pair_images, self.max_keypoints, self.max_lines,
            self.use_lines, self.ransac_threshold, self.max_edge,
        ))
        proc.start()
        child_conn.close()
        result = parent_conn.recv()
        proc.join()
        parent_conn.close()

        if isinstance(result, Exception):
            raise result
        return result
