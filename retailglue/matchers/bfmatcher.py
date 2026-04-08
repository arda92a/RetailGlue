"""
BFMatcher – brute-force matching on DINO embeddings without any learned matcher.

Uses cv2.BFMatcher with L2 norm on raw DINO embedding descriptors.  The API
mirrors LightGlueMatcher so it can be used as a drop-in replacement inside
ImageStitcher.
"""
import cv2
import numpy as np
import logging

from retailglue.entities import collect_embeddings

logger = logging.getLogger("retailglue")


class BFMatcher:
    """Product-level brute-force matcher using DINO embeddings."""

    def __init__(self, device='cpu', ransac_threshold=5.0, min_matches=4,
                 ratio_threshold=0.75, cross_check=False):
        self.device = device
        self.ransac_threshold = ransac_threshold
        self.min_matches = min_matches
        self.ratio_threshold = ratio_threshold
        self.cross_check = cross_check
        self.last_matches_info = {}

        if cross_check:
            self._bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            self._bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # ------------------------------------------------------------------
    # Helpers (same as LightGlueMatcher)
    # ------------------------------------------------------------------

    def _get_products(self, idx, detections):
        if detections is None or idx >= len(detections):
            return []
        return [d for d in detections[idx] if d.name == 'product']

    def _get_keypoints(self, products):
        if not products:
            return np.array([], dtype=np.float32).reshape(0, 2)
        return np.array(
            [[(p.xmin + p.xmax) / 2, (p.ymin + p.ymax) / 2] for p in products],
            dtype=np.float32,
        )

    def _determine_image_order(self, products0, products1, matches, w0, w1):
        if not matches:
            return 'left_right'
        avg_x0 = np.mean([(products0[m[0]].xmin + products0[m[0]].xmax) / 2 for m in matches])
        avg_x1 = np.mean([(products1[m[1]].xmin + products1[m[1]].xmax) / 2 for m in matches])
        return 'left_right' if avg_x0 / w0 > avg_x1 / w1 else 'right_left'

    def _is_partial_product(self, product, image_width, image_height, edge_threshold_ratio=0.02):
        edge_threshold_x = image_width * edge_threshold_ratio
        at_left = product.xmin <= edge_threshold_x
        at_right = product.xmax >= image_width - edge_threshold_x
        if at_left and not at_right:
            return True, 'left'
        elif at_right and not at_left:
            return True, 'right'
        return False, 'none'

    def _calculate_aspect_ratio(self, product):
        return (product.xmax - product.xmin) / (product.ymax - product.ymin + 1e-6)

    def _adjust_keypoints_for_partial_products(self, products0, products1, matches,
                                                image_order, w0, h0, w1, h1,
                                                aspect_ratio_threshold=0.6):
        pts0_expanded, pts1_expanded, adjusted_indices = [], [], []
        for match_idx, m in enumerate(matches):
            p0, p1 = products0[m[0]], products1[m[1]]
            is_partial0, side0 = self._is_partial_product(p0, w0, h0)
            is_partial1, side1 = self._is_partial_product(p1, w1, h1)
            ar0, ar1 = self._calculate_aspect_ratio(p0), self._calculate_aspect_ratio(p1)

            cx0, cy0 = (p0.xmin + p0.xmax) / 2, (p0.ymin + p0.ymax) / 2
            cx1, cy1 = (p1.xmin + p1.xmax) / 2, (p1.ymin + p1.ymax) / 2

            pts0_5 = [[cx0, cy0], [p0.xmin, p0.ymin], [p0.xmax, p0.ymin],
                       [p0.xmax, p0.ymax], [p0.xmin, p0.ymax]]
            pts1_5 = [[cx1, cy1], [p1.xmin, p1.ymin], [p1.xmax, p1.ymin],
                       [p1.xmax, p1.ymax], [p1.xmin, p1.ymax]]

            needs_adjustment = False
            if is_partial1 and not is_partial0 and ar1 < ar0 * aspect_ratio_threshold:
                needs_adjustment = True
                visible_ratio = ar1 / ar0 if ar0 > 0 else 1.0
                p0_width = p0.xmax - p0.xmin
                visible_width = p0_width * visible_ratio
                adjust_left = ((image_order == 'left_right' and side1 == 'right') or
                               (image_order == 'right_left' and side1 == 'left'))
                if adjust_left:
                    new_xmax = p0.xmin + visible_width
                    new_cx = (p0.xmin + new_xmax) / 2
                    pts0_5 = [[new_cx, cy0], [p0.xmin, p0.ymin], [new_xmax, p0.ymin],
                               [new_xmax, p0.ymax], [p0.xmin, p0.ymax]]
                else:
                    new_xmin = p0.xmax - visible_width
                    new_cx = (p0.xmax + new_xmin) / 2
                    pts0_5 = [[new_cx, cy0], [new_xmin, p0.ymin], [p0.xmax, p0.ymin],
                               [p0.xmax, p0.ymax], [new_xmin, p0.ymax]]
            elif is_partial0 and not is_partial1 and ar0 < ar1 * aspect_ratio_threshold:
                needs_adjustment = True
                visible_ratio = ar0 / ar1 if ar1 > 0 else 1.0
                visible_width = (p1.xmax - p1.xmin) * visible_ratio
                adjust_left = ((image_order == 'left_right' and side0 == 'right') or
                               (image_order == 'right_left' and side0 == 'left'))
                if adjust_left:
                    new_xmax = p1.xmin + visible_width
                    new_cx = (p1.xmin + new_xmax) / 2
                    pts1_5 = [[new_cx, cy1], [p1.xmin, p1.ymin], [new_xmax, p1.ymin],
                               [new_xmax, p1.ymax], [p1.xmin, p1.ymax]]
                else:
                    new_xmin = p1.xmax - visible_width
                    new_cx = (p1.xmax + new_xmin) / 2
                    pts1_5 = [[new_cx, cy1], [new_xmin, p1.ymin], [p1.xmax, p1.ymin],
                               [p1.xmax, p1.ymax], [new_xmin, p1.ymax]]

            pts0_expanded.extend(pts0_5)
            pts1_expanded.extend(pts1_5)
            adjusted_indices.extend([match_idx if needs_adjustment else -1] * 5)

        return pts0_expanded, pts1_expanded, adjusted_indices

    # ------------------------------------------------------------------
    # Core BFMatcher logic
    # ------------------------------------------------------------------

    def _match_bf(self, descs0, descs1):
        """Brute-force match two sets of DINO embeddings. Returns [(i, j, score), ...]."""
        if len(descs0) == 0 or len(descs1) == 0:
            return []

        descs0 = np.asarray(descs0, dtype=np.float32)
        descs1 = np.asarray(descs1, dtype=np.float32)

        if self.cross_check:
            raw_matches = self._bf.match(descs0, descs1)
            result = [(m.queryIdx, m.trainIdx, 1.0 / (1.0 + m.distance))
                      for m in raw_matches]
        else:
            raw_matches = self._bf.knnMatch(descs0, descs1, k=2)
            result = []
            for match_pair in raw_matches:
                if len(match_pair) < 2:
                    if len(match_pair) == 1:
                        m = match_pair[0]
                        result.append((m.queryIdx, m.trainIdx, 1.0 / (1.0 + m.distance)))
                    continue
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    result.append((m.queryIdx, m.trainIdx, 1.0 / (1.0 + m.distance)))

        return result

    # ------------------------------------------------------------------
    # Process pair (mirrors LightGlueMatcher._process_pair)
    # ------------------------------------------------------------------

    def _process_pair(self, i0, i1, pair_idx, detections=None, skip_min_matches_filter=False):
        image0, idx0 = (i0.image, i0.idx) if hasattr(i0, 'image') else (i0, pair_idx)
        image1, idx1 = (i1.image, i1.idx) if hasattr(i1, 'image') else (i1, pair_idx + 1)

        products0 = self._get_products(idx0, detections)
        products1 = self._get_products(idx1, detections)
        logger.info(f"BFMatcher pair {idx0}-{idx1}: {len(products0)} vs {len(products1)} products")

        empty = np.array([], dtype=np.float32).reshape(0, 2)
        if not products0 or not products1:
            logger.warning(f"BFMatcher pair {idx0}-{idx1}: no products, returning empty")
            self.last_matches_info = {}
            return empty, empty

        embeddings0 = np.array(collect_embeddings(products0), dtype=np.float32)
        embeddings1 = np.array(collect_embeddings(products1), dtype=np.float32)
        logger.info(f"BFMatcher pair {idx0}-{idx1}: embeddings {embeddings0.shape} vs {embeddings1.shape}, "
                     f"has_emb0={sum(1 for p in products0 if p.embedding is not None)}/{len(products0)}, "
                     f"has_emb1={sum(1 for p in products1 if p.embedding is not None)}/{len(products1)}")

        if len(embeddings0) == 0 or len(embeddings1) == 0:
            logger.warning(f"BFMatcher pair {idx0}-{idx1}: empty embeddings, returning empty")
            self.last_matches_info = {}
            return empty, empty

        matches = self._match_bf(embeddings0, embeddings1)
        logger.info(f"BFMatcher: {len(matches)} matches from {len(products0)}x{len(products1)} products")

        if len(matches) < self.min_matches and not skip_min_matches_filter:
            self.last_matches_info = {
                'products0': products0, 'products1': products1, 'matches': matches,
            }
            return empty, empty

        h0, w0 = image0.shape[:2]
        h1, w1 = image1.shape[:2]
        image_order = self._determine_image_order(products0, products1, matches, w0, w1)

        pts0_exp, pts1_exp, adj_indices = self._adjust_keypoints_for_partial_products(
            products0, products1, matches, image_order, w0, h0, w1, h1)
        match_indices = [idx if idx >= 0 else (i // 5) for i, idx in enumerate(adj_indices)]
        pts0_all = np.array(pts0_exp, dtype=np.float32)
        pts1_all = np.array(pts1_exp, dtype=np.float32)

        self.last_matches_info = {
            'products0': products0, 'products1': products1, 'matches': matches,
            'inlier_matches': matches, 'num_inliers': len(pts0_all), 'image_order': image_order,
        }

        if len(pts0_all) >= 4:
            H, mask = cv2.findHomography(
                pts0_all, pts1_all, cv2.USAC_MAGSAC,
                ransacReprojThreshold=self.ransac_threshold,
                confidence=0.999, maxIters=5000,
            )
            if H is not None and mask is not None:
                mask = mask.ravel().astype(bool)
                pts0_in, pts1_in = pts0_all[mask], pts1_all[mask]
                inlier_idx = {match_indices[i] for i, m in enumerate(mask) if m}
                outlier_idx = {match_indices[i] for i, m in enumerate(mask) if not m} - inlier_idx
                self.last_matches_info.update({
                    'inlier_matches': [matches[i] for i in range(len(matches)) if i in inlier_idx],
                    'outlier_matches': [matches[i] for i in range(len(matches)) if i in outlier_idx],
                    'num_inliers': len(pts0_in),
                    'num_outliers': len(pts0_all) - len(pts0_in),
                    'inlier_mask': mask,
                    'pts0_all': pts0_all, 'pts1_all': pts1_all,
                })
                return pts0_in, pts1_in

        return pts0_all, pts1_all

    # ------------------------------------------------------------------
    # Public API (same as LightGlueMatcher)
    # ------------------------------------------------------------------

    def inference(self, image_pairs, detections=None, skip_min_matches_filter=False):
        batch_src, batch_tgt = [], []
        for pair_idx, (i0, i1) in enumerate(image_pairs):
            pts0, pts1 = self._process_pair(i0, i1, pair_idx, detections, skip_min_matches_filter)
            batch_src.append(pts0)
            batch_tgt.append(pts1)
        return batch_src, batch_tgt
