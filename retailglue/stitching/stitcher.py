import cv2
import time
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import networkx as nx
from io import BytesIO
from functools import wraps
from collections import defaultdict
from typing import Dict

from retailglue.entities import Polygon, BoundingBox
from retailglue.matchers import create_matcher
from retailglue.stitching.blender import Blender
from retailglue.stitching.transforms import DetectionTransformer

import logging
logger = logging.getLogger("retailglue")


def track_runtime(name=None):
    def decorator(func):
        label = name or func.__name__
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.perf_counter()
            try:
                return func(self, *args, **kwargs)
            finally:
                self.runtimes[label].append(time.perf_counter() - start)
        return wrapper
    return decorator


def visualize_graph(graph, title='Graph'):
    if isinstance(graph, list):
        combined = nx.DiGraph()
        for subgraph in graph:
            combined.add_nodes_from(subgraph.nodes(data=True))
            combined.add_edges_from(subgraph.edges(data=True))
        graph_viz = combined
    else:
        graph_viz = graph

    fig = plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph_viz, seed=42)
    nx.draw_networkx_nodes(graph_viz, pos, node_color='lightblue', node_size=700)
    nx.draw_networkx_labels(graph_viz, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(graph_viz, pos, edge_color='gray', arrows=True, arrowsize=20)
    edge_labels = {(u, v): f"{d['weight']*1000:.4f}" for u, v, d in graph_viz.edges(data=True) if 'weight' in d}
    nx.draw_networkx_edge_labels(graph_viz, pos, edge_labels, font_size=8, font_color='red')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_arr = np.array(Image.open(buf))
    buf.close()
    plt.close(fig)
    return img_arr


class StitchingImage:
    def __init__(self, idx: int, image: np.ndarray):
        self.idx = idx
        self.image = np.array(image) if not isinstance(image, np.ndarray) else image
        self.costs: Dict[int, float] = {}
        self.source_points: Dict[int, np.ndarray] = {}
        self.target_points: Dict[int, np.ndarray] = {}


class ImageStitcher:
    def __init__(self, config, **kwargs):
        self.config = config
        self.device = config.device
        self.verbose = getattr(config, 'verbose', False)
        self.unique_frac_threshold = getattr(config, 'unique_frac_threshold', 0.2)
        self.mask_warp_max_dim = getattr(config, 'mask_warp_max_dim', 2048)
        self.final_image_max_dim = getattr(config, 'final_image_max_dim', 4096)
        self.model_name = config.model_name
        self.model = create_matcher(self.model_name, device=self.device, config=config)
        self.blending = config.blending
        self.blender = Blender(config, runtimes=defaultdict(list))
        self.straightening = config.straightening
        self.detection_transformer = DetectionTransformer(verbose=self.verbose)
        self.iou_threshold = getattr(config, 'iou_threshold', 0.02)
        self.horizontal_overlap_threshold = getattr(config, 'horizontal_overlap_threshold', 0.02)
        self.vertical_separation_threshold = getattr(config, 'vertical_separation_threshold', 0.60)
        self.y_distribution_num_bins = getattr(config, 'y_distribution_num_bins', 3)
        self.y_distribution_min_points_per_bin = getattr(config, 'y_distribution_min_points_per_bin', 5)
        self.y_distribution_min_bins = getattr(config, 'y_distribution_min_bins', 2)
        self.runtimes = defaultdict(list)
        self.blender.runtimes = self.runtimes
        self.detection_transformer.runtimes = self.runtimes
        self.last_keyframes = []
        self.last_directions = []

    def _log_runtime_summary(self):
        if not self.verbose or not self.runtimes:
            return
        lines = []
        for key, values in self.runtimes.items():
            if not values:
                continue
            total = sum(values)
            avg = total / len(values)
            lines.append(f"{key}: count={len(values)} total={total*1000:.1f}ms avg={avg*1000:.1f}ms")
        if lines:
            logger.info("Runtime summary:\n" + "\n".join(lines))

    @track_runtime()
    def get_matching_keypoints(self, batch_pairs, detections=None, skip_min_matches_filter=False):
        from retailglue.matchers.lightglue import LightGlueMatcher
        from retailglue.matchers.bfmatcher import BFMatcher

        if isinstance(self.model, (LightGlueMatcher, BFMatcher)):
            batch_src, batch_tgt = self.model.inference(batch_pairs, detections=detections, skip_min_matches_filter=skip_min_matches_filter)
        else:
            batch_src, batch_tgt = self.model.inference(batch_pairs)

        filtered_src = list(batch_src)
        filtered_tgt = list(batch_tgt)

        if isinstance(self.model, (LightGlueMatcher, BFMatcher)) and hasattr(self.model, 'last_matches_info'):
            info = self.model.last_matches_info
            if 'inlier_mask' in info and 'matches' in info and 'products0' in info and 'products1' in info:
                mask = info['inlier_mask']
                matches = info['matches']
                products0 = info['products0']
                products1 = info['products1']
                rej0, rej1 = [], []
                for match_idx, m in enumerate(matches):
                    p0, p1 = products0[m[0]], products1[m[1]]
                    cx0, cy0 = (p0.xmin + p0.xmax) / 2, (p0.ymin + p0.ymax) / 2
                    cx1, cy1 = (p1.xmin + p1.xmax) / 2, (p1.ymin + p1.ymax) / 2
                    pts0_5 = [[cx0, cy0], [p0.xmin, p0.ymin], [p0.xmax, p0.ymin], [p0.xmax, p0.ymax], [p0.xmin, p0.ymax]]
                    pts1_5 = [[cx1, cy1], [p1.xmin, p1.ymin], [p1.xmax, p1.ymin], [p1.xmax, p1.ymax], [p1.xmin, p1.ymax]]
                    for i in range(5):
                        pt_idx = match_idx * 5 + i
                        if pt_idx < len(mask) and not mask[pt_idx]:
                            rej0.append(pts0_5[i])
                            rej1.append(pts1_5[i])
                rejected_src = [np.array(rej0, dtype=np.float32) if rej0 else np.zeros((0, 2), dtype=np.float32)]
                rejected_tgt = [np.array(rej1, dtype=np.float32) if rej1 else np.zeros((0, 2), dtype=np.float32)]
            else:
                rejected_src = [np.zeros((0, 2), dtype=np.float32) for _ in batch_src]
                rejected_tgt = [np.zeros((0, 2), dtype=np.float32) for _ in batch_tgt]
        else:
            rejected_src = [np.zeros((0, 2), dtype=np.float32) for _ in batch_src]
            rejected_tgt = [np.zeros((0, 2), dtype=np.float32) for _ in batch_tgt]

        for src_pts, tgt_pts, (im1, im2) in zip(filtered_src, filtered_tgt, batch_pairs):
            im1.costs[im2.idx] = 1 / len(src_pts) if len(src_pts) > 0 else 1.0
            im1.source_points[im2.idx] = src_pts
            im1.target_points[im2.idx] = tgt_pts

        return filtered_src, filtered_tgt, rejected_src, rejected_tgt

    @track_runtime()
    def _calculate_homography_matrix(self, source_points, target_points, source_image_shape):
        if len(source_points) < 4 or len(target_points) < 4:
            return None
        H, _ = cv2.findHomography(source_points, target_points, cv2.USAC_MAGSAC,
                                  ransacReprojThreshold=10.0, confidence=0.999, maxIters=5000)
        msgs = self._analyze_homography_matrix(H, source_image_shape)
        if msgs:
            if self.verbose:
                logger.warning(msgs)
            return None
        return H

    @track_runtime()
    def _transform(self, homography, coords):
        x1, y1, x2, y2 = coords
        pts = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
        new_coords = cv2.perspectiveTransform(np.float32([pts]), homography)[0]
        return new_coords.astype(int)

    @track_runtime()
    def _calculate_a_to_b_matrix(self, traverse_graph, idx1, idx2, source_image_shape):
        try:
            path = nx.shortest_path(traverse_graph, idx1, idx2, weight='weight')
        except nx.NetworkXNoPath:
            try:
                path = nx.shortest_path(traverse_graph.to_undirected(), idx1, idx2, weight='weight')
            except nx.NetworkXNoPath:
                return None

        H = np.eye(3)
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            if traverse_graph.has_edge(s, t):
                M = traverse_graph[s][t]['homography_matrix']
            else:
                M = np.linalg.inv(traverse_graph[t][s]['homography_matrix'])
            H = np.dot(M, H)

        msgs = self._analyze_homography_matrix(H, source_image_shape)
        if msgs:
            if self.verbose:
                logger.warning(f'Path {" -> ".join(map(str, path))} produces bad cumulative homography.')
            return None
        return H

    def _convert_arrays_to_stitching_images(self, input_images):
        return [StitchingImage(i, img) for i, img in enumerate(input_images)]

    @track_runtime()
    def _calculate_final_stitched_image_size(self, images, traverse_graph, center_image_id):
        center_img = images[center_image_id]
        boxes = [[0, 0, center_img.image.shape[1], center_img.image.shape[0]]]

        for src_img in images:
            if src_img.idx == center_img.idx or src_img.idx not in traverse_graph.nodes:
                continue
            H = self._calculate_a_to_b_matrix(traverse_graph, src_img.idx, center_img.idx, src_img.image.shape)
            if H is None:
                continue
            coords = [0, 0, src_img.image.shape[1], src_img.image.shape[0]]
            new_coords = self._transform(homography=H, coords=coords)
            xmin, xmax = new_coords[:, 0].min(), new_coords[:, 0].max()
            ymin, ymax = new_coords[:, 1].min(), new_coords[:, 1].max()
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = np.array(boxes)
        xmin, ymin = boxes[:, 0].min(), boxes[:, 1].min()
        xmax, ymax = boxes[:, 2].max(), boxes[:, 3].max()
        new_size = (int(xmax - xmin), int(ymax - ymin))
        translation = np.array([[1, 0, abs(xmin)], [0, 1, abs(ymin)], [0, 0, 1]], dtype=np.float32)
        return new_size, translation

    @track_runtime()
    def _analyze_homography_matrix(self, H, shape):
        errors = []
        if H is None:
            return ["Homography matrix is None"]

        rotation_angle = abs(int(np.degrees(np.arctan2(H[0, 1], H[0, 0]))))
        max_rotation = self.config.max_allowed_rotation_angle
        if rotation_angle >= max_rotation:
            errors.append(f"Rotation angle too big: {rotation_angle}°")

        old_box = BoundingBox(0, 0, shape[1], shape[0])
        new_points = self._transform(H, list(old_box.get_pascal_voc_format()))
        old_area = old_box.get_area()
        new_poly = Polygon(new_points)
        new_area = new_poly.get_area()
        scale = new_area / old_area

        if not (0.1 < scale < 10.0):
            errors.append(f"Scaling ratio out of range: {scale:.1f}")

        convex = cv2.isContourConvex(np.array([pt.as_list() for pt in new_poly.points]))
        if not convex:
            errors.append("Perspective distortion too high")
        return errors

    @track_runtime()
    def straightening_panorama(self, image):
        h_orig, w_orig = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, None

        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        if len(approx) < 4:
            return image, None

        points = approx.reshape(-1, 2).astype(np.float32)
        corner_configs = [
            ('top_left', points[:, 0] + points[:, 1], False),
            ('top_right', points[:, 0] - points[:, 1], True),
            ('bottom_right', points[:, 0] + points[:, 1], True),
            ('bottom_left', points[:, 0] - points[:, 1], False),
        ]

        selected_indices = []
        corners = {}
        for corner_name, scores, descending in corner_configs:
            sorted_idx = np.argsort(scores)
            if descending:
                sorted_idx = sorted_idx[::-1]
            sel = None
            for idx in sorted_idx:
                if idx not in selected_indices:
                    sel = idx
                    break
            if sel is None:
                return image, None
            selected_indices.append(sel)
            corners[corner_name] = points[sel]

        src_pts = np.float32([corners['top_left'], corners['top_right'], corners['bottom_right'], corners['bottom_left']])
        lt, rt, rb, lb = src_pts
        left_angle = abs(np.degrees(np.arctan2(lb[1] - lt[1], lb[0] - lt[0])))
        right_angle = abs(np.degrees(np.arctan2(rb[1] - rt[1], rb[0] - rt[0])))
        if left_angle < 0.2 and right_angle < 0.2:
            return image, None

        w_top = np.linalg.norm(rt - lt)
        w_bot = np.linalg.norm(rb - lb)
        contour_w = int(max(w_top, w_bot))
        h_left = np.linalg.norm(lb - lt)
        h_right = np.linalg.norm(rb - rt)
        contour_h = int(max(h_left, h_right))

        max_w = max(contour_w, w_orig)
        max_h = max(contour_h, h_orig)
        w_ratio = max_w / w_orig
        h_ratio = max_h / h_orig
        if w_ratio < 0.5 or w_ratio > 2.0 or h_ratio < 0.5 or h_ratio > 2.0:
            return image, None

        cx, cy = max_w // 2, max_h // 2
        hw, hh = contour_w // 2, contour_h // 2
        dst_pts = np.float32([
            [cx - hw, cy - hh], [cx + hw, cy - hh],
            [cx + hw, cy + hh], [cx - hw, cy + hh]
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(image, M, (max_w, max_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return result, M

    @track_runtime()
    def patch_images(self, img_stitched, stitch_image, homography_matrix):
        warped = cv2.warpPerspective(stitch_image, homography_matrix, (img_stitched.shape[1], img_stitched.shape[0]))
        if not self.blending:
            mask2 = np.any(warped, axis=2).astype(np.uint8)
            result = img_stitched.copy()
            result[mask2 > 0] = warped[mask2 > 0]
            return result

        mask1, mask2, overlap, warped = self.blender.build_masks_and_exposure(img_stitched, warped)
        if np.count_nonzero(overlap) == 0:
            result = img_stitched.copy()
            result[mask2 > 0] = warped[mask2 > 0]
            return result

        blend_region = self.blender.extract_blend_region_and_weights(img_stitched, warped, (mask1, mask2, overlap), pad=8)
        result = img_stitched.copy()
        blended = self.blender.adaptive_blend(blend_region)
        y0, y1, x0, x1 = blend_region.bounds
        result[y0:y1, x0:x1] = blended
        only2 = (mask2 > 0) & (mask1 == 0)
        result[only2] = warped[only2]
        return result

    def _add_edge_with_limit(self, graph, source, target, cost, homography_matrix):
        max_edges = 4
        inv = np.linalg.inv(homography_matrix)

        source_edges = list(graph.out_edges(source, data=True))
        if source_edges:
            min_cost = min(e[2]['weight'] for e in source_edges)
            if cost > 5 * min_cost:
                if self.verbose:
                    logger.warning(f"Skipped edge {source} -> {target} (cost {cost:.4f}), "
                                   f"below 20% of best edge matches for node {source} (min_cost: {min_cost:.4f})")
                return

        if len(source_edges) >= max_edges:
            max_cost_edge = max(source_edges, key=lambda x: x[2]['weight'])
            if cost >= max_cost_edge[2]['weight']:
                if self.verbose:
                    logger.info(f"Skipped edge {source} -> {target} (cost {cost:.4f}), "
                                f"existing edges have lower costs. max cost: {max_cost_edge[2]['weight']:.4f}")
                return

        while len(source_edges) >= max_edges:
            worst = max(source_edges, key=lambda x: x[2]['weight'])
            graph.remove_edge(worst[0], worst[1])
            if self.verbose:
                logger.info(f"Removed edge {worst[0]} -> {worst[1]} (cost {worst[2]['weight']:.4f}) to make room")
            source_edges = list(graph.out_edges(source, data=True))

        target_edges = list(graph.out_edges(target, data=True))
        while len(target_edges) >= max_edges:
            worst = max(target_edges, key=lambda x: x[2]['weight'])
            graph.remove_edge(worst[0], worst[1])
            if self.verbose:
                logger.info(f"Removed edge {worst[0]} -> {worst[1]} (cost {worst[2]['weight']:.4f}) to make room for inverse")
            target_edges = list(graph.out_edges(target, data=True))

        graph.add_edge(source, target, weight=cost, homography_matrix=homography_matrix)
        graph.add_edge(target, source, weight=cost, homography_matrix=inv)
        if self.verbose:
            logger.info(f"Added edge {source} -> {target} with cost {cost:.4f}")
            logger.info(f"Added inverse edge {target} -> {source} with cost {cost:.4f}")

    @track_runtime()
    def _check_y_axis_keypoint_distribution(self, source_points, target_points,
                                            num_bins=3, min_points_per_bin=5, min_bins_with_points=2):
        if len(source_points) == 0 or len(target_points) == 0:
            return False

        def check_dist(y_coords):
            if len(y_coords) < min_points_per_bin:
                return True
            y_min, y_max = y_coords.min(), y_coords.max()
            h = y_max - y_min
            if h <= 0:
                return True
            y_norm = (y_coords - y_min) / h
            bin_counts, _ = np.histogram(y_norm, bins=num_bins, range=(0.0, 1.0))
            active = np.sum(bin_counts >= min_points_per_bin)
            max_gap = np.max(np.diff(np.sort(y_norm))) if len(y_norm) > 1 else 0.0
            return active >= min_bins_with_points and max_gap <= 0.5

        return check_dist(source_points[:, 1]) and check_dist(target_points[:, 1])

    @track_runtime()
    def _calculate_graph(self, images, detections=None):
        from retailglue.matchers.lightglue import LightGlueMatcher

        graph = nx.DiGraph()
        graph.add_nodes_from([i.idx for i in images])

        pairs = [(im1, im2) for im1 in images for im2 in images if im2.idx > im1.idx]
        batch_src, batch_tgt, _, _ = self.get_matching_keypoints(
            pairs, detections=detections)

        for src_pts, tgt_pts, (im1, im2) in zip(batch_src, batch_tgt, pairs):
            cost = im1.costs[im2.idx]
            if isinstance(self.model, LightGlueMatcher):
                min_thresh = getattr(self.config, 'lightglue_min_matching_threshold', 4)
            else:
                min_thresh = self.config.min_matching_threshold

            if cost < 1 / min_thresh:
                if not isinstance(self.model, LightGlueMatcher):
                    if not self._check_y_axis_keypoint_distribution(
                            src_pts, tgt_pts,
                            num_bins=self.y_distribution_num_bins,
                            min_points_per_bin=self.y_distribution_min_points_per_bin,
                            min_bins_with_points=self.y_distribution_min_bins):
                        continue

                H = self._calculate_homography_matrix(src_pts, tgt_pts, im1.image.shape)
                if H is not None:
                    if self.verbose:
                        logger.info(f"Image {im1.idx} and {im2.idx} have enough matching points. {len(src_pts)} vs {len(tgt_pts)}")
                    self._add_edge_with_limit(graph, im1.idx, im2.idx, cost, H)
                else:
                    if self.verbose:
                        logger.warning(f"Image {im1.idx} and {im2.idx} not good homography matrix. {len(src_pts)} vs {len(tgt_pts)}")
            else:
                if self.verbose:
                    logger.warning(f"Image {im1.idx} and {im2.idx} dont have enough matching points. {len(src_pts)} vs {len(tgt_pts)}")

        return graph

    @track_runtime()
    def _calculate_subgraphs(self, graph, images):
        components = sorted(nx.connected_components(graph.to_undirected()), key=lambda c: min(c))
        subgraphs = []
        for comp in components:
            subg = graph.subgraph(comp).copy()
            if subg.number_of_nodes() <= 1:
                subgraphs.append(subg)
                continue
            subgraphs.extend(self._refine_subgraph_by_homography(subg, images))
        return subgraphs

    @track_runtime()
    def _refine_subgraph_by_homography(self, subg, images):
        if subg.number_of_nodes() <= 1:
            return [subg]

        center = self._find_center_image(subg)
        invalid_nodes = []
        for node in subg.nodes:
            if node == center:
                continue
            try:
                H = self._calculate_a_to_b_matrix(subg, node, center, images[node].image.shape)
            except Exception:
                H = None
            if H is None:
                invalid_nodes.append(node)

        if not invalid_nodes:
            return [subg]

        g_mut = subg.copy()
        for node in invalid_nodes:
            if node not in g_mut or center not in g_mut:
                continue
            try:
                path = nx.shortest_path(g_mut.to_undirected(), node, center, weight='weight')
            except nx.NetworkXNoPath:
                continue
            if len(path) < 2:
                continue
            worst_edge, worst_weight = None, -1.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                w = None
                if g_mut.has_edge(u, v):
                    w = g_mut[u][v].get('weight')
                elif g_mut.has_edge(v, u):
                    w = g_mut[v][u].get('weight')
                if w is not None and w > worst_weight:
                    worst_weight = w
                    worst_edge = (u, v)
            if worst_edge:
                u, v = worst_edge
                if g_mut.has_edge(u, v):
                    g_mut.remove_edge(u, v)
                if g_mut.has_edge(v, u):
                    g_mut.remove_edge(v, u)

        refined = []
        for comp in nx.connected_components(g_mut.to_undirected()):
            sub = g_mut.subgraph(comp).copy()
            if sub.number_of_nodes() <= 1:
                refined.append(sub)
            else:
                refined.extend(self._refine_subgraph_by_homography(sub, images))
        return refined

    @track_runtime()
    def stitch_images(self, input_images, detections=None):
        self.runtimes.clear()
        input_images = [np.array(img) for img in input_images]

        has_dets = detections is not None
        if has_dets:
            input_images, detections, kept_indices = \
                self._filter_very_similar_input_images(input_images, detections)
        else:
            input_images, kept_indices = self._filter_very_similar_input_images(input_images)

        self.kept_indices = kept_indices

        if len(input_images) == 1:
            self._log_runtime_summary()
            if has_dets:
                return (input_images, detections, []) if self.verbose else \
                       (input_images, detections)
            return input_images

        images = self._convert_arrays_to_stitching_images(input_images)
        input_graph = self._calculate_graph(
            images,
            detections=detections if has_dets else None)
        filtered_graph = self._frame_eliminator(input_graph.copy(), images)

        viz_graph = visualize_graph(input_graph, 'Input Graph') if self.verbose else None
        viz_filtered = visualize_graph(filtered_graph, 'Filtered Graph') if self.verbose else None

        subgraphs = self._calculate_subgraphs(filtered_graph, images)
        viz_subgraphs = visualize_graph(subgraphs, f'All ({len(subgraphs)} Subgraphs)') if self.verbose else None

        if self.verbose:
            logger.info(f"Found {len(subgraphs)} subgraphs")
            for si, sg in enumerate(subgraphs):
                logger.info(f"Subgraph {si}: {sorted(sg.nodes())}")

        stitched_results = []
        transformed_det_results = []
        self.straightening_matrices = {}
        self.panorama_image_indices = {}
        self.panorama_original_indices = {}
        self.panorama_subgraphs = {}
        self.panorama_translation_matrices = {}
        self.panorama_center_image_ids = {}
        self.panorama_patch_scales = {}

        for traverse_graph in subgraphs:
            pano_idx = len(stitched_results)
            filtered_indices = sorted(traverse_graph.nodes())
            self.panorama_image_indices[pano_idx] = filtered_indices
            self.panorama_original_indices[pano_idx] = [self.kept_indices[i] for i in filtered_indices]
            self.panorama_subgraphs[pano_idx] = traverse_graph

            if traverse_graph.number_of_nodes() == 1:
                only_node = next(iter(traverse_graph.nodes))
                stitched_results.append(images[only_node].image)
                self.straightening_matrices[pano_idx] = None
                self.panorama_translation_matrices[pano_idx] = np.eye(3, dtype=np.float32)
                self.panorama_center_image_ids[pano_idx] = only_node
                self.panorama_patch_scales[pano_idx] = 1.0
                if has_dets:
                    transformed_det_results.append(detections[only_node])
                continue

            center_id = self._find_center_image(traverse_graph)
            new_size, translation = self._calculate_final_stitched_image_size(images, traverse_graph, center_id)
            self.panorama_center_image_ids[pano_idx] = center_id
            self.panorama_translation_matrices[pano_idx] = translation

            max_dim = max(new_size) if new_size else 0
            patch_scale = 1.0
            if self.final_image_max_dim and max_dim > self.final_image_max_dim:
                patch_scale = min(self.final_image_max_dim / float(max_dim), 1.0)
            self.panorama_patch_scales[pano_idx] = patch_scale

            scaled_size = (max(1, int(np.ceil(new_size[0] * patch_scale))),
                           max(1, int(np.ceil(new_size[1] * patch_scale))))
            scale_mat = np.array([[patch_scale, 0, 0], [0, patch_scale, 0], [0, 0, 1]], dtype=np.float32)
            inv_scale = 1 / patch_scale
            scale_mat_inv = np.array([[inv_scale, 0, 0], [0, inv_scale, 0], [0, 0, 1]], dtype=np.float32)
            pre_scale_translation = scale_mat @ translation
            center_transform = pre_scale_translation @ scale_mat_inv

            scaled_cache = {}

            def get_scaled(idx):
                if patch_scale >= 1.0:
                    return images[idx].image
                if idx not in scaled_cache:
                    src = images[idx].image
                    h, w = src.shape[:2]
                    scaled_cache[idx] = cv2.resize(
                        src, (max(1, int(np.ceil(w * patch_scale))), max(1, int(np.ceil(h * patch_scale)))),
                        interpolation=cv2.INTER_AREA)
                return scaled_cache[idx]

            center_image = get_scaled(center_id)
            img_stitched = cv2.warpPerspective(center_image, center_transform, scaled_size)

            stitch_order = list(nx.bfs_edges(traverse_graph, source=center_id))
            nodes_ordered = set([t for _, t in stitch_order] + [center_id])
            if set(traverse_graph.nodes) - nodes_ordered:
                stitch_order = list(nx.bfs_edges(traverse_graph.to_undirected(), source=center_id))

            if self.verbose:
                logger.info(f"Stitch order centered on {center_id}: {stitch_order}")

            det_per_image = []

            if has_dets:
                det_per_image.append((center_id, self.detection_transformer.transform_detections(detections[center_id], pre_scale_translation)))

            for _, src_idx in stitch_order:
                src_img = get_scaled(src_idx)
                H = self._calculate_a_to_b_matrix(traverse_graph, src_idx, center_id, images[src_idx].image.shape)
                if H is None:
                    continue

                total_H = pre_scale_translation @ H @ scale_mat_inv
                total_H_dets = pre_scale_translation @ H

                img_stitched = self.patch_images(img_stitched, src_img, total_H)

                if has_dets:
                    det_per_image.append((src_idx, self.detection_transformer.transform_detections(detections[src_idx], total_H_dets)))

            if self.straightening:
                img_stitched, str_matrix = self.straightening_panorama(img_stitched)
                self.straightening_matrices[pano_idx] = str_matrix
                if has_dets and str_matrix is not None:
                    det_per_image = [(idx, self.detection_transformer.transform_detections(dets, str_matrix))
                                     for idx, dets in det_per_image]
            else:
                self.straightening_matrices[pano_idx] = None

            stitched_results.append(img_stitched)
            if has_dets:
                final_dets, _ = self.detection_transformer.remove_duplicate_detections(det_per_image, stitch_order)
                transformed_det_results.append(final_dets)

        self._log_runtime_summary()

        if has_dets:
            if self.verbose:
                return stitched_results, transformed_det_results, \
                       [viz_graph, viz_filtered, viz_subgraphs]
            return stitched_results, transformed_det_results
        return (stitched_results, [viz_graph, viz_filtered, viz_subgraphs]) if self.verbose else stitched_results

    @track_runtime()
    def _find_center_image(self, subg):
        if subg.number_of_nodes() == 1:
            return next(iter(subg.nodes))
        undirected = subg.to_undirected()
        try:
            closeness = nx.closeness_centrality(undirected, distance='weight')
            center = max(closeness.items(), key=lambda x: x[1])
            if self.verbose:
                logger.info(f"Center by closeness centrality: {center[0]} (score: {center[1]:.4f})")
            return center[0]
        except Exception:
            degree_dict = dict(undirected.degree())
            return max(degree_dict.items(), key=lambda x: x[1])[0]

    @track_runtime()
    def _frame_eliminator(self, graph, images):
        if graph.number_of_nodes() < 3:
            return graph

        components = list(nx.connected_components(graph.to_undirected()))
        removed_overall = set()

        for comp in components:
            subg = graph.subgraph(comp).copy()
            if subg.number_of_nodes() < 3:
                continue

            center_id = self._find_center_image(subg)
            if self.verbose:
                logger.warning(f"Center image id: {center_id}")
            new_size, translation = self._calculate_final_stitched_image_size(images, subg, center_id)
            bridge_candidates = []

            for j in sorted(subg.nodes()):
                neighbors = list(subg.neighbors(j))
                if len(neighbors) < 2:
                    continue

                max_dim = max(new_size) if new_size else 0
                warp_scale = 1.0
                if self.mask_warp_max_dim and max_dim > self.mask_warp_max_dim:
                    warp_scale = min(self.mask_warp_max_dim / float(max_dim), 1.0)

                scaled_size = (max(1, int(np.ceil(new_size[0] * warp_scale))),
                               max(1, int(np.ceil(new_size[1] * warp_scale))))
                output_scale = np.array([[warp_scale, 0, 0], [0, warp_scale, 0], [0, 0, 1]], dtype=np.float32)

                mask_cache = {}

                def warp_mask(node_id):
                    if node_id in mask_cache:
                        return mask_cache[node_id]
                    if node_id == center_id:
                        H_tot = translation.astype(np.float32, copy=False)
                    else:
                        H = self._calculate_a_to_b_matrix(subg, node_id, center_id, images[node_id].image.shape)
                        if H is None:
                            mask_cache[node_id] = None
                            return None
                        H_tot = np.dot(translation, H).astype(np.float32, copy=False)
                    H_eff = output_scale @ H_tot
                    src_mask = np.any(images[node_id].image, axis=2).astype(np.uint8)
                    warped = cv2.warpPerspective(src_mask, H_eff, scaled_size).astype(np.uint8)
                    mask_cache[node_id] = warped
                    return warped

                for ni, nk in itertools.combinations(neighbors, 2):
                    if not (subg.has_edge(ni, nk) and subg.has_edge(ni, j) and subg.has_edge(j, nk)):
                        continue

                    m_ni, m_j, m_nk = warp_mask(ni), warp_mask(j), warp_mask(nk)
                    if m_ni is None or m_j is None or m_nk is None:
                        continue

                    mask_ni, mask_j, mask_nk = m_ni > 0, m_j > 0, m_nk > 0
                    j_total = mask_j.sum()
                    union_ni_nk = mask_ni | mask_nk
                    intersection = mask_ni & mask_nk
                    iou = intersection.sum() / union_ni_nk.sum()
                    if iou < 0.1:
                        continue

                    j_unique = mask_j & (~union_ni_nk)
                    unique_frac = float(j_unique.sum() / j_total) if j_total > 0 else 0.0
                    if unique_frac > self.unique_frac_threshold:
                        if self.verbose:
                            logger.warning(f"Bridge j={j} between ni={ni}, nk={nk}: iou_ni_nk={iou:.3f}, unique_frac={unique_frac:.3f} > {self.unique_frac_threshold}, skipping")
                        continue

                    if self.verbose:
                        logger.error(f"Bridge j={j} between ni={ni}, nk={nk}: iou_ni_nk={iou:.3f}, unique_frac={unique_frac:.3f} ")
                    bridge_candidates.append((j, ni, nk, unique_frac))

            triangle_bridges = {}
            for j, ni, nk, uf in bridge_candidates:
                tri = frozenset({j, ni, nk})
                triangle_bridges.setdefault(tri, []).append((j, uf))

            nodes_to_remove = set()
            for tri, bridges in triangle_bridges.items():
                if any(n in nodes_to_remove for n in tri):
                    continue
                if len(bridges) <= 1:
                    node, uf = bridges[0]
                    if uf < self.unique_frac_threshold:
                        nodes_to_remove.add(node)
                else:
                    bridges.sort(key=lambda x: x[1])
                    lowest_node, lowest_uf = bridges[0]
                    if lowest_uf < self.unique_frac_threshold:
                        nodes_to_remove.add(lowest_node)

            for node in nodes_to_remove:
                if node in graph.nodes():
                    graph.remove_node(node)
                    removed_overall.add(node)

        if self.verbose:
            if removed_overall:
                logger.warning(f"Pruned bridge nodes: {sorted(removed_overall)}")
            else:
                logger.info(f"No bridge nodes pruned!")

        return graph

    @track_runtime()
    def _filter_very_similar_input_images(self, images, detections=None,
                                          hash_size=8, max_distance=5):
        kept_images, kept_dets, kept_indices, hashes = [], [], [], []
        for idx, img in enumerate(images):
            resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_NEAREST)
            diff = resized[:, 1:] > resized[:, :-1]
            h = int(np.packbits(diff.flatten(), bitorder='big').tobytes().hex(), 16)
            if any(((h ^ prev).bit_count() <= max_distance) for prev in hashes):
                continue
            kept_images.append(img)
            kept_indices.append(idx)
            if detections:
                kept_dets.append(detections[idx])
            hashes.append(h)

        if detections:
            return kept_images, kept_dets, kept_indices
        return kept_images, kept_indices
