import cv2
import numpy as np
import lap
import networkx as nx

from shapely.strtree import STRtree

from retailglue.entities import Polygon, Point


def create_iou_matrix(stitched_detections, single_polygons):
    stitched_polygons = [Polygon(box.corners) for box in stitched_detections]
    single_tree = STRtree([p.geometric for p in single_polygons])
    ious = np.zeros((len(stitched_detections), len(single_polygons)))
    for i, sp in enumerate(stitched_polygons):
        hits = single_tree.query(sp.geometric, predicate="intersects")
        for j in hits:
            ious[i][j] = sp.get_iou_score(single_polygons[j])
    return ious


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    return np.asarray(matches) if matches else np.empty((0, 2), dtype=int), unmatched_a, unmatched_b


def transform_coords(homography, coords):
    if len(coords) == 0:
        return np.empty((0, 1, 2), dtype=np.float32)
    arr = np.array(coords, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, homography)


def get_homography_matrix(metadata, image_idx):
    homography_matrices = metadata.get('homography_matrices', {})

    if isinstance(homography_matrices, dict) and image_idx in homography_matrices:
        return homography_matrices[image_idx]

    center_image_id = None
    translation_matrix = None
    traverse_graph = None
    patch_scale = metadata.get('patch_scale', 1.0)

    if isinstance(homography_matrices, dict) and '_traverse_graph' in homography_matrices:
        center_image_id = homography_matrices.get('_center_image_id')
        translation_matrix = homography_matrices.get('_translation_matrix')
        traverse_graph = homography_matrices.get('_traverse_graph')
    else:
        center_image_id = metadata.get('_center_image_id')
        translation_matrix = metadata.get('_translation_matrix')
        traverse_graph = metadata.get('_traverse_graph')

    if traverse_graph is None or center_image_id is None:
        return np.eye(3)

    scale_matrix = np.array([[patch_scale, 0, 0], [0, patch_scale, 0], [0, 0, 1]], dtype=np.float32)
    pre_scale_translation = scale_matrix @ translation_matrix if translation_matrix is not None else scale_matrix

    if image_idx == center_image_id:
        return pre_scale_translation

    try:
        path = nx.shortest_path(traverse_graph, image_idx, center_image_id)
        H = np.eye(3)
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            edge = traverse_graph[s][t]
            M = edge.get('homography_matrix', edge.get('transformation_matrix', np.eye(3)))
            H = np.dot(M, H)
        return pre_scale_translation @ H
    except Exception:
        return np.eye(3)


def transform_image_corners(metadata, image_dict, image_indices=None):
    result = {}
    straightening_matrix = metadata.get('straightening_matrix', None)
    for i, image in enumerate(image_dict):
        idx = image_indices[i] if image_indices is not None else i
        corners = [[0, 0], [0, image.shape[0]], [image.shape[1], image.shape[0]], [image.shape[1], 0]]
        H = get_homography_matrix(metadata, idx)
        new_corners = transform_coords(H, [corners])[0]
        if straightening_matrix is not None:
            new_corners = transform_coords(straightening_matrix, [new_corners])[0]
        result[idx] = Polygon(new_corners)
    return result


def get_transformed_polygons(metadata, detections_list, image_indices=None):
    result = {}
    for i, detections in enumerate(detections_list):
        idx = image_indices[i] if image_indices is not None else i
        if not detections:
            result[idx] = []
            continue
        all_corners = []
        for det in detections:
            for pt in det.corners:
                all_corners.append(pt.as_list())
        if not all_corners:
            result[idx] = []
            continue
        H = get_homography_matrix(metadata, idx)
        transformed = transform_coords(H, all_corners)
        reshaped = transformed.reshape(len(detections), 4, 2)
        result[idx] = [Polygon(c.squeeze()) for c in reshaped]
    return result


def apply_straightening_to_polygons(polygons, straightening_matrix):
    if straightening_matrix is None or not polygons:
        return polygons
    all_corners = []
    counts = []
    for p in polygons:
        corners = [pt.as_list() for pt in p.points]
        all_corners.extend(corners)
        counts.append(len(corners))
    if not all_corners:
        return polygons
    transformed = transform_coords(straightening_matrix, all_corners).squeeze()
    result = []
    idx = 0
    for count in counts:
        pts = [Point(int(x), int(y)) for x, y in transformed[idx:idx + count]]
        result.append(Polygon(pts))
        idx += count
    return result


def reconstruct_metadata(stitcher, sequence_images):
    images = stitcher._convert_arrays_to_stitching_images(sequence_images)
    graph = stitcher._calculate_graph(images)
    filtered_graph = stitcher._frame_eliminator(graph.copy(), images)
    subgraphs = stitcher._calculate_subgraphs(filtered_graph, images)

    if not subgraphs:
        return [{'_traverse_graph': graph, '_raw_stitch_order': [],
                 '_translation_matrix': np.eye(3), '_center_image_id': None, '_new_size': None}]

    metadata_list = []
    for traverse_graph in subgraphs:
        if traverse_graph.number_of_nodes() == 0:
            metadata_list.append({'_traverse_graph': traverse_graph, '_raw_stitch_order': [],
                                  '_translation_matrix': np.eye(3), '_center_image_id': None, '_new_size': None})
            continue

        center_id = stitcher._find_center_image(traverse_graph)
        new_size, translation = stitcher._calculate_final_stitched_image_size(images, traverse_graph, center_id)
        stitch_order = list(nx.bfs_edges(traverse_graph, source=center_id))

        max_dim = max(new_size) if new_size else 0
        final_max = getattr(stitcher, 'final_image_max_dim', None)
        patch_scale = 1.0
        if final_max and max_dim > final_max:
            patch_scale = min(final_max / float(max_dim), 1.0)

        metadata_list.append({
            '_traverse_graph': traverse_graph, '_raw_stitch_order': stitch_order,
            '_translation_matrix': translation, '_center_image_id': center_id,
            '_new_size': new_size, '_patch_scale': patch_scale})
    return metadata_list
