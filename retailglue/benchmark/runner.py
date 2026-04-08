import gc
import glob
import os
import copy
import logging

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import networkx as nx
import torch
from PIL import Image
from shapely.errors import GEOSException
from shapely.strtree import STRtree

from retailglue.benchmark.constants import BENCHMARK_DATA_ROOT, COMBINATIONS
from retailglue.benchmark.drawer import (
    draw_deleted_polygons, draw_final_polygons, draw_unmatched,
    draw_stitched_image_detections, draw_stitched_with_detections,
)
from retailglue.benchmark.stats import calculate_stats, save_stats, wrap_results
from retailglue.benchmark.evaluation import (
    linear_assignment, create_iou_matrix, transform_image_corners,
    get_transformed_polygons, apply_straightening_to_polygons, reconstruct_metadata,
)
from retailglue.config import get_config, resolve_path
from retailglue.detector import YOLODetector
from retailglue.embeddings import extract_dino_embeddings, DINO_VARIANTS, BF_TO_DINO_VARIANT
from retailglue.entities import Polygon, Point
from retailglue.io import read_image
from retailglue.stitching.stitcher import ImageStitcher

logger = logging.getLogger("retailglue")

DINO_MODEL_NAMES = tuple(DINO_VARIANTS.keys()) + tuple(BF_TO_DINO_VARIANT.keys())


def _clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_sequences(data_root=None):
    data_root = data_root or BENCHMARK_DATA_ROOT
    images_dir = os.path.join(data_root, 'images')
    if not os.path.isdir(images_dir) or not os.listdir(images_dir):
        raise FileNotFoundError(f"Benchmark images not found in {images_dir}. "
                                "Please place benchmark images in data/benchmark/images/")
    names = [f for f in os.listdir(images_dir) if f.endswith('.jpg') and not f.startswith('.')]
    return sorted(set(n.split('_')[0] for n in names))


def create_stitcher(combination, config=None):
    if config is None:
        config = get_config()
    cfg = copy.copy(config.stitching)
    cfg.model_name = combination["model_name"]
    cfg.device = combination["device"]
    return ImageStitcher(config=cfg)


def create_sku_yolo_detector(weights_path, device='cpu'):
    return YOLODetector(weights_path, device=device)


def get_sku_yolo_detections(detector, image):
    dets = detector.detect([image], conf=0.25)
    return dets[0]


def run_detection_on_batch(detector, images, conf=0.30):
    return detector.detect(images, conf=conf)


def _get_stitcher_metadata(stitcher):
    """Read cached metadata from stitcher instead of re-running matching."""
    metadata_list = []
    subgraphs = getattr(stitcher, 'panorama_subgraphs', {})
    for pano_idx in sorted(subgraphs.keys()):
        traverse_graph = subgraphs[pano_idx]
        center_id = stitcher.panorama_center_image_ids.get(pano_idx)
        translation = stitcher.panorama_translation_matrices.get(pano_idx, np.eye(3))
        patch_scale = stitcher.panorama_patch_scales.get(pano_idx, 1.0)
        image_indices = stitcher.panorama_image_indices.get(pano_idx, [])
        stitch_order = []
        if traverse_graph is not None and traverse_graph.number_of_nodes() > 1 and center_id is not None:
            stitch_order = list(nx.bfs_edges(traverse_graph, source=center_id))
        metadata_list.append({
            '_traverse_graph': traverse_graph,
            '_raw_stitch_order': stitch_order,
            '_translation_matrix': translation,
            '_center_image_id': center_id,
            '_patch_scale': patch_scale,
            '_image_indices': image_indices,
        })
    return metadata_list


def get_stitched_image_and_detections(sequence_images, stitcher, sku_yolo_detector,
                                       model_name=None, save_path=None, config=None):
    detections = None

    if model_name in DINO_MODEL_NAMES:
        detections = run_detection_on_batch(sku_yolo_detector, sequence_images)

        dino_variant = BF_TO_DINO_VARIANT.get(model_name, model_name)
        dino_weights = None
        crop_margin = 10
        if config and hasattr(config, 'embeddings'):
            emb_cfg = config.embeddings
            if hasattr(emb_cfg, 'dino_weights'):
                dino_weights = getattr(emb_cfg.dino_weights, dino_variant, None)
                if dino_weights:
                    dino_weights = resolve_path(dino_weights)
            crop_margin = getattr(emb_cfg, 'crop_margin', 10)
        extract_dino_embeddings(sequence_images, detections, stitcher.device,
                                variant=model_name, weights_path=dino_weights,
                                crop_margin=crop_margin)

    stitch_result = stitcher.stitch_images(sequence_images, detections=detections)

    if isinstance(stitch_result, tuple):
        stitched_images, det_list = stitch_result
    else:
        stitched_images = stitch_result
        det_list = []

    reconstruct_metas = _get_stitcher_metadata(stitcher)
    if not isinstance(stitched_images, list):
        stitched_images = [stitched_images]
    if not isinstance(reconstruct_metas, list):
        reconstruct_metas = [reconstruct_metas]
    while len(reconstruct_metas) < len(stitched_images):
        reconstruct_metas.append(reconstruct_metas[0] if reconstruct_metas else {})
    while len(det_list) < len(stitched_images):
        det_list.append([])

    results = []
    for idx, (img, meta) in enumerate(zip(stitched_images, reconstruct_metas)):
        stitched_dets = get_sku_yolo_detections(sku_yolo_detector, img)
        stitch_order = meta.get('_raw_stitch_order', [])
        straightening_matrices = getattr(stitcher, 'straightening_matrices', {})
        patch_scale = meta.get('_patch_scale', 1.0)

        metadata = {
            "homography_matrices": {
                '_traverse_graph': meta.get('_traverse_graph'),
                '_translation_matrix': meta.get('_translation_matrix'),
                '_center_image_id': meta.get('_center_image_id'),
            },
            "stitch_order": stitch_order,
            "detections": stitched_dets,
            "panorama_index": idx,
            "straightening_matrix": straightening_matrices.get(idx, None),
            "patch_scale": patch_scale,
            "stitched_image_size": (img.shape[1], img.shape[0]),
            "_image_indices": meta.get('_image_indices', []),
        }

        if save_path:
            suffix = "" if idx == 0 else f"_{idx}"
            Image.fromarray(img).save(os.path.join(save_path, f"StitchedImage{suffix}.jpg"))
            save_meta = metadata.copy()
            save_meta['detections'] = [d.to_dict() for d in stitched_dets]
            np.save(os.path.join(save_path, f"Metadata{suffix}.npy"), save_meta)

        results.append((img, metadata,
                         det_list[idx] if idx < len(det_list) else []))
    return results


def get_mostly_unvisible_polygon_idxs(image_polygons, polygon_trees, polygon_dict, ordering):
    delete_indices = {k: [] for k in image_polygons}
    image_order = {}
    for i, (src, tgt) in enumerate(ordering):
        if src not in image_order:
            image_order[src] = -1
        if tgt not in image_order:
            image_order[tgt] = i
    for k in image_polygons:
        if k not in image_order:
            image_order[k] = len(ordering)

    for img_idx, img_poly in image_polygons.items():
        for other_idx, other_poly in image_polygons.items():
            if img_idx == other_idx or image_order[img_idx] >= image_order[other_idx]:
                continue
            hits = polygon_trees[img_idx].query(other_poly.geometric, predicate="intersects")
            for h in hits:
                if polygon_dict[img_idx][h].get_inside_rate(other_poly) > 0.8:
                    delete_indices[img_idx].append(h)
    return delete_indices


def prune_duplicate_polygons(product_polygons, polygon_trees, polygon_dict, delete_indices, img_idx, other_idx):
    for pidx, ppoly in enumerate(product_polygons):
        if pidx in delete_indices.get(img_idx, []):
            continue
        hits = polygon_trees[other_idx].query(ppoly.geometric, predicate="intersects")
        for oidx in hits:
            if oidx in delete_indices.get(other_idx, []):
                continue
            opoly = polygon_dict[other_idx][oidx]
            if opoly.get_inside_rate(ppoly) > 0.8 or ppoly.get_inside_rate(opoly) > 0.8:
                if ppoly.get_area() > opoly.get_area():
                    delete_indices[other_idx].append(oidx)
                else:
                    delete_indices[img_idx].append(pidx)
    return delete_indices


def delete_unvisible_products(image_polygons, product_polygon_dict, ordering):
    trees = {k: STRtree([p.geometric for p in v]) for k, v in product_polygon_dict.items()}
    delete_indices = get_mostly_unvisible_polygon_idxs(image_polygons, trees, product_polygon_dict, ordering)
    for img_idx, products in product_polygon_dict.items():
        for other_idx in image_polygons:
            if img_idx == other_idx:
                continue
            delete_indices = prune_duplicate_polygons(products, trees, product_polygon_dict,
                                                      delete_indices, img_idx, other_idx)
    return delete_indices


def run_benchmark(combinations=None, data_root=None, config=None):
    data_root = data_root or BENCHMARK_DATA_ROOT
    combinations = combinations or COMBINATIONS

    if config is None:
        config = get_config()

    sequences = get_sequences(data_root)
    detectors = {}  # cache: detector_name -> detector instance

    # Resolve detector weight paths
    det_weights = {}
    if hasattr(config, 'detector'):
        w = getattr(config.detector, 'sku_yolo_weights', None)
        if w:
            det_weights['sku_yolo_detector'] = resolve_path(w)
        w = getattr(config.detector, 'pvpss_weights', None)
        if w:
            det_weights['pvpss_detector'] = resolve_path(w)
    if 'sku_yolo_detector' not in det_weights:
        det_weights['sku_yolo_detector'] = resolve_path('weights/sku_yolo/best_sku110k.pt')
    if 'pvpss_detector' not in det_weights:
        det_weights['pvpss_detector'] = resolve_path('weights/pvpss/weights.pt')

    for combination in combinations:
        device = combination.get('device', 'cpu')

        if 'sku_yolo_detector' not in detectors:
            weights = det_weights.get('sku_yolo_detector')
            if weights and os.path.isfile(weights):
                detectors['sku_yolo_detector'] = create_sku_yolo_detector(weights, device=device)
            else:
                logger.warning(f"Detector weights not found: {weights}")
                continue

        detector = detectors['sku_yolo_detector']
        stitcher = create_stitcher(combination, config)

        for seq_no in sequences:
            folder = f"{combination['model_name']}_{combination['device']}_sku_yolo_detector"
            save_path = os.path.join(data_root, "stitching_results", folder, str(seq_no))
            os.makedirs(save_path, exist_ok=True)
            prefix = f"{seq_no}. Sequence ({folder})"
            logger.info(f"{prefix}: Started")

            seq_paths = sorted(glob.glob(os.path.join(data_root, "images", f"{seq_no}_*.jpg")))
            seq_images = [read_image(p) for p in seq_paths]

            single_dets = []
            for img in seq_images:
                single_dets.append(get_sku_yolo_detections(detector, img))
            logger.info(f"{prefix}: Single image detections: {[len(d) for d in single_dets]}")

            try:
                panorama_results = get_stitched_image_and_detections(
                    seq_images, stitcher, detector,
                    model_name=combination['model_name'],
                    save_path=save_path, config=config)
                logger.info(f"{prefix}: {len(panorama_results)} panorama(s) generated")
            except Exception as e:
                logger.error(f"{prefix}: Stitching error: {e}")
                continue

            for pano_idx, pano_result in enumerate(panorama_results):
                stitched_image, metadata, product_dets = pano_result
                pano_suffix = "" if pano_idx == 0 else f"_pano{pano_idx}"

                # Filter to only images belonging to this panorama
                pano_image_indices = metadata.get('_image_indices', list(range(len(seq_images))))
                pano_images = [seq_images[i] for i in pano_image_indices]
                pano_single_dets = [single_dets[i] for i in pano_image_indices]

                vis = draw_stitched_with_detections(stitched_image, product_dets)
                vis.save(os.path.join(save_path, f"Stitched Image with Products{pano_suffix}.jpg"))

                image_transformed_polygons = transform_image_corners(metadata, pano_images, image_indices=pano_image_indices)
                transformed_polygons = get_transformed_polygons(metadata, pano_single_dets, image_indices=pano_image_indices)

                try:
                    delete_indices = delete_unvisible_products(
                        image_transformed_polygons, transformed_polygons, metadata.get('stitch_order', []))
                except GEOSException as e:
                    logger.error(f"{prefix} [Pano {pano_idx}]: GEOS error: {e}")
                    continue

                total_single = sum(len(d) for d in pano_single_dets)
                before_delete = {k: list(v) for k, v in transformed_polygons.items()}
                single_transformed = []
                for img_idx in transformed_polygons:
                    arr = np.array(transformed_polygons[img_idx])
                    transformed_polygons[img_idx] = np.delete(arr, delete_indices[img_idx], axis=0).tolist()
                    single_transformed.extend(transformed_polygons[img_idx])

                straightening_matrix = metadata.get('straightening_matrix')
                if straightening_matrix is not None:
                    single_transformed = apply_straightening_to_polygons(single_transformed, straightening_matrix)

                stitched_dets = metadata['detections']

                ious = create_iou_matrix(stitched_dets, single_transformed)
                matches, unmatched_stitched, unmatched_single = linear_assignment(1 - np.array(ious), thresh=0.4)

                vis = draw_unmatched(
                    stitched_image,
                    [stitched_dets[i] for i in unmatched_stitched],
                    [single_transformed[i] for i in unmatched_single] if single_transformed else [])
                vis.save(os.path.join(save_path, f"Unmatcheds{pano_suffix}.jpg"))

                draw_deleted_polygons(stitched_image, delete_indices, before_delete).save(
                    os.path.join(save_path, f"Deleted Polygons{pano_suffix}.jpg"))
                draw_final_polygons(stitched_image, single_transformed).save(
                    os.path.join(save_path, f"Final Polygons{pano_suffix}.jpg"))
                draw_stitched_image_detections(stitched_image, stitched_dets).save(
                    os.path.join(save_path, f"Stitched Detections{pano_suffix}.jpg"))

                stats = calculate_stats(
                    matches=len(matches), unmatched_single=len(unmatched_single),
                    unmatched_stitched=len(unmatched_stitched),
                    box_count_of_stitched=len(stitched_dets),
                    box_count_of_single_pruned=len(single_transformed),
                    box_count_of_single=total_single)
                stats['num_input_images'] = len(seq_images)
                stats['num_panoramas'] = len(panorama_results)
                stats['fully_stitched'] = len(panorama_results) == 1
                save_stats(stats, save_path, suffix=pano_suffix)
                logger.info(f"{prefix} [Pano {pano_idx}]: F1={stats['f1_score']}, "
                           f"Precision={stats['precision']}, Recall={stats['recall']}")

            del panorama_results, seq_images, single_dets
            _clear_memory()

    wrap_results(data_root)
    logger.info("Benchmark complete. Results saved.")
