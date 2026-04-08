import cv2
import time
import numpy as np
import logging
from functools import wraps
from collections import defaultdict

from shapely.strtree import STRtree
from shapely.geometry import box as shapely_box

from retailglue.entities import Detection


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


class DetectionTransformer:
    def __init__(self, verbose=False, runtimes=None):
        self.verbose = verbose
        self.runtimes = runtimes if runtimes is not None else defaultdict(list)

    @track_runtime()
    def transform_detections(self, detections, homography_matrix):
        result = []
        for det in detections:
            transformed = self._transform_single(det, homography_matrix)
            if transformed is not None:
                result.append(transformed)
        return result

    def _transform_single(self, detection, H):
        if isinstance(detection, Detection):
            return detection.rectify(H)
        return detection

    @track_runtime()
    def remove_duplicate_detections(self, det_per_image, stitch_order):
        if not det_per_image:
            return [], []

        detection_trees = {
            img_idx: STRtree([shapely_box(d.xmin, d.ymin, d.xmax, d.ymax) for d in dets])
            for img_idx, dets in det_per_image if dets
        }

        delete_indices = {img_idx: set() for img_idx, _ in det_per_image}
        image_order = {}
        counter = 0

        if stitch_order:
            first_source = stitch_order[0][0]
            image_order[first_source] = 0
            counter = 1
        for source, target in stitch_order:
            if target not in image_order:
                image_order[target] = counter
                counter += 1
        for img_idx, _ in det_per_image:
            if img_idx not in image_order:
                image_order[img_idx] = counter
                counter += 1

        for img_idx, img_dets in det_per_image:
            if img_idx not in detection_trees:
                continue
            img_ord = image_order[img_idx]

            for other_idx, other_dets in det_per_image:
                if img_idx == other_idx or other_idx not in detection_trees:
                    continue
                if img_ord >= image_order[other_idx]:
                    continue

                other_tree = detection_trees[other_idx]
                for det_idx, det in enumerate(img_dets):
                    if det_idx in delete_indices[img_idx]:
                        continue
                    det_box = shapely_box(det.xmin, det.ymin, det.xmax, det.ymax)
                    overlapping = other_tree.query(det_box, predicate="intersects")
                    for other_det_idx in overlapping:
                        if other_det_idx in delete_indices[other_idx]:
                            continue
                        other_det = other_dets[other_det_idx]
                        other_box = shapely_box(other_det.xmin, other_det.ymin, other_det.xmax, other_det.ymax)
                        inter = det_box.intersection(other_box).area
                        if inter == 0:
                            continue
                        inside1 = inter / det_box.area if det_box.area > 0 else 0
                        inside2 = inter / other_box.area if other_box.area > 0 else 0
                        if inside1 > 0.3 or inside2 > 0.3:
                            if det_box.area > other_box.area:
                                delete_indices[other_idx].add(other_det_idx)
                            else:
                                delete_indices[img_idx].add(det_idx)

        final, keep = [], []
        flat_idx = 0
        for img_idx, dets in det_per_image:
            for det_idx, det in enumerate(dets):
                if det_idx not in delete_indices[img_idx]:
                    final.append(det)
                    keep.append(flat_idx)
                flat_idx += 1
        return final, keep
