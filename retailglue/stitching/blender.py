import cv2
import time
import numpy as np
from functools import wraps
from dataclasses import dataclass
from collections import defaultdict


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


class Blender:
    def __init__(self, config=None, runtimes=None):
        self.runtimes = runtimes if isinstance(runtimes, dict) else defaultdict(list)
        self.eps = 1e-6
        self.gaussian_kernel_size = 15
        self.erode_kernel_size = 3

    @dataclass
    class BlendRegion:
        canvas: np.ndarray
        warped: np.ndarray
        mask1: np.ndarray
        mask2: np.ndarray
        overlap: np.ndarray
        bounds: tuple
        weight_1: np.ndarray
        weight_2: np.ndarray

    @track_runtime()
    def build_masks_and_exposure(self, canvas, warped):
        mask1 = np.any(canvas, axis=2).astype(np.uint8)
        mask2 = np.any(warped, axis=2).astype(np.uint8)
        overlap = (mask1 & mask2).astype(np.uint8)

        if np.count_nonzero(overlap) == 0:
            return mask1, mask2, overlap, warped

        b1, g1, r1, _ = cv2.mean(canvas, mask=overlap)
        b2, g2, r2, _ = cv2.mean(warped, mask=overlap)
        gains = np.array([
            (b1 / (b2 + self.eps)) if b2 > 0 else 1.0,
            (g1 / (g2 + self.eps)) if g2 > 0 else 1.0,
            (r1 / (r2 + self.eps)) if r2 > 0 else 1.0,
        ], dtype=np.float32)
        warped = np.clip(warped.astype(np.float32) * gains[None, None, :], 0, 255).astype(np.uint8)
        return mask1, mask2, overlap, warped

    @track_runtime()
    def extract_blend_region_and_weights(self, canvas, warped, masks, pad):
        mask1, mask2, overlap = masks
        ys, xs = np.where(overlap > 0)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        y0 = max(y0 - pad, 0)
        x0 = max(x0 - pad, 0)
        y1 = min(y1 + pad + 1, overlap.shape[0])
        x1 = min(x1 + pad + 1, overlap.shape[1])

        bounds = (y0, y1, x0, x1)
        canvas_roi = canvas[y0:y1, x0:x1]
        warped_roi = warped[y0:y1, x0:x1]
        mask1_roi = mask1[y0:y1, x0:x1]
        mask2_roi = mask2[y0:y1, x0:x1]
        overlap_roi = overlap[y0:y1, x0:x1]

        kernel = np.ones((self.erode_kernel_size, self.erode_kernel_size), np.uint8)
        mask1_e = cv2.erode(mask1_roi, kernel, iterations=1)
        mask2_e = cv2.erode(mask2_roi, kernel, iterations=1)

        dt1 = cv2.distanceTransform(mask1_e, cv2.DIST_L2, 3)
        dt2 = cv2.distanceTransform(mask2_e, cv2.DIST_L2, 3)
        total = dt1 + dt2 + self.eps
        weight_1 = dt1 / total
        weight_2 = dt2 / total

        return Blender.BlendRegion(
            canvas=canvas_roi, warped=warped_roi,
            mask1=mask1_roi, mask2=mask2_roi, overlap=overlap_roi,
            bounds=bounds, weight_1=weight_1, weight_2=weight_2)

    @track_runtime()
    def adaptive_blend(self, roi):
        diff = cv2.absdiff(roi.canvas, roi.warped)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        diff_f = diff_gray.astype(np.float32)
        local_mean = cv2.GaussianBlur(diff_f, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
        sq_mean = cv2.GaussianBlur(diff_f ** 2, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
        local_std = np.sqrt(np.maximum(sq_mean - local_mean ** 2, 0))
        T_map = local_mean + 0.5 * local_std
        high_diff = (diff_f > T_map) & (roi.overlap > 0)

        result = np.zeros_like(roi.canvas, dtype=np.float32)
        canvas_f = roi.canvas.astype(np.float32)
        warped_f = roi.warped.astype(np.float32)

        only1 = (roi.mask1 > 0) & (roi.mask2 == 0)
        only2 = (roi.mask2 > 0) & (roi.mask1 == 0)
        result[only1] = canvas_f[only1]
        result[only2] = warped_f[only2]

        feather = (roi.overlap > 0) & (~high_diff)
        w1 = roi.weight_1[feather][:, None]
        w2 = roi.weight_2[feather][:, None]
        result[feather] = w1 * canvas_f[feather] + w2 * warped_f[feather]

        choose1 = (roi.weight_1 >= roi.weight_2) & high_diff
        choose2 = (~choose1) & high_diff
        result[choose1] = canvas_f[choose1]
        result[choose2] = warped_f[choose2]

        return np.clip(result, 0, 255).astype(np.uint8)
