import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import cv2
import torch
import numpy as np
import logging

logger = logging.getLogger("retailglue")


class RomaV2Matcher:
    def __init__(self, device='cpu', num_samples=5000, setting='fast', ransac_threshold=5.0):
        self.device_str = device
        self.num_samples = num_samples
        self.setting = setting
        self.ransac_threshold = ransac_threshold
        self.model = self._load_model()

    def _load_model(self):
        import romav2.device as roma_device
        roma_device.device = torch.device(self.device_str)
        from romav2 import RoMaV2

        if 'mps' in self.device_str or 'cpu' in self.device_str:
            cfg = RoMaV2.Cfg(compile=False)
            model = RoMaV2(cfg=cfg)
        else:
            model = RoMaV2()

        model.apply_setting(self.setting)
        logger.info(f"RoMaV2 initialized: device={self.device_str}, setting={self.setting}")
        return model

    @torch.inference_mode()
    def _match_pair(self, img0: np.ndarray, img1: np.ndarray):
        H_A, W_A = img0.shape[:2]
        H_B, W_B = img1.shape[:2]

        try:
            preds = self.model.match(img0, img1)
            matches, confidence, *_ = self.model.sample(preds, self.num_samples)
            kpts_A, kpts_B = self.model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
            kpts_A = kpts_A.cpu().numpy().astype(np.float32)
            kpts_B = kpts_B.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.error(f"RoMaV2 matching failed: {e}")
            return (np.array([], dtype=np.float32).reshape(0, 2),
                    np.array([], dtype=np.float32).reshape(0, 2))

        if len(kpts_A) < 4:
            return kpts_A, kpts_B

        _, mask = cv2.findHomography(kpts_A, kpts_B, cv2.USAC_MAGSAC,
                                     ransacReprojThreshold=self.ransac_threshold,
                                     maxIters=5000, confidence=0.9999)
        if mask is None:
            return kpts_A, kpts_B
        inlier = mask.ravel().astype(bool)
        return kpts_A[inlier], kpts_B[inlier]

    def inference(self, image_pairs):
        batch_src, batch_tgt = [], []
        for i0, i1 in image_pairs:
            img0 = i0.image if hasattr(i0, 'image') else i0
            img1 = i1.image if hasattr(i1, 'image') else i1
            pts0, pts1 = self._match_pair(img0, img1)
            batch_src.append(pts0)
            batch_tgt.append(pts1)
        return batch_src, batch_tgt
