import cv2
import numpy as np
from typing import List
from ultralytics import YOLO
from retailglue.entities import BoundingBox, Detection


class YOLODetector:
    def __init__(self, weights: str, device: str = "cpu", img_size: int = 1024):
        self.model = YOLO(weights)
        self.device = device
        self.img_size = img_size
        self._is_obb = (self.model.task == 'obb')

    def detect(self, images: List[np.ndarray], conf: float = 0.25) -> List[List[Detection]]:
        results_all = []
        for image in images:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            results = self.model.predict(
                image_bgr, conf=conf, device=self.device,
                imgsz=self.img_size, verbose=False,
            )
            detections = []
            for r in results:
                if self._is_obb:
                    if r.obb is None:
                        continue
                    for obb in r.obb:
                        cls_id = int(obb.cls[0])
                        cls_name = self.model.names.get(cls_id, '')
                        if cls_name != 'product':
                            continue
                        # Convert OBB xyxyxyxy to axis-aligned bbox
                        pts = obb.xyxyxyxy[0].cpu().numpy()  # (4, 2)
                        x1, y1 = pts.min(axis=0)
                        x2, y2 = pts.max(axis=0)
                        score = float(obb.conf[0])
                        detections.append(Detection(int(x1), int(y1), int(x2), int(y2),
                                                    score=score, name="product"))
                else:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        score = float(box.conf[0])
                        detections.append(Detection(int(x1), int(y1), int(x2), int(y2),
                                                    score=score, name="product"))
            results_all.append(detections)
        return results_all
