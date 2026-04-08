import os
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

BENCHMARK_DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "benchmark")

COMBINATIONS = [
    {"model_name": "lightglue_dinov3_vits", "device": DEVICE, "detector": "sku_yolo_detector"},
    {"model_name": "lightglue_dinov3_vitb", "device": DEVICE, "detector": "sku_yolo_detector"},
    {"model_name": "lightglue_dinov3_vitl", "device": DEVICE, "detector": "sku_yolo_detector"},
    {"model_name": "bfmatcher_dinov3_vits", "device": DEVICE, "detector": "sku_yolo_detector"},
    {"model_name": "bfmatcher_dinov3_vitb", "device": DEVICE, "detector": "sku_yolo_detector"},
    {"model_name": "bfmatcher_dinov3_vitl", "device": DEVICE, "detector": "sku_yolo_detector"},
    {"model_name": "lightgluestick", "device": DEVICE, "detector": "sku_yolo_detector"},
    {"model_name": "gluestick", "device": DEVICE, "detector": "sku_yolo_detector"},
    {"model_name": "roma_v2", "device": DEVICE, "detector": "sku_yolo_detector"},
    {"model_name": "superglue", "device": DEVICE, "detector": "sku_yolo_detector"},
    {"model_name": "lightglue_superpoint", "device": DEVICE, "detector": "sku_yolo_detector"},
    {"model_name": "lightglue_disk", "device": DEVICE, "detector": "sku_yolo_detector"},
    {"model_name": "eloftr", "device": DEVICE, "detector": "sku_yolo_detector"},
    {"model_name": "lightglue_minima", "device": DEVICE, "detector": "sku_yolo_detector"},
]
