import logging
import os
import numpy as np
import torch
import torchvision.transforms as T
from typing import List, Optional
from PIL import Image
from retailglue.entities import Detection

logger = logging.getLogger("retailglue")

DINO_VARIANTS = {
    'lightglue_dinov3_vits': {
        'hub_name': 'dinov3_vits16',
        'embedding_dim': 384,
        'repo': 'facebookresearch/dinov3',
    },
    'lightglue_dinov3_vitb': {
        'hub_name': 'dinov3_vitb16',
        'embedding_dim': 768,
        'repo': 'facebookresearch/dinov3',
    },
    'lightglue_dinov3_vitl': {
        'hub_name': 'dinov3_vitl16',
        'embedding_dim': 1024,
        'repo': 'facebookresearch/dinov3',
    },
    'lightglue_dinov3_vith': {
        'hub_name': 'dinov3_vith16plus',
        'embedding_dim': 1280,
        'repo': 'facebookresearch/dinov3',
    },
    'lightglue_dinov2_vits': {
        'hub_name': 'dinov2_vits14',
        'embedding_dim': 384,
        'repo': 'facebookresearch/dinov2',
    },
}

# BFMatcher variants use the same DINO backbones
BF_TO_DINO_VARIANT = {
    'bfmatcher_dinov3_vits': 'lightglue_dinov3_vits',
    'bfmatcher_dinov3_vitb': 'lightglue_dinov3_vitb',
    'bfmatcher_dinov3_vitl': 'lightglue_dinov3_vitl',
}

_dino_models = {}


def _get_dino_transform():
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_dino_model(device: str, variant: str = 'lightglue_dinov3_vits',
                     weights_path: Optional[str] = None):
    global _dino_models
    variant = BF_TO_DINO_VARIANT.get(variant, variant)
    cached = _dino_models.get(variant)
    if cached is not None and cached[1] == device:
        return cached[0]

    cfg = DINO_VARIANTS[variant]

    if weights_path and os.path.isfile(weights_path):
        model = torch.hub.load(cfg['repo'], cfg['hub_name'], pretrained=False)
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"DINO {cfg['hub_name']} loaded from local weights: {weights_path}")
    else:
        if weights_path:
            logger.warning(f"DINO weights not found at '{weights_path}', "
                           f"falling back to pretrained hub download")
        model = torch.hub.load(cfg['repo'], cfg['hub_name'], pretrained=True)
        logger.info(f"DINO {cfg['hub_name']} loaded from torch hub (pretrained=True)")

    model = model.to(device).eval()
    _dino_models[variant] = (model, device)
    logger.info(f"DINO {cfg['hub_name']} ready on {device}, dim={cfg['embedding_dim']}")
    return model


@torch.no_grad()
def extract_dino_embeddings(images: List[np.ndarray], detections: List[List[Detection]],
                             device: str, variant: str = 'lightglue_dinov3_vits',
                             weights_path: Optional[str] = None, crop_margin: int = 10):
    model = load_dino_model(device, variant, weights_path)
    transform = _get_dino_transform()

    for image, dets in zip(images, detections):
        products = [d for d in dets if d.name == 'product']
        if not products:
            continue
        pil_image = Image.fromarray(image)
        tensors = []
        for product in products:
            padded = product.add_margin(crop_margin, max_right=pil_image.width, max_bottom=pil_image.height)
            crop = pil_image.crop(padded.get_pascal_voc_format())
            if crop.mode != 'RGB':
                crop = crop.convert('RGB')
            try:
                tensors.append(transform(crop))
            except Exception:
                tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(tensors, dim=0).to(device)
        embeddings = model(batch)
        # NOTE: Do NOT L2-normalize — the fine-tuned LightGlue checkpoints
        # were trained on raw DINOv3 embeddings (norm ≈ 8, not 1.0).
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)

        for product, emb in zip(products, embeddings_np):
            product.embedding = emb
