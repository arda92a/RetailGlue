from __future__ import annotations
import torch
import numpy as np
import logging
from typing import List, Tuple, Optional

from transformers import (
    SuperGlueImageProcessor, SuperGlueForKeypointMatching,
    LightGlueImageProcessor, LightGlueForKeypointMatching,
    EfficientLoFTRImageProcessor, EfficientLoFTRForKeypointMatching,
)
from transformers.models.lightglue.modeling_lightglue import LightGlueKeypointMatchingOutput
from transformers.models.superpoint.modeling_superpoint import SuperPointKeypointDescriptionOutput

HF_MODEL_REGISTRY = {
    "superglue": "magic-leap-community/superglue_outdoor",
    "lightglue_superpoint": "ETH-CVG/lightglue_superpoint",
    "lightglue_disk": "ETH-CVG/lightglue_disk",
    "lightglue_minima": "stevenbucaille/lightglue_minima",
    "eloftr": "zju-community/efficientloftr",
    "matchanything_eloftr": "zju-community/matchanything_eloftr",
}


class HFModel:
    def __init__(self, model_name: str, size=None, device: str = 'cpu', batch_size: int = 4):
        self.device = device
        self.size = size
        self.batch_size = batch_size
        self.last_keypoints = {}

        if 'superglue' in model_name:
            repo = HF_MODEL_REGISTRY["superglue"]
            self.processor = SuperGlueImageProcessor.from_pretrained(repo, size=size)
            self.model = SuperGlueForKeypointMatching.from_pretrained(repo)
        elif 'lightglue' in model_name:
            if 'superpoint' in model_name:
                repo = HF_MODEL_REGISTRY["lightglue_superpoint"]
            elif 'disk' in model_name:
                repo = HF_MODEL_REGISTRY["lightglue_disk"]
            elif 'minima' in model_name:
                repo = HF_MODEL_REGISTRY["lightglue_minima"]
            else:
                raise NotImplementedError(f"Unknown LightGlue variant: {model_name}")
            self.processor = LightGlueImageProcessor.from_pretrained(repo, size=size)
            self.model = LightGlueForKeypointMatching.from_pretrained(repo)
        elif 'loftr' in model_name:
            if 'matchanything' in model_name:
                repo = HF_MODEL_REGISTRY["matchanything_eloftr"]
            else:
                repo = HF_MODEL_REGISTRY["eloftr"]
            self.processor = EfficientLoFTRImageProcessor.from_pretrained(repo, size=size)
            self.model = EfficientLoFTRForKeypointMatching.from_pretrained(repo)
        else:
            raise NotImplementedError(f"Unknown model: {model_name}")

        self.model.eval()
        self.model.to(device)

    def inference(self, image_pairs):
        np_pairs = []
        image_sizes = []
        image_indices = []

        for i0, i1 in image_pairs:
            if hasattr(i0, 'image'):
                img0, idx0 = np.copy(i0.image), i0.idx
            else:
                img0, idx0 = np.copy(i0), None
            if hasattr(i1, 'image'):
                img1, idx1 = np.copy(i1.image), i1.idx
            else:
                img1, idx1 = np.copy(i1), None

            np_pairs.append([img0, img1])
            image_sizes.append(((img0.shape[0], img0.shape[1]), (img1.shape[0], img1.shape[1])))
            image_indices.append((idx0, idx1))

        inputs = self.processor(images=np_pairs, return_tensors="pt")
        inputs.to(self.device)

        with torch.no_grad():
            if isinstance(self.model, LightGlueForKeypointMatching):
                outputs = self._forward_lightglue(
                    pixel_values=inputs["pixel_values"],
                    image_indices=image_indices,
                    image_sizes=image_sizes
                )
            else:
                outputs = self.model(**inputs)

        batch_processed = self.processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)

        batch_src, batch_tgt = [], []
        for out in batch_processed:
            batch_src.append(out["keypoints0"].cpu().numpy())
            batch_tgt.append(out["keypoints1"].cpu().numpy())
        return batch_src, batch_tgt

    def _forward_lightglue(self, pixel_values, image_indices=None, image_sizes=None):
        if pixel_values.ndim != 5 or pixel_values.size(1) != 2:
            raise ValueError("Input must be 5D tensor: (batch, 2, C, H, W)")

        device = pixel_values.device
        batch_size, _, channels, height, width = pixel_values.shape
        flat_pixels = pixel_values.reshape(batch_size * 2, channels, height, width).contiguous()

        unique_tensors = []
        flat_to_unique = []
        hash_to_idx = {}
        for t in flat_pixels:
            key = (t.numel(), float(t.sum().item()), float((t.float() * t.float()).sum().item()))
            if key in hash_to_idx:
                flat_to_unique.append(hash_to_idx[key])
            else:
                hash_to_idx[key] = len(unique_tensors)
                flat_to_unique.append(hash_to_idx[key])
                unique_tensors.append(t)

        unique_pixels = torch.stack(unique_tensors, dim=0)

        keypoints_u, descriptors_u, mask_u = [], [], []
        for batch in torch.split(unique_pixels, self.batch_size):
            det: SuperPointKeypointDescriptionOutput = self.model.keypoint_detector(batch.to(device))
            kpts_b, _, desc_b, masks_b = det[:4]
            for i in range(kpts_b.shape[0]):
                keypoints_u.append(kpts_b[i])
                descriptors_u.append(desc_b[i])
                mask_u.append(masks_b[i])

        if image_indices is not None and image_sizes is not None:
            self.last_keypoints = {}
            for pair_idx, (idx0, idx1) in enumerate(image_indices):
                u0 = flat_to_unique[2 * pair_idx]
                u1 = flat_to_unique[2 * pair_idx + 1]
                (h0, w0), (h1, w1) = image_sizes[pair_idx]
                for img_idx, u_idx, oh, ow in [(idx0, u0, h0, w0), (idx1, u1, h1, w1)]:
                    if img_idx is not None and img_idx not in self.last_keypoints:
                        kpts = keypoints_u[u_idx]
                        m = mask_u[u_idx]
                        valid = kpts[m.bool()] if m is not None else kpts
                        px = valid.clone()
                        px[:, 0] *= ow
                        px[:, 1] *= oh
                        self.last_keypoints[img_idx] = px.cpu().numpy()

        global_max_k = max(k.shape[-2] for k in keypoints_u)

        matches_batch, scores_batch, prune_batch, mask_batch, kpts_batch = [], [], [], [], []

        for pair_idx in range(batch_size):
            u0 = flat_to_unique[2 * pair_idx]
            u1 = flat_to_unique[2 * pair_idx + 1]

            kpts0, kpts1 = keypoints_u[u0], keypoints_u[u1]
            desc0, desc1 = descriptors_u[u0], descriptors_u[u1]
            m0, m1 = mask_u[u0], mask_u[u1]

            def _pad(tensor, target, dim=0):
                diff = target - tensor.shape[dim]
                if diff <= 0:
                    return tensor
                pad_shape = list(tensor.shape)
                pad_shape[dim] = diff
                return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=dim)

            kpts0, kpts1 = _pad(kpts0, global_max_k), _pad(kpts1, global_max_k)
            desc0, desc1 = _pad(desc0, global_max_k), _pad(desc1, global_max_k)
            m0, m1 = _pad(m0, global_max_k), _pad(m1, global_max_k)

            pair_kpts = torch.stack([kpts0, kpts1], dim=0).unsqueeze(0).to(device)
            pair_desc = torch.stack([desc0, desc1], dim=0).unsqueeze(0).to(device)
            pair_mask = torch.stack([m0, m1], dim=0).unsqueeze(0).to(device)

            abs_kpts = pair_kpts.clone()
            abs_kpts[:, :, :, 0] *= width
            abs_kpts[:, :, :, 1] *= height

            matches, matching_scores, prune, _, _ = self.model._match_image_pair(
                abs_kpts, pair_desc, height, width,
                mask=pair_mask, output_attentions=False, output_hidden_states=False,
            )

            matches_batch.append(matches)
            scores_batch.append(matching_scores)
            prune_batch.append(prune)
            mask_batch.append(pair_mask)
            kpts_batch.append(pair_kpts)

        return LightGlueKeypointMatchingOutput(
            loss=None,
            matches=torch.cat(matches_batch, dim=0),
            matching_scores=torch.cat(scores_batch, dim=0),
            keypoints=torch.cat(kpts_batch, dim=0),
            prune=torch.cat(prune_batch, dim=0),
            mask=torch.cat(mask_batch, dim=0),
            hidden_states=None,
            attentions=None,
        )
