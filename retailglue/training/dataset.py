"""Product pairs dataset for LightGlue training.

Expected data structure:
    data_dir/
    ├── images/
    │   ├── shelf_001.jpg
    │   └── ...
    ├── annotations/
    │   ├── shelf_001.json
    │   └── ...
    └── matches.json

Annotation JSON format (per image):
    {
        "products": [
            {"product_id": 0, "bbox": [x1, y1, x2, y2], "embedding": [0.1, ...]}
        ]
    }

Matches JSON format:
    {
        "pairs": [
            {"image0": "shelf_001", "image1": "shelf_002", "matches": [[0, 0], [2, 1]]}
        ]
    }
"""

import collections.abc
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("retailglue.training")


def _collate(batch):
    """Collate nested dicts/lists of tensors into batches."""
    if not isinstance(batch, list):
        return batch
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, dim=0)
    elif isinstance(elem, (int, float)):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: _collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        return [_collate(samples) for samples in zip(*batch)]
    elif elem is None:
        return elem
    return torch.stack(batch, 0)


class ProductPairsDataset:
    """Manager for product pair datasets with train/val/test splits.

    Args:
        data_dir: Path to dataset root containing images/, annotations/, matches.json.
        embedding_dim: Dimension of product embeddings.
        max_products: Maximum products per view (padded if fewer).
        batch_size: Batch size for data loaders.
        num_workers: Number of data loading workers.
        train_ratio: Fraction for training split.
        val_ratio: Fraction for validation split.
        shuffle_seed: Random seed for reproducible splitting.
    """

    def __init__(
        self,
        data_dir,
        embedding_dim=384,
        max_products=400,
        batch_size=8,
        num_workers=4,
        train_ratio=0.7,
        val_ratio=0.2,
        shuffle_seed=42,
    ):
        self.data_dir = Path(data_dir)
        self.embedding_dim = embedding_dim
        self.max_products = max_products
        self.batch_size = batch_size
        self.num_workers = num_workers

        matches_file = self.data_dir / "matches.json"
        if not matches_file.exists():
            raise FileNotFoundError(f"Matches file not found: {matches_file}")

        with open(matches_file) as f:
            matches_data = json.load(f)

        pairs = [
            {
                "image0": p["image0"],
                "image1": p["image1"],
                "matches": p["matches"],
            }
            for p in matches_data["pairs"]
        ]

        # Deterministic shuffle and split
        rng = np.random.RandomState(shuffle_seed)
        rng.shuffle(pairs)

        total = len(pairs)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        self.splits = {
            "train": pairs[:train_end],
            "val": pairs[train_end:val_end],
            "test": pairs[val_end:],
        }

        logger.info(
            f"Dataset: {len(self.splits['train'])} train, "
            f"{len(self.splits['val'])} val, "
            f"{len(self.splits['test'])} test pairs"
        )

    def get_loader(self, split, shuffle=None):
        """Create a DataLoader for the given split."""
        assert split in ("train", "val", "test")
        dataset = _PairDataset(
            self.splits[split],
            self.data_dir,
            self.embedding_dim,
            self.max_products,
        )
        if shuffle is None:
            shuffle = split == "train"
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=_collate,
            drop_last=(split == "train"),
            pin_memory=True,
            prefetch_factor=2,
        )


class _PairDataset(torch.utils.data.Dataset):
    """Internal per-split dataset for product pair loading."""

    def __init__(self, pairs, data_dir, embedding_dim, max_products):
        self.pairs = pairs
        self.image_dir = data_dir / "images"
        self.annotation_dir = data_dir / "annotations"
        self.embedding_dim = embedding_dim
        self.max_products = max_products

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        name0, name1 = pair["image0"], pair["image1"]

        view0 = self._load_view(name0)
        view1 = self._load_view(name1)

        gt_matches0, gt_matches1 = self._build_gt_matches(
            pair["matches"], view0["n_products"], view1["n_products"]
        )
        gt_assignment = self._build_assignment_matrix(
            pair["matches"], view0["n_products"], view1["n_products"]
        )

        return {
            "view0": view0,
            "view1": view1,
            "gt_matches0": gt_matches0,
            "gt_matches1": gt_matches1,
            "gt_assignment": gt_assignment,
        }

    def _load_view(self, name: str) -> Dict:
        """Load annotation for a single view and extract keypoints/embeddings."""
        ann_path = self.annotation_dir / f"{name}.json"
        if ann_path.exists():
            with open(ann_path) as f:
                products = json.load(f).get("products", [])
        else:
            products = []
            logger.warning(f"Annotation not found: {ann_path}")

        # Get image size
        image_path = self.image_dir / f"{name}.jpg"
        if not image_path.exists():
            image_path = self.image_dir / f"{name}.png"

        from PIL import Image
        with Image.open(image_path) as img:
            image_size = [img.width, img.height]

        # Extract keypoints (bbox centers) and embeddings
        keypoints, embeddings = [], []
        for product in products[: self.max_products]:
            bbox = product["bbox"]
            keypoints.append([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            embeddings.append(product["embedding"])

        n_products = len(keypoints)

        # Pad to max_products for batching
        pad_n = self.max_products - n_products
        if pad_n > 0:
            keypoints.extend([[0, 0]] * pad_n)
            embeddings.extend([[0.0] * self.embedding_dim] * pad_n)

        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        descriptors = torch.tensor(embeddings, dtype=torch.float32)
        keypoint_scores = torch.zeros(self.max_products)
        keypoint_scores[:n_products] = 1.0

        return {
            "image_size": torch.tensor(image_size, dtype=torch.float32),
            "keypoints": keypoints,
            "descriptors": descriptors,
            "keypoint_scores": keypoint_scores,
            "n_products": n_products,
        }

    def _build_gt_matches(
        self, matches: List, n0: int, n1: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build ground truth match index tensors."""
        gt_matches0 = torch.full((self.max_products,), -1, dtype=torch.long)
        gt_matches1 = torch.full((self.max_products,), -1, dtype=torch.long)

        for idx0, idx1 in matches:
            if idx0 < n0 and idx1 < n1:
                gt_matches0[idx0] = idx1
                gt_matches1[idx1] = idx0

        return gt_matches0, gt_matches1

    def _build_assignment_matrix(
        self, matches: List, n0: int, n1: int
    ) -> torch.Tensor:
        """Build binary assignment matrix for loss computation."""
        assignment = torch.zeros(
            (self.max_products, self.max_products), dtype=torch.float32
        )
        for idx0, idx1 in matches:
            if idx0 < n0 and idx1 < n1:
                assignment[idx0, idx1] = 1.0

        return assignment
