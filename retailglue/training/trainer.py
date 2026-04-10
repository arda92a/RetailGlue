"""LightGlue training loop.

Supports mixed precision, gradient clipping, LR scheduling,
checkpoint management, and optional W&B logging.
"""

import copy
import logging
import random
import shutil
import signal
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from retailglue.matchers.lightglue import LightGlue
from retailglue.training.dataset import ProductPairsDataset

logger = logging.getLogger("retailglue.training")


# ── Utilities ──────────────────────────────────────────────────────────────────


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def fork_rng(seed=None):
    """Fork RNG state so evaluation doesn't affect training randomness."""
    state = (
        torch.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    if seed is not None:
        set_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(state[0])
        np.random.set_state(state[1])
        random.setstate(state[2])
        if state[3] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state[3])


class AverageMetric:
    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def update(self, tensor):
        tensor = tensor[~torch.isnan(tensor)]
        self._sum += tensor.sum().item()
        self._count += len(tensor)

    def compute(self):
        return self._sum / self._count if self._count > 0 else float("nan")


def batch_to_device(batch, device):
    """Recursively move nested dict/list of tensors to device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(v, device) for v in batch)
    return batch


def _extract_views(data):
    """Flatten view0/view1 dicts into keypoints0/descriptors0/... format."""
    pred = {}
    for suffix in ("0", "1"):
        view = data[f"view{suffix}"]
        for k, v in view.items():
            if k not in ("image", "cache"):
                pred[f"{k}{suffix}"] = v
    return pred


# ── Checkpoint management ──────────────────────────────────────────────────────


def save_checkpoint(model, optimizer, lr_scheduler, conf, results, best_eval,
                    epoch, output_dir, name=None):
    """Save training checkpoint and track best model."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "conf": conf,
        "epoch": epoch,
        "eval": results,
    }
    if name is None:
        name = f"checkpoint_{epoch}.tar"
    path = output_dir / name
    torch.save(checkpoint, str(path))
    logger.info(f"Saved checkpoint: {name}")

    best_key = conf.get("train", {}).get("best_key", "loss/total")
    if results and name != "lightglue_best.tar":
        val = results.get(best_key, float("inf"))
        if val < best_eval:
            best_eval = val
            shutil.copy(str(path), str(output_dir / "lightglue_best.tar"))
            logger.info(f"New best: {best_key}={best_eval:.4f}")

    # Keep only the last N regular checkpoints
    keep = conf.get("train", {}).get("keep_last_checkpoints", 10)
    ckpts = sorted(output_dir.glob("checkpoint_*.tar"), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-keep]:
        old.unlink()

    return best_eval


# ── Evaluation ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, loader, device):
    """Run evaluation over a data loader, return averaged metrics."""
    model.eval()
    results = {}
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = batch_to_device(data, device)
        flat = _extract_views(data)
        pred = model({**data, **flat})
        losses, metrics = model.loss(pred, {**pred, **data})
        numbers = {**metrics, **{f"loss/{k}": v for k, v in losses.items()}}
        for k, v in numbers.items():
            if k not in results:
                results[k] = AverageMetric()
            results[k].update(v if isinstance(v, torch.Tensor) and v.dim() == 1
                              else torch.tensor([v]) if not isinstance(v, torch.Tensor)
                              else v.flatten())
    return {k: v.compute() for k, v in results.items()}


# ── Test evaluation ────────────────────────────────────────────────────────────


@torch.no_grad()
def test_evaluate(model, dataset, conf, device):
    """Run final test evaluation on best model WITHOUT padding."""
    import json
    from PIL import Image

    model.eval()
    data_dir = Path(conf["data"]["data_dir"])
    test_pairs = dataset.splits["test"]

    pair_precisions, pair_recalls = [], []
    total_tp = total_fp = total_fn = 0

    for pair_info in tqdm(test_pairs, desc="Test (no padding)"):
        name0, name1 = pair_info["image0"], pair_info["image1"]
        gt_matches = set(tuple(m) for m in pair_info["matches"])

        ann0_path = data_dir / "annotations" / f"{name0}.json"
        ann1_path = data_dir / "annotations" / f"{name1}.json"

        with open(ann0_path) as f:
            ann0 = json.load(f)
        with open(ann1_path) as f:
            ann1 = json.load(f)

        img0_path = data_dir / "images" / f"{name0}.jpg"
        img1_path = data_dir / "images" / f"{name1}.jpg"
        if not img0_path.exists():
            img0_path = img0_path.with_suffix(".png")
        if not img1_path.exists():
            img1_path = img1_path.with_suffix(".png")

        with Image.open(img0_path) as img:
            size0 = [img.width, img.height]
        with Image.open(img1_path) as img:
            size1 = [img.width, img.height]

        products0 = ann0.get("products", [])
        products1 = ann1.get("products", [])

        kpts0 = torch.tensor(
            [[(p["bbox"][0] + p["bbox"][2]) / 2, (p["bbox"][1] + p["bbox"][3]) / 2]
             for p in products0], dtype=torch.float32
        )
        kpts1 = torch.tensor(
            [[(p["bbox"][0] + p["bbox"][2]) / 2, (p["bbox"][1] + p["bbox"][3]) / 2]
             for p in products1], dtype=torch.float32
        )
        desc0 = torch.tensor([p["embedding"] for p in products0], dtype=torch.float32)
        desc1 = torch.tensor([p["embedding"] for p in products1], dtype=torch.float32)

        if len(kpts0) == 0 or len(kpts1) == 0:
            continue

        batch = {
            "keypoints0": kpts0.unsqueeze(0).to(device),
            "keypoints1": kpts1.unsqueeze(0).to(device),
            "descriptors0": desc0.unsqueeze(0).to(device),
            "descriptors1": desc1.unsqueeze(0).to(device),
            "view0": {"image_size": torch.tensor([size0]).to(device)},
            "view1": {"image_size": torch.tensor([size1]).to(device)},
        }

        pred = model(batch)
        matches0 = pred["matches0"][0].cpu().numpy()
        pred_matches = set(
            (i, int(matches0[i])) for i in range(len(matches0)) if matches0[i] != -1
        )

        tp = len(pred_matches & gt_matches)
        fp = len(pred_matches - gt_matches)
        fn = len(gt_matches - pred_matches)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if tp + fp > 0:
            pair_precisions.append(tp / (tp + fp))
        if tp + fn > 0:
            pair_recalls.append(tp / (tp + fn))

    precision = np.mean(pair_precisions) if pair_precisions else 0
    recall = np.mean(pair_recalls) if pair_recalls else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    logger.info(f"[Test] precision={precision:.4f}, recall={recall:.4f}, F1={f1:.4f}")
    logger.info(f"[Test] TP={total_tp}, FP={total_fp}, FN={total_fn}")
    return {"precision": precision, "recall": recall, "f1": f1}


# ── Main training loop ─────────────────────────────────────────────────────────


def train(conf):
    """Train LightGlue with the given configuration dict.

    Args:
        conf: dict with keys 'data', 'model', 'train'. See train_lightglue.py
              for the full configuration schema.

    Returns:
        Path to the output directory containing checkpoints.
    """
    train_conf = conf["train"]
    model_conf = conf["model"]
    data_conf = conf["data"]

    set_seed(train_conf.get("seed", 42))

    # Output directory
    output_dir = Path(train_conf.get("output_dir", "outputs/lightglue"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Dataset
    dataset = ProductPairsDataset(
        data_dir=data_conf["data_dir"],
        embedding_dim=data_conf.get("embedding_dim", model_conf.get("input_dim", 384)),
        max_products=data_conf.get("max_products", 400),
        batch_size=train_conf.get("batch_size", 8),
        num_workers=train_conf.get("num_workers", 4),
        train_ratio=data_conf.get("train_ratio", 0.7),
        val_ratio=data_conf.get("val_ratio", 0.2),
        shuffle_seed=data_conf.get("shuffle_seed", 42),
    )
    train_loader = dataset.get_loader("train")
    val_loader = dataset.get_loader("val")
    logger.info(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Model
    model = LightGlue(model_conf).to(device)

    # Load pretrained weights if specified
    pretrained_path = model_conf.get("pretrained_weights")
    if pretrained_path and Path(pretrained_path).exists():
        cp = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        state = cp.get("model", cp)
        # Handle matcher. prefix from TwoViewPipeline checkpoints
        state = {k.replace("matcher.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded pretrained weights from {pretrained_path}")

    torch.backends.cudnn.benchmark = True

    # Optimizer
    optimizer_name = train_conf.get("optimizer", "adam")
    lr = train_conf.get("lr", 1e-4)
    optimizer_cls = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }[optimizer_name]
    optimizer = optimizer_cls(model.parameters(), lr=lr)

    # LR scheduler
    lr_conf = train_conf.get("lr_schedule", {})
    sched_type = lr_conf.get("type")
    if sched_type == "exp":
        exp_div = lr_conf.get("exp_div_10", 10)
        gamma = 10 ** (-1 / exp_div)
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lambda _: gamma
        )
    elif sched_type == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_conf.get("step_size", 5),
            gamma=lr_conf.get("factor", 0.5),
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lambda _: 1.0
        )
    on_epoch = lr_conf.get("on_epoch", True)

    # Mixed precision
    mp = train_conf.get("mixed_precision")
    use_mp = mp is not None
    mp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(mp, torch.float32)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_mp and device.type == "cuda"))

    # Training state
    epochs = train_conf.get("epochs", 15)
    eval_every = train_conf.get("eval_every_iter", 50)
    log_every = train_conf.get("log_every_iter", 50)
    clip_grad = train_conf.get("clip_grad")
    best_key = train_conf.get("best_key", "loss/total")
    best_eval = float("inf")

    # Graceful interruption
    stop = False
    def _sigint(sig, frame):
        nonlocal stop
        if stop:
            raise KeyboardInterrupt
        logger.info("Interrupt received, finishing current epoch...")
        stop = True
    signal.signal(signal.SIGINT, _sigint)

    logger.info(f"Starting training for {epochs} epochs (lr={lr}, optimizer={optimizer_name})")

    for epoch in range(epochs):
        if stop:
            break

        logger.info(f"Epoch {epoch}/{epochs - 1}")
        set_seed(train_conf.get("seed", 42) + epoch)
        model.train()

        if on_epoch and epoch > 0:
            old_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step()
            logger.info(f"LR: {old_lr:.2e} → {optimizer.param_groups[0]['lr']:.2e}")

        for it, data in enumerate(train_loader):
            if stop:
                break

            optimizer.zero_grad()

            with torch.autocast(
                device_type=device.type,
                enabled=use_mp,
                dtype=mp_dtype,
            ):
                data = batch_to_device(data, device)
                flat = _extract_views(data)
                pred = model({**data, **flat})
                losses, _ = model.loss(pred, {**pred, **data})
                loss = torch.mean(losses["total"])

            if torch.isnan(loss):
                logger.warning(f"NaN loss at epoch {epoch}, iter {it} — skipping")
                continue

            scaler.scale(loss).backward()
            if clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()

            if not on_epoch:
                lr_scheduler.step()

            # Logging
            if it % log_every == 0:
                loss_strs = [f"{k}={torch.mean(v).item():.4f}" for k, v in losses.items()]
                logger.info(f"  [iter {it}] {', '.join(loss_strs)}")

            # Validation
            if it > 0 and it % eval_every == 0:
                with fork_rng(seed=train_conf.get("seed", 42)):
                    results = evaluate(model, val_loader, device)
                str_results = [f"{k}={v:.4f}" for k, v in results.items() if isinstance(v, float)]
                logger.info(f"  [val] {', '.join(str_results)}")
                if results.get(best_key, float("inf")) < best_eval:
                    best_eval = save_checkpoint(
                        model, optimizer, lr_scheduler, conf, results,
                        best_eval, epoch, output_dir, name="lightglue_best.tar",
                    )
                model.train()

        # End-of-epoch validation & checkpoint
        with fork_rng(seed=train_conf.get("seed", 42)):
            results = evaluate(model, val_loader, device)
        str_results = [f"{k}={v:.4f}" for k, v in results.items() if isinstance(v, float)]
        logger.info(f"  [val epoch {epoch}] {', '.join(str_results)}")
        best_eval = save_checkpoint(
            model, optimizer, lr_scheduler, conf, results,
            best_eval, epoch, output_dir,
        )
        torch.cuda.empty_cache()

    logger.info("Training complete.")

    # Load best and run test evaluation
    best_path = output_dir / "lightglue_best.tar"
    if best_path.exists():
        cp = torch.load(str(best_path), map_location=device, weights_only=False)
        model.load_state_dict(cp["model"])
        logger.info(f"Loaded best checkpoint (epoch {cp.get('epoch', '?')})")

    test_results = test_evaluate(model, dataset, conf, device)

    return output_dir
