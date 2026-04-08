# RetailGlue: Semantic Product-Level Image Stitching for Retail Shelf Panoramas

<p align="center">
  <a href="https://arda92a.github.io/retailglue_web/"><b>Project Page</b></a> &nbsp;|&nbsp;
  <a href="#citation"><b>Paper</b></a> &nbsp;|&nbsp;
  <a href="#quick-start"><b>Quick Start</b></a>
</p>

<p align="center">
  <b>CVPR 2026 &mdash; Image Matching: Local Features & Beyond Workshop</b>
</p>

<p align="center">
  <img src="figures/pipeline_retailglue.svg" alt="RetailGlue Pipeline" width="100%">
</p>

<p align="center"><i>Figure 1. RetailGlue replaces ambiguous low-level feature matching with semantic product correspondences for robust retail shelf stitching.</i></p>

> Panoramic shelf images are fundamental to retail analytics, yet constructing them remains a persistent challenge due to repetitive product patterns that confound stitching methods based on local feature matches. We propose a product-level image stitching pipeline that replaces ambiguous low-level feature matching with semantic product correspondences.

## Highlights

- **88.4% F1 Score** with DINOv3 ViT-B/16 — best among all methods
- **20x speedup** over dense matchers (RoMa v2)
- **7x speedup** over sparse baselines (DISK + LightGlue)

## Overview

RetailGlue is a semantic product-level image stitching pipeline for retail shelves. Instead of relying on local features (SIFT, ORB, SuperPoint), it leverages:

- **YOLOv11l** trained on SKU-110K for product detection
- **DINOv3** embeddings as semantic product descriptors
- **Fine-tuned LightGlue** for product-level correspondence matching
- **Graph-based multi-image stitching** with automatic panorama partitioning

This approach achieves a **20x runtime speedup** over dense matchers and **7x** over sparse baselines while delivering higher F1 scores (88.4% with ViT-B/16).

## Repository Structure

```
retailglue/
├── entities.py              # Core data types (Point, Polygon, BoundingBox, Detection)
├── config.py                # YAML configuration loader
├── io.py                    # Image I/O utilities
├── detector.py              # SKU YOLO product detector
├── embeddings.py            # DINOv3 embedding extraction
├── visualization.py         # Drawing and visualization
├── matchers/
│   ├── __init__.py          # Matcher factory
│   ├── lightglue.py         # Product-level LightGlue (core contribution)
│   ├── lightgluestick.py    # LightGlueStick baseline
│   ├── gluestick.py         # GlueStick baseline
│   ├── roma.py              # RoMa v2 baseline
│   └── hf_model.py          # HuggingFace models (SuperGlue, SP+LG, DISK, ELoFTR, MINIMA)
├── stitching/
│   ├── stitcher.py          # Core stitching engine (graph-based multi-panorama)
│   ├── blender.py           # Adaptive distance-transform blending
│   └── transforms.py        # Homography-based detection transformation
└── benchmark/
    ├── runner.py             # Benchmark orchestrator
    ├── evaluation.py         # IOU matching, Hungarian assignment
    ├── stats.py              # Precision, Recall, F1 computation
    └── drawer.py             # Result visualization
```

## Installation

```bash
# Using uv (recommended)
uv sync

# With all extras (RoMa, HuggingFace models, dev tools):
uv sync --all-extras
```

## Quick Start

### Interactive Interface

```bash
uv run python run_interface.py
```

Opens a Gradio web interface where you can upload shelf images and select stitching models.

### Run Benchmark

```bash
# Run all model combinations
uv run python run_benchmark.py

# Run a specific model
uv run python run_benchmark.py --model lightglue_dino --device cuda
```

## Models

### Core (Our Method)
| Model | Descriptor | Dim | Description |
|-------|-----------|-----|-------------|
| `lightglue_dino` | DINOv3 ViT-S/16 | 384 | Default, fastest |
| `lightglue_dino_vitb` | DINOv3 ViT-B/16 | 768 | Best F1 (88.4%) |
| `lightglue_dino_vitl` | DINOv3 ViT-L/16 | 1024 | Largest variant |
| `lightglue_dinov2` | DINOv2 ViT-S/14 | 384 | DINOv2 baseline |

### Baselines
| Model | Type |
|-------|------|
| `lightgluestick` | SuperPoint + LSD line segments |
| `gluestick` | SuperPoint + line matching |
| `roma_v2` | Dense (RoMa v2) |
| `superglue` | SuperPoint + SuperGlue (HF) |
| `lightglue_superpoint` | SuperPoint + LightGlue (HF) |
| `lightglue_disk` | DISK + LightGlue (HF) |
| `lightglue_minima` | MINIMA + LightGlue (HF) |
| `eloftr` | EfficientLoFTR (HF) |

## Model Weights

Place model weights in the following structure:

```
weights/
├── sku_yolo/
│   └── best_sku110k.pt              # YOLOv11l trained on SKU-110K
├── dino/
│   ├── dinov3_vits16_pretrain.pth   # DINOv3 ViT-S/16 (384-dim)
│   ├── dinov3_vitb16_pretrain.pth   # DINOv3 ViT-B/16 (768-dim)
│   ├── dinov3_vitl16_pretrain.pth   # DINOv3 ViT-L/16 (1024-dim)
│   └── dinov2_vits14_pretrain.pth   # DINOv2 ViT-S/14 (384-dim)
└── lightglue/
    ├── lightglue_dino.pth           # Fine-tuned for DINOv3 ViT-S (input_dim=384)
    ├── lightglue_dino_vitb.pth      # Fine-tuned for DINOv3 ViT-B (input_dim=768)
    ├── lightglue_dino_vitl.pth      # Fine-tuned for DINOv3 ViT-L (input_dim=1024)
    └── lightglue_dinov2.pth         # Fine-tuned for DINOv2 ViT-S (input_dim=384)
```

All paths are relative to the repository root and configured in `config.yaml`.
If DINO weights are not found locally, they will be downloaded from `torch.hub` automatically.
LightGlue fine-tuned weights are **required** for product-level matching.

## Qualitative Results

<p align="center">
  <img src="figures/success_cases.svg" alt="Success Case" width="50%">
</p>

<p align="center"><i>Figure 2. In a densely packed, highly repetitive scene, the local feature matcher (DISK + LightGlue) fails due to structural ambiguity, while RetailGlue successfully resolves identical products for a perfectly aligned panorama.</i></p>

<p align="center">
  <img src="figures/multi_panorama_stitch.svg" alt="Multi-panorama" width="100%">
</p>

<p align="center"><i>Figure 3. Multi-panorama generation via graph partitioning. Given a challenging 8-frame sequence with insufficient overlap, our pipeline naturally divides the sequence into three geometrically consistent sub-panoramas.</i></p>

## Citation

```bibtex
@inproceedings{retailglue2026,
  title={RetailGlue: Semantic Product-Level Image Stitching for Retail Shelf Panoramas},
  author={{\"O}zt{\"u}ner, Arda and Yal{\c{c}}{\i}ner, {\.I}brahim {\c{S}}amil and Calab, Server and {\c{C}}elik, Ayberk},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2026}
}
```

## License

Apache 2.0

---

<p align="center">
  <a href="https://arda92a.github.io/retailglue_web/">Project Page</a> &nbsp;&bull;&nbsp;
  REM People &nbsp;&bull;&nbsp;
  CVPR 2026
</p>
