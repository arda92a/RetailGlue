"""Download a RetailGlue LightGlue dataset from the Hugging Face Hub and
re-materialize the on-disk layout expected by ``ProductPairsDataset``.

The HF repos contain two configs:
    * ``frames`` — one row per annotated frame (image + per-product bboxes/embeddings)
    * ``pairs``  — one row per image pair with ground-truth product matches

This script reconstructs the original structure::

    <output_dir>/
        images/seqXXX_YY.jpg
        annotations/seqXXX_YY.json      # {"products": [{product_id, bbox, embedding}, ...]}
        matches.json                    # {"pairs": [{image0, image1, matches}, ...]}

Usage:
    python scripts/download_lightglue_dataset.py \\
        --repo-id arda92/RetailGlue-lightglue-dinov3-vits \\
        --output-dir data/lightglue/lightglue_dinov3_vits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="HF dataset repo id.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Target directory for the reconstructed dataset.",
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split to pull (default: train)."
    )
    parser.add_argument(
        "--revision", default=None, help="Optional branch/tag/commit on the HF repo."
    )
    args = parser.parse_args()

    out: Path = args.output_dir
    images_dir = out / "images"
    ann_dir = out / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] downloading frames from {args.repo_id}")
    frames = load_dataset(
        args.repo_id, "frames", split=args.split, revision=args.revision
    )

    print(f"[info] writing {len(frames)} images + annotations")
    for row in frames:
        file_name = row["file_name"]
        stem = Path(file_name).stem
        row["image"].save(images_dir / file_name)
        annotation = {
            "products": [
                {
                    "product_id": int(pid),
                    "bbox": [int(v) for v in bbox],
                    "embedding": [float(v) for v in emb],
                }
                for pid, bbox, emb in zip(
                    row["product_ids"], row["bboxes"], row["embeddings"]
                )
            ]
        }
        with open(ann_dir / f"{stem}.json", "w") as f:
            json.dump(annotation, f)

    print(f"[info] downloading pairs from {args.repo_id}")
    pairs = load_dataset(
        args.repo_id, "pairs", split=args.split, revision=args.revision
    )
    matches = {
        "pairs": [
            {
                "image0": row["image0"],
                "image1": row["image1"],
                "matches": [[int(a), int(b)] for a, b in row["matches"]],
            }
            for row in pairs
        ]
    }
    with open(out / "matches.json", "w") as f:
        json.dump(matches, f)

    print(f"[done] dataset written to {out.resolve()}")


if __name__ == "__main__":
    main()
