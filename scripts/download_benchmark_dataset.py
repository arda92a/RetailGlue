"""Download the RemRetail100_v2 benchmark images from the Hugging Face Hub
into ``data/benchmark/images/`` so ``run_benchmark.py`` can use them directly.

Usage:
    python scripts/download_benchmark_dataset.py
    python scripts/download_benchmark_dataset.py --output-dir data/benchmark/images
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

DEFAULT_REPO = "arda92/RemRetail100_v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmark/images"),
        help="Target directory for the extracted JPG files.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] downloading {args.repo_id} [{args.split}]")
    ds = load_dataset(args.repo_id, split=args.split, revision=args.revision)
    print(f"[info] writing {len(ds)} images to {args.output_dir}")
    for row in ds:
        ds_name = row.get("file_name") or f"{row['sequence_id']}_{int(row['frame_index']):02d}.jpg"
        row["image"].save(args.output_dir / ds_name)
    print("[done] benchmark images ready.")


if __name__ == "__main__":
    main()
