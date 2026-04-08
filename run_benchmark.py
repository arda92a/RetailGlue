#!/usr/bin/env python3
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')

from retailglue.benchmark.runner import run_benchmark
from retailglue.benchmark.constants import COMBINATIONS, DEVICE


def main():
    parser = argparse.ArgumentParser(description="RetailGlue Stitching Benchmark")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (e.g. lightglue_dinov3_vits, roma_v2, gluestick)")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to use")
    parser.add_argument("--data-root", type=str, default=None, help="Benchmark data root directory")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    if args.model:
        # Find matching combination from COMBINATIONS, or build one with default detector
        match = [c for c in COMBINATIONS if c["model_name"] == args.model]
        if match:
            combinations = [dict(match[0], device=args.device)]
        else:
            combinations = [{"model_name": args.model, "device": args.device, "detector": "sku_yolo_detector"}]
    else:
        combinations = COMBINATIONS

    config = None
    if args.config:
        from retailglue.config import get_config
        config = get_config(args.config)

    run_benchmark(combinations=combinations, data_root=args.data_root, config=config)


if __name__ == "__main__":
    main()
