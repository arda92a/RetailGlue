"""Train LightGlue for product-level feature matching.

Usage:
    python train_lightglue.py --config config.yaml
    python train_lightglue.py --data_dir /path/to/dataset --epochs 15
    python train_lightglue.py --config config.yaml --lr 5e-5 --mixed_precision float16
"""

import argparse
import logging
import yaml
from pathlib import Path

from retailglue.training.trainer import train


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Train LightGlue product matcher")

    # Config file (optional — CLI args override)
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML config file")

    # Data
    parser.add_argument("--data_dir", type=str, help="Path to training dataset")

    # Model
    parser.add_argument("--input_dim", type=int, help="Embedding input dimension")
    parser.add_argument("--descriptor_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=9)
    parser.add_argument("--pretrained_weights", type=str,
                        help="Path to pretrained LightGlue checkpoint for fine-tuning")

    # Training
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"])
    parser.add_argument("--mixed_precision", "--mp", type=str,
                        choices=["float16", "bfloat16"], default=None)
    parser.add_argument("--clip_grad", type=float, help="Gradient clipping norm")
    parser.add_argument("--output_dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--experiment_name", type=str, default="lightglue")

    args = parser.parse_args()

    # Load base config from YAML if available
    conf = {"data": {}, "model": {}, "train": {}}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            yaml_conf = yaml.safe_load(f) or {}
        # Extract training section from RetailGlue config
        if "training" in yaml_conf:
            training_yaml = yaml_conf["training"]
            conf["data"] = training_yaml.get("data", {})
            conf["model"] = training_yaml.get("model", {})
            conf["train"] = training_yaml.get("train", {})

    # CLI overrides
    if args.data_dir:
        conf["data"]["data_dir"] = args.data_dir
    if args.input_dim:
        conf["model"]["input_dim"] = args.input_dim
        conf["data"]["embedding_dim"] = args.input_dim
    if args.descriptor_dim:
        conf["model"]["descriptor_dim"] = args.descriptor_dim
    if args.n_layers:
        conf["model"]["n_layers"] = args.n_layers
    if args.pretrained_weights:
        conf["model"]["pretrained_weights"] = args.pretrained_weights
    if args.epochs:
        conf["train"]["epochs"] = args.epochs
    if args.batch_size:
        conf["train"]["batch_size"] = args.batch_size
    if args.lr:
        conf["train"]["lr"] = args.lr
    if args.optimizer:
        conf["train"]["optimizer"] = args.optimizer
    if args.mixed_precision:
        conf["train"]["mixed_precision"] = args.mixed_precision
    if args.clip_grad:
        conf["train"]["clip_grad"] = args.clip_grad
    if args.output_dir:
        conf["train"]["output_dir"] = args.output_dir
    if args.experiment_name:
        conf["train"]["experiment_name"] = args.experiment_name

    # Validate required config
    if "data_dir" not in conf["data"]:
        parser.error("--data_dir is required (or set training.data.data_dir in config)")
    if "input_dim" not in conf["model"]:
        parser.error("--input_dim is required (or set training.model.input_dim in config)")

    train(conf)


if __name__ == "__main__":
    main()
