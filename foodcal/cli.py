"""Command-line entrypoints for the food-calorie course project."""
import argparse
import json
from pathlib import Path
import platform
import sys

from .data import load_config, make_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--config", default="configs/default.json")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", default="configs/default.json")
    prepare.add_argument("--images", default="data/raw/food-101")
    prepare.add_argument("--output", default="data/processed/food101_10")
    train = sub.add_parser("train")
    train.add_argument("--config", default="configs/default.json")
    train.add_argument("--manifest", default="data/processed/food101_10/manifest.csv")
    train.add_argument("--output", required=True)
    train.add_argument("--mode", choices=["frozen", "last_block"], default="frozen")
    train.add_argument("--augment", action="store_true")
    train.add_argument("--device", default="auto")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--manifest", default="data/processed/food101_10/manifest.csv")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.command == "check":
        import torch
        import torchvision
        config = load_config(args.config)
        pretrained = Path(config["model"]["pretrained_weights"])
        result = {"project": "food-calorie", "python": platform.python_version(), "python_executable": sys.executable,
                  "torch": torch.__version__, "torchvision": torchvision.__version__,
                  "mps_available": torch.backends.mps.is_available(), "cuda_available": torch.cuda.is_available(),
                  "classes": config["classes"], "pretrained_weights_exist": pretrained.is_file(),
                  "food_checkpoint_exists": Path("models/food10/best.pt").is_file(),
                  "manifest_exists": Path(config["data"]["manifest"]).is_file(),
                  "note": "Environment check only; ImageNet initialization is not food-classifier performance"}
    elif args.command == "prepare":
        result = make_manifest(args.images, args.output, load_config(args.config))
    elif args.command == "train":
        from .training import train as run_train
        result = run_train(args.manifest, load_config(args.config), args.output,
                           mode=args.mode, augment=args.augment, device=args.device)
    else:
        from .training import evaluate as run_evaluate
        result = run_evaluate(args.manifest, args.checkpoint, args.output, split=args.split, device=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
