from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_bootstrap_path()

from src.inference.predict import run_inference
from src.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run terrain height inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--image-alt", type=str, default=None)
    parser.add_argument("--metadata", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    run_inference(
        config=config,
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        image_alt_path=args.image_alt,
        metadata_path=args.metadata,
    )


if __name__ == "__main__":
    main()
