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
from src.utils.config import apply_dot_overrides, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run terrain height inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--image-alt", type=str, default=None)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--shadow-mask", type=str, default=None)
    parser.add_argument("--shadow-mask-alt", type=str, default=None)
    parser.add_argument("overrides", nargs="*", help="Dot overrides, e.g. dataset.input_mode=rgb_pair")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    apply_dot_overrides(config, args.overrides)
    run_inference(
        config=config,
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        image_alt_path=args.image_alt,
        metadata_path=args.metadata,
        shadow_mask_path=args.shadow_mask,
        shadow_mask_alt_path=args.shadow_mask_alt,
    )


if __name__ == "__main__":
    main()
