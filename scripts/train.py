from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_bootstrap_path()

from src.training.engine import train_from_config
from src.utils.config import apply_dot_overrides, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a terrain height estimation model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("overrides", nargs="*", help="Dot overrides, e.g. training.input_mode=rgb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    apply_dot_overrides(config, args.overrides)
    train_from_config(config, resume_checkpoint=args.resume)


if __name__ == "__main__":
    main()
