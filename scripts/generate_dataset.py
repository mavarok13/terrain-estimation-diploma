from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_bootstrap_path()

from src.generation.procedural import generate_dataset
from src.utils.config import apply_dot_overrides, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a procedural terrain dataset")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*", help="Dot overrides, e.g. dataset.num_samples=16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    apply_dot_overrides(config, args.overrides)
    generate_dataset(config)


if __name__ == "__main__":
    main()
