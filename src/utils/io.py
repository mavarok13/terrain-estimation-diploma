from __future__ import annotations

from pathlib import Path

import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_repo_path(path_value: str | Path, repo_root: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path(repo_root) / path).resolve()


def save_checkpoint(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)
