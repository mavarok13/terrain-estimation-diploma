from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset.terrain_dataset import build_model_input
from src.models import UNet
from src.utils.io import ensure_dir, load_checkpoint, resolve_repo_path
from src.utils.metadata import encode_metadata_vector, encoded_metadata_size
from src.utils.visualization import save_inference_outputs


def _read_rgb(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).float()


def run_inference(
    config: dict,
    checkpoint_path: str | Path,
    image_path: str | Path,
    image_alt_path: str | Path | None,
    metadata_path: str | Path | None,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    train_config = ckpt["train_config"]
    dataset_cfg = train_config["dataset"]
    model_cfg = train_config["model"]

    image_size = int(config["dataset"].get("image_size", dataset_cfg["image_size"]))
    use_pair = bool(config["dataset"].get("use_pair", dataset_cfg["use_pair"]))
    use_metadata = bool(config["dataset"].get("use_metadata", dataset_cfg["use_metadata"]))
    metadata_keys = list(config["dataset"].get("metadata_keys", dataset_cfg["metadata_keys"]))

    input_channels = 3 * (2 if use_pair else 1) + (encoded_metadata_size(metadata_keys) if use_metadata else 0)
    model = UNet(in_channels=input_channels, out_channels=1, base_channels=int(model_cfg["base_channels"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    image_t = _read_rgb(resolve_repo_path(image_path, repo_root), image_size)
    image_alt_t = None
    if use_pair:
        if image_alt_path is None:
            raise ValueError("Pair mode is enabled but --image-alt was not provided")
        image_alt_t = _read_rgb(resolve_repo_path(image_alt_path, repo_root), image_size)

    metadata_t = None
    if use_metadata:
        if metadata_path is None:
            raise ValueError("Metadata conditioning is enabled but --metadata was not provided")
        with resolve_repo_path(metadata_path, repo_root).open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        metadata_t = torch.from_numpy(encode_metadata_vector(meta, metadata_keys))

    model_input = build_model_input(image_t, image_alt_t, metadata_t).unsqueeze(0)
    with torch.no_grad():
        pred = model(model_input).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    output_dir = resolve_repo_path(config["inference"]["output_dir"], repo_root)
    ensure_dir(output_dir)
    save_inference_outputs(output_dir=output_dir, image=image_t, prediction=pred)
    print(f"Saved inference outputs to: {output_dir}")
