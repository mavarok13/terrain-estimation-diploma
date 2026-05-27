from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset.input_modes import (
    build_model_input_for_mode,
    encode_pair_metadata_maps,
    encode_sun_metadata_maps,
    input_channels_for_mode,
    input_mode_requires_metadata,
    input_mode_requires_pair,
    input_mode_requires_shadow,
    resolve_input_mode,
)
from src.models import UNet
from src.utils.io import ensure_dir, load_checkpoint, resolve_repo_path
from src.utils.metadata import encode_metadata_vector
from src.utils.visualization import save_inference_outputs


def _read_rgb(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).float()


def _read_mask(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("L")
    image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array[None, ...]).float()


def run_inference(
    config: dict,
    checkpoint_path: str | Path,
    image_path: str | Path,
    image_alt_path: str | Path | None,
    metadata_path: str | Path | None,
    shadow_mask_path: str | Path | None = None,
    shadow_mask_alt_path: str | Path | None = None,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    train_config = ckpt["train_config"]
    dataset_cfg = train_config["dataset"]
    model_cfg = train_config["model"]

    image_size = int(config["dataset"].get("image_size", dataset_cfg["image_size"]))
    train_training_cfg = train_config.get("training", {})
    input_mode = config.get("training", {}).get("input_mode") or config["dataset"].get("input_mode")
    input_mode = input_mode or train_training_cfg.get("input_mode")
    if input_mode is None:
        input_mode = resolve_input_mode(train_training_cfg, dataset_cfg)

    metadata_keys = list(dataset_cfg.get("metadata_keys", []))
    input_channels = int(model_cfg.get("input_channels", input_channels_for_mode(input_mode, metadata_keys)))
    model = UNet(in_channels=input_channels, out_channels=1, base_channels=int(model_cfg["base_channels"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    image_t = _read_rgb(resolve_repo_path(image_path, repo_root), image_size)
    image_alt_t = None
    if input_mode_requires_pair(input_mode):
        if image_alt_path is None:
            raise ValueError(f"{input_mode} requires --image-alt")
        image_alt_t = _read_rgb(resolve_repo_path(image_alt_path, repo_root), image_size)

    metadata_t = None
    if input_mode_requires_metadata(input_mode):
        if metadata_path is None:
            raise ValueError(f"{input_mode} requires --metadata")
        with resolve_repo_path(metadata_path, repo_root).open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if input_mode == "grayscale_shadow_sun":
            metadata_t = encode_sun_metadata_maps(meta)
        elif input_mode in {"grayscale_pair_shadow_sun", "grayscale_pair_shadowmask_sun", "rgb_pair_metadata"}:
            metadata_t = encode_pair_metadata_maps(meta)
        else:
            metadata_t = torch.from_numpy(encode_metadata_vector(meta, metadata_keys))

    shadow_mask_t = None
    shadow_mask_alt_t = None
    if input_mode_requires_shadow(input_mode):
        if shadow_mask_path is None:
            raise ValueError(f"{input_mode} requires --shadow-mask")
        shadow_mask_t = _read_mask(resolve_repo_path(shadow_mask_path, repo_root), image_size)
        if input_mode == "grayscale_pair_shadowmask_sun":
            if shadow_mask_alt_path is None:
                raise ValueError(f"{input_mode} requires --shadow-mask-alt")
            shadow_mask_alt_t = _read_mask(resolve_repo_path(shadow_mask_alt_path, repo_root), image_size)

    model_input = build_model_input_for_mode(
        image_t,
        image_alt_t,
        metadata_t,
        input_mode,
        shadow_mask=shadow_mask_t,
        shadow_mask_alt=shadow_mask_alt_t,
    ).unsqueeze(0)
    with torch.no_grad():
        pred = model(model_input).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    output_dir = resolve_repo_path(config["inference"]["output_dir"], repo_root)
    ensure_dir(output_dir)
    save_inference_outputs(output_dir=output_dir, image=image_t, prediction=pred)
    print(f"Saved inference outputs to: {output_dir}")
