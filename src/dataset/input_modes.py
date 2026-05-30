from __future__ import annotations

import numpy as np
import torch

from src.utils.metadata import encode_metadata_value, encoded_metadata_size


SUPPORTED_INPUT_MODES = {
    "rgb": 3,
    "grayscale": 1,
    "grayscale_shadow_sun": 4,
    "grayscale_pair_shadow_sun": 7,
    "grayscale_pair_shadowmask_sun": 8,
    "rgb_pair_shadowmask_sun": 12,
    "rgb_pair": 6,
    "grayscale_pair": 2,
    "rgb_pair_metadata": 10,
    "rgb_pair_full_metadata": None,
}


def resolve_input_mode(training_cfg: dict | None, dataset_cfg: dict | None = None) -> str:
    training_cfg = training_cfg or {}
    dataset_cfg = dataset_cfg or {}
    mode = training_cfg.get("input_mode") or dataset_cfg.get("input_mode")
    if mode is None:
        use_pair = bool(dataset_cfg.get("use_pair", False))
        use_metadata = bool(dataset_cfg.get("use_metadata", False))
        if use_pair and use_metadata:
            mode = "rgb_pair_full_metadata"
        elif use_pair:
            mode = "rgb_pair"
        else:
            mode = "rgb"
    if mode not in SUPPORTED_INPUT_MODES:
        supported = ", ".join(sorted(SUPPORTED_INPUT_MODES))
        raise ValueError(f"Unsupported input_mode '{mode}'. Supported modes: {supported}")
    return str(mode)


def input_channels_for_mode(input_mode: str, metadata_keys: list[str] | None = None) -> int:
    if input_mode not in SUPPORTED_INPUT_MODES:
        supported = ", ".join(sorted(SUPPORTED_INPUT_MODES))
        raise ValueError(f"Unsupported input_mode '{input_mode}'. Supported modes: {supported}")
    if input_mode == "rgb_pair_full_metadata":
        if metadata_keys is None:
            raise ValueError("rgb_pair_full_metadata requires metadata_keys to compute input channels")
        return 6 + encoded_metadata_size(metadata_keys)
    channels = SUPPORTED_INPUT_MODES[input_mode]
    if channels is None:
        raise AssertionError(f"Missing channel rule for input_mode '{input_mode}'")
    return channels


def input_mode_requires_pair(input_mode: str) -> bool:
    return input_mode in {
        "rgb_pair",
        "grayscale_pair",
        "grayscale_pair_shadow_sun",
        "grayscale_pair_shadowmask_sun",
        "rgb_pair_shadowmask_sun",
        "rgb_pair_metadata",
        "rgb_pair_full_metadata",
    }


def input_mode_requires_metadata(input_mode: str) -> bool:
    return input_mode in {
        "grayscale_shadow_sun",
        "grayscale_pair_shadow_sun",
        "grayscale_pair_shadowmask_sun",
        "rgb_pair_shadowmask_sun",
        "rgb_pair_metadata",
        "rgb_pair_full_metadata",
    }


def input_mode_requires_shadow(input_mode: str) -> bool:
    return input_mode in {
        "grayscale_shadow_sun",
        "grayscale_pair_shadow_sun",
        "grayscale_pair_shadowmask_sun",
        "rgb_pair_shadowmask_sun",
    }


def rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    if image.shape[0] != 3:
        raise ValueError(f"Expected RGB tensor [3,H,W], got {tuple(image.shape)}")
    weights = image.new_tensor([0.299, 0.587, 0.114])[:, None, None]
    return (image * weights).sum(dim=0, keepdim=True)


def encode_pair_metadata_maps(metadata: dict) -> torch.Tensor:
    keys = [
        "sun_azimuth_deg",
        "sun_elevation_deg",
        "sun_azimuth_alt_deg",
        "sun_elevation_alt_deg",
    ]
    values = []
    for key in keys:
        if key not in metadata:
            raise KeyError(f"Missing metadata field required by pair sun metadata modes: {key}")
        values.append(encode_metadata_value(key, float(metadata[key])))
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def encode_sun_metadata_maps(metadata: dict) -> torch.Tensor:
    keys = ["sun_azimuth_deg", "sun_elevation_deg"]
    values = []
    for key in keys:
        if key not in metadata:
            raise KeyError(f"Missing metadata field required by grayscale_shadow_sun: {key}")
        values.append(encode_metadata_value(key, float(metadata[key])))
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def build_model_input_for_mode(
    image: torch.Tensor,
    image_alt: torch.Tensor | None,
    metadata: torch.Tensor | None,
    input_mode: str,
    shadow_mask: torch.Tensor | None = None,
    shadow_mask_alt: torch.Tensor | None = None,
) -> torch.Tensor:
    if input_mode == "rgb":
        return image
    if input_mode == "grayscale":
        return rgb_to_grayscale(image)
    if input_mode == "grayscale_shadow_sun":
        if shadow_mask is None:
            raise ValueError("grayscale_shadow_sun requires a shadow mask")
        if metadata is None:
            raise ValueError("grayscale_shadow_sun requires metadata")
        if metadata.ndim != 1:
            raise ValueError(f"Expected metadata tensor [C], got {tuple(metadata.shape)}")
        h, w = image.shape[1], image.shape[2]
        meta_maps = metadata[:, None, None].expand(metadata.shape[0], h, w)
        return torch.cat([rgb_to_grayscale(image), shadow_mask, meta_maps], dim=0)
    if input_mode == "rgb_pair":
        if image_alt is None:
            raise ValueError("rgb_pair requires an alternate image")
        return torch.cat([image, image_alt], dim=0)
    if input_mode == "grayscale_pair":
        if image_alt is None:
            raise ValueError("grayscale_pair requires an alternate image")
        return torch.cat([rgb_to_grayscale(image), rgb_to_grayscale(image_alt)], dim=0)
    if input_mode in {"grayscale_pair_shadow_sun", "grayscale_pair_shadowmask_sun"}:
        if image_alt is None:
            raise ValueError(f"{input_mode} requires an alternate image")
        if shadow_mask is None:
            raise ValueError(f"{input_mode} requires a shadow mask")
        if input_mode == "grayscale_pair_shadowmask_sun" and shadow_mask_alt is None:
            raise ValueError(f"{input_mode} requires an alternate shadow mask")
        if metadata is None:
            raise ValueError(f"{input_mode} requires metadata")
        if metadata.ndim != 1:
            raise ValueError(f"Expected metadata tensor [C], got {tuple(metadata.shape)}")
        h, w = image.shape[1], image.shape[2]
        meta_maps = metadata[:, None, None].expand(metadata.shape[0], h, w)
        channels = [rgb_to_grayscale(image), rgb_to_grayscale(image_alt), shadow_mask]
        if input_mode == "grayscale_pair_shadowmask_sun":
            channels.append(shadow_mask_alt)
        channels.append(meta_maps)
        return torch.cat(channels, dim=0)
    if input_mode == "rgb_pair_shadowmask_sun":
        if image_alt is None:
            raise ValueError(f"{input_mode} requires an alternate image")
        if shadow_mask is None or shadow_mask_alt is None:
            raise ValueError(f"{input_mode} requires primary and alternate shadow masks")
        if metadata is None:
            raise ValueError(f"{input_mode} requires metadata")
        if metadata.ndim != 1:
            raise ValueError(f"Expected metadata tensor [C], got {tuple(metadata.shape)}")
        h, w = image.shape[1], image.shape[2]
        meta_maps = metadata[:, None, None].expand(metadata.shape[0], h, w)
        return torch.cat([image, image_alt, shadow_mask, shadow_mask_alt, meta_maps], dim=0)
    if input_mode in {"rgb_pair_metadata", "rgb_pair_full_metadata"}:
        if image_alt is None:
            raise ValueError(f"{input_mode} requires an alternate image")
        if metadata is None:
            raise ValueError(f"{input_mode} requires metadata")
        if metadata.ndim != 1:
            raise ValueError(f"Expected metadata tensor [C], got {tuple(metadata.shape)}")
        h, w = image.shape[1], image.shape[2]
        meta_maps = metadata[:, None, None].expand(metadata.shape[0], h, w)
        return torch.cat([image, image_alt, meta_maps], dim=0)
    input_channels_for_mode(input_mode)
    raise AssertionError("unreachable")


def required_files_for_mode(input_mode: str) -> list[str]:
    files = ["rgb.png"]
    if input_mode_requires_pair(input_mode):
        files.append("rgb_alt.png or rgb_morning.png/rgb_evening.png via manifest/CLI")
    if input_mode_requires_metadata(input_mode):
        files.append("metadata.json/meta.json")
    if input_mode_requires_shadow(input_mode):
        files.append("shadow_mask.png or manifest shadow_relpath")
    return files
