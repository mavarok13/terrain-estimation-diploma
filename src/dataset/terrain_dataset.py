from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.metadata import encode_metadata_vector, encoded_metadata_size


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image_path: Path
    image_alt_path: Path | None
    height_path: Path
    shadow_path: Path
    meta_path: Path


def _read_rgb(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _read_mask(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _read_height(path: Path, image_size: int) -> np.ndarray:
    height = np.load(path).astype(np.float32)
    height = cv2.resize(height, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return np.clip(height, 0.0, 1.0)


def build_model_input(
    image: torch.Tensor,
    image_alt: torch.Tensor | None,
    metadata: torch.Tensor | None,
) -> torch.Tensor:
    channels = [image]
    if image_alt is not None:
        channels.append(image_alt)
    if metadata is not None:
        if metadata.ndim != 1:
            raise ValueError(f"Expected metadata tensor [C], got {tuple(metadata.shape)}")
        h, w = image.shape[1], image.shape[2]
        meta_maps = metadata[:, None, None].expand(metadata.shape[0], h, w)
        channels.append(meta_maps)
    return torch.cat(channels, dim=0)


class TerrainDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        manifest_path: str | Path,
        image_size: int,
        metadata_keys: list[str],
        use_pair: bool,
        use_metadata: bool,
        split: str,
        shadow_augmentation: bool = False,
    ) -> None:
        self.root = Path(root)
        self.manifest_path = Path(manifest_path)
        self.image_size = int(image_size)
        self.metadata_keys = list(metadata_keys)
        self.use_pair = bool(use_pair)
        self.use_metadata = bool(use_metadata)
        self.split = split
        self.shadow_augmentation = bool(shadow_augmentation and split == "train")

        self.records = self._load_records()
        self.metadata_dim = encoded_metadata_size(self.metadata_keys) if self.use_metadata else 0

    def _load_records(self) -> list[SampleRecord]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        records: list[SampleRecord] = []
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sample_dir = self.root / row["sample_dir"]
                image_alt_rel = row.get("image_alt_relpath", "").strip()
                records.append(
                    SampleRecord(
                        sample_id=row["sample_id"],
                        image_path=sample_dir / row["image_relpath"],
                        image_alt_path=(sample_dir / image_alt_rel) if image_alt_rel else None,
                        height_path=sample_dir / row["height_relpath"],
                        shadow_path=sample_dir / row["shadow_relpath"],
                        meta_path=sample_dir / row["meta_relpath"],
                    )
                )

        if not records:
            raise RuntimeError(f"No dataset samples found in {self.manifest_path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _augment_shadow_response(self, image: np.ndarray, shadow_mask: np.ndarray) -> np.ndarray:
        strength = np.random.uniform(0.7, 1.25)
        softness = np.clip(shadow_mask[..., None], 0.0, 1.0)
        image = image * (1.0 - softness + softness * strength)
        return np.clip(image, 0.0, 1.0)

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        gain = np.random.uniform(0.85, 1.15)
        bias = np.random.uniform(-0.06, 0.06)
        gamma = np.random.uniform(0.85, 1.15)
        image = np.clip(image * gain + bias, 0.0, 1.0)
        image = np.power(image, gamma)
        return np.clip(image, 0.0, 1.0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image = _read_rgb(record.image_path, self.image_size)
        shadow_mask = _read_mask(record.shadow_path, self.image_size)
        height = _read_height(record.height_path, self.image_size)

        image_alt = None
        if self.use_pair:
            if record.image_alt_path is None:
                raise RuntimeError(f"Sample {record.sample_id} has no alternate image")
            image_alt = _read_rgb(record.image_alt_path, self.image_size)

        if self.shadow_augmentation:
            image = self._augment_image(image)
            image = self._augment_shadow_response(image, shadow_mask)
            if image_alt is not None:
                image_alt = self._augment_image(image_alt)

        with record.meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)

        metadata = None
        if self.use_metadata:
            metadata_np = encode_metadata_vector(meta, self.metadata_keys)
            metadata = torch.from_numpy(metadata_np)

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image_alt_t = torch.from_numpy(image_alt.transpose(2, 0, 1)).float() if image_alt is not None else None
        height_t = torch.from_numpy(height[None, ...]).float()
        shadow_t = torch.from_numpy(shadow_mask[None, ...]).float()
        model_input = build_model_input(image_t, image_alt_t, metadata)

        output: dict[str, torch.Tensor | str] = {
            "input": model_input,
            "image": image_t,
            "height": height_t,
            "shadow_mask": shadow_t,
            "sample_id": record.sample_id,
        }
        if image_alt_t is not None:
            output["image_alt"] = image_alt_t
        if metadata is not None:
            output["metadata"] = metadata
        return output
