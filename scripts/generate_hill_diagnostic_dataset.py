from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def _bootstrap_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_bootstrap_path()

from src.generation.procedural import _compute_normals, _render_rgb, _save_png, _terrain_albedo
from src.utils.config import apply_dot_overrides, load_yaml_config
from src.utils.io import ensure_dir, resolve_repo_path


PALETTES = [
    {"id": "grass_rock", "ambient": 0.34, "diffuse": 0.9, "shadow_strength": 0.58, "fog_strength": 0.04},
    {"id": "desert", "ambient": 0.38, "diffuse": 0.82, "shadow_strength": 0.52, "fog_strength": 0.02},
    {"id": "dark_soil", "ambient": 0.30, "diffuse": 0.95, "shadow_strength": 0.62, "fog_strength": 0.03},
]


def _normalize(height: np.ndarray) -> np.ndarray:
    hmin = float(height.min())
    hmax = float(height.max())
    if hmax <= hmin:
        return np.zeros_like(height, dtype=np.float32)
    return ((height - hmin) / (hmax - hmin)).astype(np.float32)


def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)].astype(np.float32)
    return xx, yy


def _hill_height(hill_type: str, size: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    xx, yy = _grid(size)
    cx = float(rng.uniform(-0.25, 0.25))
    cy = float(rng.uniform(-0.25, 0.25))
    radius = float(rng.uniform(0.28, 0.55))
    height_scale = float(rng.uniform(0.45, 1.0))
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy)

    if hill_type == "gaussian":
        height = np.exp(-(r ** 2) / (2.0 * radius ** 2))
    elif hill_type == "cone":
        height = np.clip(1.0 - r / radius, 0.0, 1.0)
    elif hill_type == "frustum":
        height = np.clip(1.0 - r / radius, 0.0, 1.0)
        height = np.maximum(height, 0.45) * (r < radius)
    elif hill_type == "ridge":
        angle = float(rng.uniform(0.0, np.pi))
        rotated = np.cos(angle) * dx + np.sin(angle) * dy
        height = np.exp(-(rotated ** 2) / (2.0 * (radius * 0.35) ** 2)) * np.clip(1.0 - np.abs(dy) * 0.25, 0.0, 1.0)
    elif hill_type == "asymmetric":
        sx = radius * float(rng.uniform(0.45, 0.85))
        sy = radius * float(rng.uniform(0.85, 1.45))
        skew = 1.0 + 0.55 * np.tanh(3.0 * dx)
        height = np.exp(-((dx ** 2) / (2.0 * sx ** 2) + (dy ** 2) / (2.0 * sy ** 2))) * skew
    elif hill_type == "double":
        offset = radius * 0.9
        h1 = np.exp(-(((xx - cx - offset) ** 2 + dy ** 2) / (2.0 * (radius * 0.65) ** 2)))
        h2 = np.exp(-(((xx - cx + offset) ** 2 + dy ** 2) / (2.0 * (radius * 0.55) ** 2)))
        height = h1 + 0.85 * h2
    elif hill_type in {"crater", "valley"}:
        mound = np.exp(-(r ** 2) / (2.0 * radius ** 2))
        pit = np.exp(-(r ** 2) / (2.0 * (radius * 0.38) ** 2))
        height = mound - 1.15 * pit
    else:
        raise ValueError(f"Unsupported diagnostic hill type: {hill_type}")

    height = _normalize(height) * height_scale
    return height.astype(np.float32), {
        "hill_type": hill_type,
        "hill_height": height_scale,
        "hill_radius": radius,
        "hill_center": [cx, cy],
    }


def _variant_settings(index: int, cfg: dict, rng: np.random.Generator) -> tuple[str, int, float, float, float, float]:
    variant_kind = ["same_geometry_different_colors", "same_color_different_suns", "morning_evening_pair", "different_geometry_similar_color"][index % 4]
    palette_id = index % len(PALETTES)
    if not bool(cfg.get("randomize_albedo", True)):
        palette_id = 0
    sun_az = float(rng.uniform(0.0, 360.0)) if bool(cfg.get("randomize_sun", True)) else 135.0
    sun_el = float(rng.uniform(12.0, 55.0)) if bool(cfg.get("randomize_sun", True)) else 28.0
    morning_az = 95.0 + float(rng.uniform(-12.0, 12.0))
    evening_az = 265.0 + float(rng.uniform(-12.0, 12.0))
    return variant_kind, palette_id, sun_az, sun_el, morning_az, evening_az


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a controlled single-hill diagnostic dataset")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*", help="Dot overrides, e.g. diagnostic.num_geometries=4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    apply_dot_overrides(config, args.overrides)
    diag = config["diagnostic"]
    rng = np.random.default_rng(int(config.get("seed", 123)))
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = resolve_repo_path(diag["output_dir"], repo_root)
    samples_root = output_dir / "samples"
    ensure_dir(samples_root)

    hill_types = list(diag["hill_types"])
    rows: list[dict[str, str]] = []
    image_size = int(diag["image_size"])
    num_geometries = int(diag["num_geometries"])
    variants_per_geometry = int(diag["variants_per_geometry"])

    for geometry_idx in range(num_geometries):
        hill_type = str(hill_types[geometry_idx % len(hill_types)])
        base_height, hill_meta = _hill_height(hill_type, image_size, rng)
        for variant_idx in range(variants_per_geometry):
            sample_id = f"hill_{geometry_idx:04d}_{variant_idx:02d}"
            sample_dir = samples_root / sample_id
            ensure_dir(sample_dir)
            for stale_name in ("rgb.png", "shadow_mask.png"):
                stale_path = sample_dir / stale_name
                if stale_path.exists():
                    stale_path.unlink()
            variant_kind, palette_idx, sun_az, sun_el, morning_az, evening_az = _variant_settings(variant_idx, diag, rng)
            height = base_height
            if variant_kind == "different_geometry_similar_color":
                alt_type = str(hill_types[(geometry_idx + variant_idx + 1) % len(hill_types)])
                height, hill_meta = _hill_height(alt_type, image_size, rng)
                palette_idx = geometry_idx % len(PALETTES)

            normals = _compute_normals(height, z_scale=1.6)
            render_cfg = PALETTES[palette_idx].copy()
            albedo = _terrain_albedo(height, normals, None, rng)
            rgb_morning, shadow_morning = _render_rgb(height, normals, morning_az, 18.0, render_cfg, rng, albedo=albedo)
            rgb_evening, shadow_evening = _render_rgb(height, normals, evening_az, 18.0, render_cfg, rng, albedo=albedo)

            _save_png(sample_dir / "rgb_morning.png", rgb_morning)
            _save_png(sample_dir / "rgb_evening.png", rgb_evening)
            np.save(sample_dir / "height.npy", height.astype(np.float32))
            _save_png(sample_dir / "height.png", height)
            _save_png(sample_dir / "shadow_morning.png", shadow_morning)
            _save_png(sample_dir / "shadow_evening.png", shadow_evening)

            metadata = {
                **hill_meta,
                "sample_id": sample_id,
                "terrain_id": f"hill_{geometry_idx:04d}",
                "variant_kind": variant_kind,
                "material_palette_id": render_cfg["id"],
                "sun_azimuth": sun_az,
                "sun_elevation": sun_el,
                "sun_azimuth_deg": morning_az if bool(diag.get("generate_pairs", True)) else sun_az,
                "sun_elevation_deg": 18.0 if bool(diag.get("generate_pairs", True)) else sun_el,
                "sun_azimuth_alt_deg": evening_az,
                "sun_elevation_alt_deg": 18.0,
                "camera_azimuth_deg": 0.0,
                "camera_pitch_deg": 0.0,
                "camera_roll_deg": 0.0,
                "camera_altitude_m": 1000.0,
                "camera_fov_deg": 45.0,
                "pair_info": {"morning": "rgb_morning.png", "evening": "rgb_evening.png", "generated": bool(diag.get("generate_pairs", True))},
            }
            with (sample_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

            rows.append(
                {
                    "sample_id": sample_id,
                    "split": "train" if len(rows) < int(round(num_geometries * variants_per_geometry * 0.85)) else "val",
                    "sample_dir": f"samples/{sample_id}",
                    "image_relpath": "rgb_morning.png",
                    "image_alt_relpath": "rgb_evening.png" if bool(diag.get("generate_pairs", True)) else "",
                    "height_relpath": "height.npy",
                    "shadow_relpath": "shadow_morning.png",
                    "shadow_alt_relpath": "shadow_evening.png" if bool(diag.get("generate_pairs", True)) else "",
                    "meta_relpath": "metadata.json",
                }
            )

    fieldnames = [
        "sample_id",
        "split",
        "sample_dir",
        "image_relpath",
        "image_alt_relpath",
        "height_relpath",
        "shadow_relpath",
        "shadow_alt_relpath",
        "meta_relpath",
    ]
    for filename, subset in (
        ("manifest.csv", rows),
        ("train.csv", [row for row in rows if row["split"] == "train"]),
        ("val.csv", [row for row in rows if row["split"] == "val"]),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subset)

    print(f"Diagnostic hill dataset generated at: {output_dir}")


if __name__ == "__main__":
    main()
