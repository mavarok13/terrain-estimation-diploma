from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.io import ensure_dir, resolve_repo_path


def _normalize(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    amin = float(array.min())
    amax = float(array.max())
    if amax <= amin:
        return np.zeros_like(array, dtype=np.float32)
    return (array - amin) / (amax - amin)


def _sample_fractal_noise(size: int, octave_count: int, grid_min: int, grid_max: int, rng: np.random.Generator) -> np.ndarray:
    total = np.zeros((size, size), dtype=np.float32)
    amplitude = 1.0
    amplitude_sum = 0.0

    for octave in range(octave_count):
        grid = int(rng.integers(grid_min, grid_max + 1))
        grid = max(4, grid * (2 ** octave))
        small = rng.normal(0.0, 1.0, size=(grid, grid)).astype(np.float32)
        up = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
        total += amplitude * up
        amplitude_sum += amplitude
        amplitude *= 0.5

    return total / max(amplitude_sum, 1e-6)


def _add_gaussian_bumps(height: np.ndarray, count: int, rng: np.random.Generator, sign: float) -> np.ndarray:
    size = height.shape[0]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    out = height.copy()
    for _ in range(count):
        cx = float(rng.uniform(0.15, 0.85) * size)
        cy = float(rng.uniform(0.15, 0.85) * size)
        sigma_x = float(rng.uniform(0.06, 0.2) * size)
        sigma_y = float(rng.uniform(0.06, 0.2) * size)
        amp = float(rng.uniform(0.2, 1.0)) * sign
        blob = np.exp(-(((xx - cx) ** 2) / (2.0 * sigma_x ** 2) + ((yy - cy) ** 2) / (2.0 * sigma_y ** 2)))
        out += amp * blob.astype(np.float32)
    return out


def _terrain_height(size: int, terrain_cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    base = _sample_fractal_noise(
        size=size,
        octave_count=int(terrain_cfg["octave_count"]),
        grid_min=int(terrain_cfg["base_grid_min"]),
        grid_max=int(terrain_cfg["base_grid_max"]),
        rng=rng,
    )
    ridged = 1.0 - np.abs(_sample_fractal_noise(size, 3, 5, 18, rng))
    height = 0.65 * base + float(terrain_cfg["ridge_weight"]) * ridged

    peak_count = int(rng.integers(int(terrain_cfg["peak_count_min"]), int(terrain_cfg["peak_count_max"]) + 1))
    valley_count = int(rng.integers(int(terrain_cfg["valley_count_min"]), int(terrain_cfg["valley_count_max"]) + 1))
    height = _add_gaussian_bumps(height, peak_count, rng, sign=1.0)
    height = _add_gaussian_bumps(height, valley_count, rng, sign=-0.8)

    blur_sigma = float(rng.uniform(float(terrain_cfg["blur_sigma_min"]), float(terrain_cfg["blur_sigma_max"])))
    height = cv2.GaussianBlur(height, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    yy, xx = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)].astype(np.float32)
    radial = np.sqrt(xx ** 2 + yy ** 2)
    edge_mask = np.clip(1.15 - radial, 0.0, 1.0)
    height *= edge_mask
    height = _normalize(height)

    elevation_scale_m = float(
        rng.uniform(float(terrain_cfg["elevation_scale_m_min"]), float(terrain_cfg["elevation_scale_m_max"]))
    )
    return height, {
        "peak_count": peak_count,
        "valley_count": valley_count,
        "blur_sigma": blur_sigma,
        "elevation_scale_m": elevation_scale_m,
    }


def _compute_normals(height: np.ndarray, z_scale: float) -> np.ndarray:
    gy, gx = np.gradient(height * z_scale)
    nx = -gx
    ny = -gy
    nz = np.ones_like(height, dtype=np.float32)
    normals = np.stack([nx, ny, nz], axis=-1).astype(np.float32)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= np.clip(norms, 1e-6, None)
    return normals


def _sun_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array(
        [
            np.sin(az) * np.cos(el),
            -np.cos(az) * np.cos(el),
            np.sin(el),
        ],
        dtype=np.float32,
    )


def _shift_image(image: np.ndarray, dx: float, dy: float, fill_value: float) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(fill_value),
    )


def _shadow_mask(height: np.ndarray, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    sun = _sun_vector(azimuth_deg, elevation_deg)
    light_xy = sun[:2]
    norm_xy = float(np.linalg.norm(light_xy))
    if norm_xy < 1e-6:
        return np.zeros_like(height, dtype=np.float32)

    light_xy = light_xy / norm_xy
    tan_el = float(np.tan(np.deg2rad(max(elevation_deg, 1.0))))
    shadow = np.zeros_like(height, dtype=bool)
    current = height.astype(np.float32)

    max_steps = max(8, height.shape[0] // 8)
    for step in range(1, max_steps + 1):
        dx = -light_xy[0] * step
        dy = -light_xy[1] * step
        shifted = _shift_image(current, dx=dx, dy=dy, fill_value=-1.0)
        threshold = current + tan_el * (step / height.shape[0])
        shadow |= shifted > threshold
    return shadow.astype(np.float32)


def _render_rgb(height: np.ndarray, normals: np.ndarray, azimuth_deg: float, elevation_deg: float, render_cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    sun = _sun_vector(azimuth_deg, elevation_deg)
    diffuse = np.clip(np.sum(normals * sun[None, None, :], axis=-1), 0.0, 1.0)
    shadow = _shadow_mask(height, azimuth_deg, elevation_deg)

    slope = 1.0 - np.clip(normals[..., 2], 0.0, 1.0)
    color_noise = _sample_fractal_noise(height.shape[0], 3, 10, 24, rng)
    color_noise = _normalize(color_noise)

    green = np.array([0.24, 0.42, 0.18], dtype=np.float32)
    brown = np.array([0.48, 0.40, 0.24], dtype=np.float32)
    rock = np.array([0.52, 0.50, 0.48], dtype=np.float32)
    snow = np.array([0.86, 0.88, 0.90], dtype=np.float32)

    lowland_mix = height[..., None]
    slope_mix = slope[..., None]
    albedo = (1.0 - lowland_mix) * green + lowland_mix * brown
    albedo = (1.0 - slope_mix) * albedo + slope_mix * rock
    snow_mask = np.clip((height - 0.78) / 0.18, 0.0, 1.0)[..., None]
    albedo = (1.0 - snow_mask) * albedo + snow_mask * snow

    tint = 0.85 + 0.3 * color_noise[..., None]
    albedo = np.clip(albedo * tint, 0.0, 1.0)

    ambient = float(render_cfg["ambient"])
    diffuse_weight = float(render_cfg["diffuse"])
    shadow_strength = float(render_cfg["shadow_strength"])
    fog_strength = float(render_cfg["fog_strength"])

    light_term = ambient + diffuse_weight * diffuse
    light_term *= 1.0 - shadow_strength * shadow
    rgb = albedo * light_term[..., None]

    fog = fog_strength * (1.0 - diffuse)[..., None]
    sky_tint = np.array([0.62, 0.72, 0.86], dtype=np.float32)
    rgb = rgb * (1.0 - fog) + sky_tint * fog
    return np.clip(rgb, 0.0, 1.0), shadow


def _save_png(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    arr = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)


def _metadata_for_sample(sample_id: str, terrain_meta: dict, capture_meta: dict) -> dict:
    meta = {
        "sample_id": sample_id,
        "terrain_id": sample_id,
        "lighting_setup": "primary_alt_pair",
    }
    meta.update(terrain_meta)
    meta.update(capture_meta)
    return meta


def generate_dataset(config: dict) -> None:
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)

    dataset_cfg = config["dataset"]
    terrain_cfg = config["terrain"]
    render_cfg = config["render"]
    camera_cfg = config["camera"]

    repo_root = Path(__file__).resolve().parents[2]
    output_root = resolve_repo_path(dataset_cfg["output_root"], repo_root)
    dataset_root = output_root / dataset_cfg["name"]
    samples_root = dataset_root / "samples"
    ensure_dir(samples_root)

    manifest_rows: list[dict[str, str]] = []
    num_samples = int(dataset_cfg["num_samples"])
    image_size = int(dataset_cfg["image_size"])
    train_ratio = float(dataset_cfg["train_ratio"])
    write_alt_image = bool(dataset_cfg["write_alt_image"])

    for index in tqdm(range(num_samples), desc="Generating terrain dataset"):
        sample_id = f"sample_{index:05d}"
        sample_dir = samples_root / sample_id
        ensure_dir(sample_dir)

        height, terrain_meta = _terrain_height(image_size, terrain_cfg, rng)
        normals = _compute_normals(height, z_scale=terrain_meta["elevation_scale_m"] / 1000.0)

        sun_azimuth = float(rng.uniform(0.0, 360.0))
        sun_elevation = float(rng.uniform(float(render_cfg["sun_elevation_min_deg"]), float(render_cfg["sun_elevation_max_deg"])))
        alt_delta = float(
            rng.uniform(
                float(render_cfg["alt_sun_azimuth_delta_deg_min"]),
                float(render_cfg["alt_sun_azimuth_delta_deg_max"]),
            )
        )
        sun_azimuth_alt = (sun_azimuth + alt_delta) % 360.0
        sun_elevation_alt = float(rng.uniform(float(render_cfg["sun_elevation_min_deg"]), float(render_cfg["sun_elevation_max_deg"])))

        rgb, shadow = _render_rgb(height, normals, sun_azimuth, sun_elevation, render_cfg, rng)
        rgb_alt = None
        if write_alt_image:
            rgb_alt, _ = _render_rgb(height, normals, sun_azimuth_alt, sun_elevation_alt, render_cfg, rng)

        camera_meta = {
            "sun_azimuth_deg": sun_azimuth,
            "sun_elevation_deg": sun_elevation,
            "sun_azimuth_alt_deg": sun_azimuth_alt,
            "sun_elevation_alt_deg": sun_elevation_alt,
            "camera_azimuth_deg": float(rng.uniform(float(camera_cfg["azimuth_deg_min"]), float(camera_cfg["azimuth_deg_max"]))),
            "camera_pitch_deg": float(rng.uniform(float(camera_cfg["pitch_deg_min"]), float(camera_cfg["pitch_deg_max"]))),
            "camera_roll_deg": float(rng.uniform(float(camera_cfg["roll_deg_min"]), float(camera_cfg["roll_deg_max"]))),
            "camera_altitude_m": float(rng.uniform(float(camera_cfg["altitude_m_min"]), float(camera_cfg["altitude_m_max"]))),
            "camera_fov_deg": float(rng.uniform(float(camera_cfg["fov_deg_min"]), float(camera_cfg["fov_deg_max"]))),
            "timestamp": f"synthetic_{index:05d}",
            "image_size": image_size,
        }

        meta = _metadata_for_sample(sample_id, terrain_meta, camera_meta)

        _save_png(sample_dir / "rgb.png", rgb)
        if rgb_alt is not None:
            _save_png(sample_dir / "rgb_alt.png", rgb_alt)
        np.save(sample_dir / "height.npy", height.astype(np.float32))
        _save_png(sample_dir / "height_vis.png", height)
        np.save(sample_dir / "normal.npy", normals.astype(np.float32))
        _save_png(sample_dir / "shadow_mask.png", shadow)
        with (sample_dir / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

        split = "train" if index < int(round(num_samples * train_ratio)) else "val"
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "sample_dir": f"samples/{sample_id}",
                "image_relpath": "rgb.png",
                "image_alt_relpath": "rgb_alt.png" if write_alt_image else "",
                "height_relpath": "height.npy",
                "shadow_relpath": "shadow_mask.png",
                "meta_relpath": "meta.json",
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
        "meta_relpath",
    ]

    for filename, rows in (
        ("manifest.csv", manifest_rows),
        ("train.csv", [row for row in manifest_rows if row["split"] == "train"]),
        ("val.csv", [row for row in manifest_rows if row["split"] == "val"]),
    ):
        with (dataset_root / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Dataset generated at: {dataset_root}")
