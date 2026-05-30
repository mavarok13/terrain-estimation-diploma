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


def _terrain_albedo(
    height: np.ndarray,
    normals: np.ndarray,
    generation_cfg: dict | None,
    rng: np.random.Generator,
) -> np.ndarray:
    slope = 1.0 - np.clip(normals[..., 2], 0.0, 1.0)
    color_noise = _sample_fractal_noise(height.shape[0], 3, 10, 24, rng)
    color_noise = _normalize(color_noise)

    generation_cfg = generation_cfg or {}
    green = np.array([0.24, 0.42, 0.18], dtype=np.float32)
    brown = np.array([0.48, 0.40, 0.24], dtype=np.float32)
    rock = np.array([0.52, 0.50, 0.48], dtype=np.float32)
    snow = np.array([0.86, 0.88, 0.90], dtype=np.float32)
    if bool(generation_cfg.get("shuffle_palettes", False)):
        palette = [green, brown, rock]
        rng.shuffle(palette)
        green, brown, rock = palette

    if bool(generation_cfg.get("randomize_albedo_independent_of_height", False)):
        lowland_mix = _normalize(_sample_fractal_noise(height.shape[0], 4, 4, 18, rng))[..., None]
    else:
        lowland_mix = height[..., None]
    slope_mix = slope[..., None]
    albedo = (1.0 - lowland_mix) * green + lowland_mix * brown
    albedo = (1.0 - slope_mix) * albedo + slope_mix * rock
    if not bool(generation_cfg.get("disable_height_based_snow", False)):
        snow_mask = np.clip((height - 0.78) / 0.18, 0.0, 1.0)[..., None]
        albedo = (1.0 - snow_mask) * albedo + snow_mask * snow

    noise_strength = float(generation_cfg.get("albedo_noise_strength", 0.3))
    tint = 1.0 - 0.5 * noise_strength + noise_strength * color_noise[..., None]
    return np.clip(albedo * tint, 0.0, 1.0)


def _render_rgb(
    height: np.ndarray,
    normals: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    render_cfg: dict,
    rng: np.random.Generator,
    generation_cfg: dict | None = None,
    albedo: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sun = _sun_vector(azimuth_deg, elevation_deg)
    diffuse = np.clip(np.sum(normals * sun[None, None, :], axis=-1), 0.0, 1.0)
    shadow = _shadow_mask(height, azimuth_deg, elevation_deg)
    if albedo is None:
        albedo = _terrain_albedo(height, normals, generation_cfg, rng)

    ambient = float(render_cfg["ambient"])
    diffuse_weight = float(render_cfg["diffuse"])
    shadow_strength = float(render_cfg["shadow_strength"])
    fog_strength = float(render_cfg["fog_strength"])
    exposure = float(render_cfg.get("exposure", 1.0))
    gamma = float(render_cfg.get("gamma", 1.0))

    light_term = ambient + diffuse_weight * diffuse
    light_term *= 1.0 - shadow_strength * shadow
    rgb = albedo * light_term[..., None]

    fog = fog_strength * (1.0 - diffuse)[..., None]
    sky_tint = np.array([0.62, 0.72, 0.86], dtype=np.float32)
    rgb = rgb * (1.0 - fog) + sky_tint * fog
    rgb = np.clip(rgb * exposure, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-6:
        rgb = np.power(rgb, gamma)
    return np.clip(rgb, 0.0, 1.0), shadow


def _save_png(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    arr = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)


def _rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    return np.clip(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2], 0.0, 1.0).astype(np.float32)


def _metadata_for_sample(sample_id: str, terrain_meta: dict, capture_meta: dict) -> dict:
    meta = {
        "sample_id": sample_id,
        "terrain_id": sample_id,
        "lighting_setup": "primary_alt_pair",
    }
    meta.update(terrain_meta)
    meta.update(capture_meta)
    return meta


def _shadow_albedo_options(shadow_cfg: dict) -> tuple[str, dict]:
    albedo_cfg = shadow_cfg.get("albedo", {})
    if albedo_cfg is None:
        albedo_cfg = {}
    if not isinstance(albedo_cfg, dict):
        raise ValueError("shadow_geometry.albedo must be a mapping when provided")

    mode = str(albedo_cfg.get("mode", "")).lower().strip()
    if not mode:
        mode = "random" if bool(shadow_cfg.get("random_albedo", False)) else "grayscale"
    if mode not in {"random", "grayscale", "height_based"}:
        raise ValueError(f"Unsupported shadow_geometry albedo mode: {mode}")

    merged = dict(shadow_cfg)
    merged.update(albedo_cfg)
    return mode, merged


def _height_color_palette(palette_cfg: object) -> np.ndarray:
    default = np.array(
        [
            [0.16, 0.30, 0.14],
            [0.36, 0.52, 0.20],
            [0.56, 0.47, 0.27],
            [0.72, 0.69, 0.60],
        ],
        dtype=np.float32,
    )
    if palette_cfg is None or str(palette_cfg).lower() in {"default", "natural"}:
        return default
    if isinstance(palette_cfg, dict):
        colors = [palette_cfg.get(key) for key in ("low", "mid", "high", "very_high") if palette_cfg.get(key) is not None]
    else:
        colors = palette_cfg
    palette = np.asarray(colors, dtype=np.float32)
    if palette.ndim != 2 or palette.shape[0] < 2 or palette.shape[1] != 3:
        raise ValueError("height_color_palette must contain at least two RGB colors")
    if float(palette.max()) > 1.0:
        palette = palette / 255.0
    return np.clip(palette, 0.0, 1.0).astype(np.float32)


def _height_based_shadow_geometry_albedo(
    height: np.ndarray,
    normals: np.ndarray,
    rng: np.random.Generator,
    albedo_cfg: dict,
) -> np.ndarray:
    size = height.shape[0]
    palette = _height_color_palette(albedo_cfg.get("height_color_palette", "natural"))
    height_color_strength = float(np.clip(float(albedo_cfg.get("height_color_strength", 0.45)), 0.0, 1.0))
    color_noise_strength = float(max(0.0, albedo_cfg.get("color_noise_strength", 0.08)))
    texture_strength = float(max(0.0, albedo_cfg.get("texture_strength", 0.12)))

    h = _normalize(height)
    scaled = h * (palette.shape[0] - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, palette.shape[0] - 1)
    lower = np.clip(lower, 0, palette.shape[0] - 1)
    mix = (scaled - lower)[..., None]
    height_color = (1.0 - mix) * palette[lower] + mix * palette[upper]

    base_luma = 0.62 + 0.12 * (_normalize(_sample_fractal_noise(size, 3, 5, 18, rng)) - 0.5)
    albedo = (1.0 - height_color_strength) * base_luma[..., None] + height_color_strength * height_color

    slope = 1.0 - np.clip(normals[..., 2], 0.0, 1.0)
    slope_rock = np.array([0.50, 0.47, 0.40], dtype=np.float32)
    slope_mix = np.clip(slope * 1.4, 0.0, 0.35)[..., None]
    albedo = (1.0 - slope_mix) * albedo + slope_mix * slope_rock

    if texture_strength > 0.0:
        coarse = _normalize(_sample_fractal_noise(size, 4, 4, 20, rng))
        fine = _normalize(_sample_fractal_noise(size, 3, 16, 40, rng))
        texture = 0.65 * coarse + 0.35 * fine
        texture = 1.0 + texture_strength * (texture[..., None] - 0.5)
        albedo *= texture

    if color_noise_strength > 0.0:
        color_noise = rng.normal(0.0, color_noise_strength, size=(size, size, 3)).astype(np.float32)
        color_noise = cv2.GaussianBlur(color_noise, (0, 0), sigmaX=max(1.0, size / 96.0), sigmaY=max(1.0, size / 96.0))
        albedo += color_noise

    return np.clip(albedo, 0.05, 1.0).astype(np.float32)


def _shadow_geometry_albedo(
    height: np.ndarray,
    normals: np.ndarray,
    rng: np.random.Generator,
    shadow_cfg: dict,
) -> tuple[np.ndarray, str]:
    size = height.shape[0]
    mode, albedo_cfg = _shadow_albedo_options(shadow_cfg)
    if mode == "grayscale":
        return np.full((size, size, 3), 0.72, dtype=np.float32), mode
    if mode == "height_based":
        return _height_based_shadow_geometry_albedo(height, normals, rng, albedo_cfg), mode

    brightness_min = float(albedo_cfg.get("albedo_brightness_min", 0.55))
    brightness_max = float(albedo_cfg.get("albedo_brightness_max", 1.35))
    contrast_min = float(albedo_cfg.get("albedo_contrast_min", 0.55))
    contrast_max = float(albedo_cfg.get("albedo_contrast_max", 1.55))
    pattern_strength = float(albedo_cfg.get("albedo_pattern_strength", 0.55))
    color_noise_min = float(albedo_cfg.get("albedo_color_noise_strength_min", 0.03))
    color_noise_max = float(albedo_cfg.get("albedo_color_noise_strength_max", 0.12))
    if "albedo_noise_strength_min" in albedo_cfg or "albedo_noise_strength_max" in albedo_cfg:
        noise_min = float(albedo_cfg.get("albedo_noise_strength_min", albedo_cfg.get("albedo_noise_strength", 0.04)))
        noise_max = float(albedo_cfg.get("albedo_noise_strength_max", albedo_cfg.get("albedo_noise_strength", 0.04)))
        noise_strength = float(rng.uniform(noise_min, noise_max))
    else:
        noise_strength = float(albedo_cfg.get("albedo_noise_strength", 0.04))

    natural_palettes = [
        np.array([[0.25, 0.42, 0.18], [0.42, 0.52, 0.25], [0.46, 0.37, 0.22], [0.55, 0.54, 0.48]], dtype=np.float32),
        np.array([[0.34, 0.47, 0.22], [0.58, 0.52, 0.30], [0.50, 0.36, 0.20], [0.62, 0.60, 0.52]], dtype=np.float32),
        np.array([[0.20, 0.34, 0.20], [0.30, 0.45, 0.28], [0.38, 0.32, 0.24], [0.50, 0.50, 0.45]], dtype=np.float32),
        np.array([[0.48, 0.42, 0.26], [0.62, 0.54, 0.34], [0.43, 0.30, 0.18], [0.58, 0.55, 0.47]], dtype=np.float32),
    ]
    palette = natural_palettes[int(rng.integers(0, len(natural_palettes)))].copy()
    palette *= rng.uniform(0.90, 1.12, size=(1, 3)).astype(np.float32)
    brightness = float(rng.uniform(brightness_min, brightness_max))
    contrast = float(rng.uniform(contrast_min, contrast_max))
    coarse_grid = int(rng.integers(3, 10))
    weights = rng.uniform(0.0, 1.0, size=(coarse_grid, coarse_grid, palette.shape[0])).astype(np.float32)
    weights = cv2.resize(weights, (size, size), interpolation=cv2.INTER_CUBIC)
    weights = cv2.GaussianBlur(weights, (0, 0), sigmaX=max(3.0, size / 18.0), sigmaY=max(3.0, size / 18.0))
    weights = np.clip(weights, 0.0, None)
    weights = weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-6)

    albedo = np.tensordot(weights, palette, axes=([-1], [0])).astype(np.float32)
    pattern = rng.uniform(0.0, 1.0, size=(coarse_grid, coarse_grid, 1)).astype(np.float32)
    pattern = cv2.resize(pattern, (size, size), interpolation=cv2.INTER_CUBIC)
    if pattern.ndim == 2:
        pattern = pattern[..., None]
    pattern = cv2.GaussianBlur(pattern, (0, 0), sigmaX=max(3.0, size / 18.0), sigmaY=max(3.0, size / 18.0))
    if pattern.ndim == 2:
        pattern = pattern[..., None]
    pattern = _normalize(pattern)

    albedo = albedo * (1.0 - pattern_strength + pattern_strength * (0.45 + 1.1 * pattern))
    albedo = (albedo - 0.5) * contrast + 0.5

    color_noise_strength = float(rng.uniform(color_noise_min, color_noise_max))
    color_noise = rng.normal(0.0, color_noise_strength, size=(size, size, 3)).astype(np.float32)
    color_noise = cv2.GaussianBlur(color_noise, (0, 0), sigmaX=max(1.0, size / 96.0), sigmaY=max(1.0, size / 96.0))
    albedo = albedo + color_noise
    albedo *= brightness

    if bool(albedo_cfg.get("albedo_noise_texture", True)) and noise_strength > 0.0:
        fine_noise = rng.normal(0.0, noise_strength, size=(size, size, 3)).astype(np.float32)
        albedo = albedo + fine_noise

    return np.clip(albedo, 0.05, 1.0).astype(np.float32), mode


def _shadow_geometry_height(
    size: int,
    cfg: dict,
    terrain_cfg: dict,
    rng: np.random.Generator,
    shadow_suns: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, dict]:
    difficulty = int(cfg.get("difficulty", 1))
    if difficulty >= 3:
        height, meta = _terrain_height(size, terrain_cfg, rng)
        meta.update({"dataset_mode": "shadow_geometry", "difficulty": difficulty, "curriculum_stage": "procedural"})
        return height, meta

    max_hills = 1 if difficulty == 1 else 3
    min_hills = 1 if difficulty == 1 else 2
    hill_count = int(rng.integers(min_hills, max_hills + 1))
    shapes = list(cfg.get("hill_shapes", ["circular", "elongated", "ridge", "plateau"]))
    yy, xx = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)].astype(np.float32)
    height = np.zeros((size, size), dtype=np.float32)
    hills: list[dict] = []
    individual_shadow_masks: list[list[np.ndarray]] = []
    min_sep_factor = 2.6 if difficulty == 1 else 1.55
    max_shadow_overlap = float(cfg.get("max_shadow_overlap", 0.01 if difficulty == 1 else 0.08))

    for hill_idx in range(hill_count):
        accepted_profile = None
        accepted_hill = None
        accepted_shadows = None
        for _ in range(120):
            radius = float(rng.uniform(float(cfg.get("radius_min", 0.16)), float(cfg.get("radius_max", 0.30))))
            cx = float(rng.uniform(-0.52, 0.52))
            cy = float(rng.uniform(-0.52, 0.52))
            if not all(np.hypot(cx - h["center"][0], cy - h["center"][1]) > min_sep_factor * (radius + h["radius"]) for h in hills):
                continue

            shape = str(shapes[(hill_idx + int(rng.integers(0, len(shapes)))) % len(shapes)])
            orientation = float(rng.uniform(0.0, np.pi))
            hill_height = float(rng.uniform(float(cfg.get("height_min", 0.45)), float(cfg.get("height_max", 1.0))))
            dx = xx - cx
            dy = yy - cy
            xr = np.cos(orientation) * dx + np.sin(orientation) * dy
            yr = -np.sin(orientation) * dx + np.cos(orientation) * dy

            if shape == "circular":
                profile = np.exp(-((dx * dx + dy * dy) / (2.0 * radius * radius)))
            elif shape == "elongated":
                profile = np.exp(-((xr * xr) / (2.0 * (radius * 1.45) ** 2) + (yr * yr) / (2.0 * (radius * 0.55) ** 2)))
            elif shape == "ridge":
                profile = np.exp(-((xr * xr) / (2.0 * (radius * 2.0) ** 2) + (yr * yr) / (2.0 * (radius * 0.30) ** 2)))
            elif shape == "plateau":
                rho = np.sqrt((xr / max(radius, 1e-6)) ** 2 + (yr / max(radius * 0.85, 1e-6)) ** 2)
                profile = 1.0 / (1.0 + np.exp((rho - 0.72) / 0.08))
            else:
                raise ValueError(f"Unsupported shadow_geometry hill shape: {shape}")

            candidate = (hill_height * profile).astype(np.float32)
            candidate_shadows = []
            for azimuth, elevation in shadow_suns or []:
                candidate_shadows.append(_shadow_mask(candidate, azimuth, elevation) > 0.5)
            overlaps = []
            for existing_per_sun in individual_shadow_masks:
                for sun_idx, candidate_shadow in enumerate(candidate_shadows):
                    existing_shadow = existing_per_sun[sun_idx]
                    denom = max(float(candidate_shadow.sum()), 1.0)
                    overlaps.append(float(np.logical_and(candidate_shadow, existing_shadow).sum()) / denom)
            if overlaps and max(overlaps) > max_shadow_overlap:
                continue

            accepted_profile = candidate
            accepted_hill = {"shape": shape, "center": [cx, cy], "radius": radius, "height": hill_height, "orientation_rad": orientation}
            accepted_shadows = candidate_shadows
            break

        if accepted_profile is None or accepted_hill is None or accepted_shadows is None:
            continue
        height = np.maximum(height, accepted_profile)
        hills.append(accepted_hill)
        individual_shadow_masks.append(accepted_shadows)

    height = cv2.GaussianBlur(height, (0, 0), sigmaX=float(cfg.get("height_blur_sigma", 1.2)), sigmaY=float(cfg.get("height_blur_sigma", 1.2)))
    height = np.clip(height, 0.0, 1.0).astype(np.float32)
    elevation_scale_m = float(rng.uniform(float(cfg.get("elevation_scale_m_min", 120.0)), float(cfg.get("elevation_scale_m_max", 700.0))))
    return height, {
        "dataset_mode": "shadow_geometry",
        "difficulty": difficulty,
        "curriculum_stage": "controlled_hills",
        "hill_count": len(hills),
        "hills": hills,
        "elevation_scale_m": elevation_scale_m,
    }


def _write_manifest_files(dataset_root: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    for filename, subset in (
        ("manifest.csv", rows),
        ("train.csv", [row for row in rows if row["split"] == "train"]),
        ("val.csv", [row for row in rows if row["split"] == "val"]),
    ):
        with (dataset_root / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subset)


def _generate_shadow_geometry_dataset(config: dict, rng: np.random.Generator, dataset_root: Path, samples_root: Path) -> None:
    dataset_cfg = config["dataset"]
    terrain_cfg = config["terrain"]
    render_cfg = dict(config["render"])
    camera_cfg = config["camera"]
    shadow_cfg = dict(config.get("shadow_geometry", {}))
    if "albedo" not in shadow_cfg and "albedo" in config:
        shadow_cfg["albedo"] = config["albedo"]

    manifest_rows: list[dict[str, str]] = []
    num_samples = int(dataset_cfg["num_samples"])
    image_size = int(dataset_cfg["image_size"])
    train_ratio = float(dataset_cfg["train_ratio"])
    include_noon = bool(shadow_cfg.get("include_noon", False))
    albedo_mode, _ = _shadow_albedo_options(shadow_cfg)

    render_cfg.setdefault("ambient", 0.24)
    render_cfg.setdefault("diffuse", 0.95)
    render_cfg.setdefault("shadow_strength", 0.82)
    render_cfg.setdefault("fog_strength", 0.0)

    for index in tqdm(range(num_samples), desc="Generating shadow geometry dataset"):
        sample_id = f"sample_{index:05d}"
        sample_dir = samples_root / sample_id
        ensure_dir(sample_dir)

        sunrise_az = float((90.0 + rng.uniform(-35.0, 35.0)) % 360.0)
        sunset_az = float((270.0 + rng.uniform(-35.0, 35.0)) % 360.0)
        low_el_min = float(shadow_cfg.get("sun_elevation_min_deg", render_cfg.get("sun_elevation_min_deg", 8.0)))
        low_el_max = float(shadow_cfg.get("sun_elevation_max_deg", render_cfg.get("sun_elevation_max_deg", 28.0)))
        sunrise_el = float(rng.uniform(low_el_min, low_el_max))
        sunset_el = float(rng.uniform(low_el_min, low_el_max))

        height, terrain_meta = _shadow_geometry_height(
            image_size,
            shadow_cfg,
            terrain_cfg,
            rng,
            shadow_suns=[(sunrise_az, sunrise_el), (sunset_az, sunset_el)],
        )
        normals = _compute_normals(height, z_scale=float(terrain_meta["elevation_scale_m"]) / 1000.0)
        albedo, albedo_mode = _shadow_geometry_albedo(height, normals, rng, shadow_cfg)

        rgb, shadow = _render_rgb(height, normals, sunrise_az, sunrise_el, render_cfg, rng, albedo=albedo)
        rgb_alt, shadow_alt = _render_rgb(height, normals, sunset_az, sunset_el, render_cfg, rng, albedo=albedo)
        gray = _rgb_to_luma(rgb)
        gray_alt = _rgb_to_luma(rgb_alt)

        _save_png(sample_dir / "gray.png", gray)
        _save_png(sample_dir / "gray_alt.png", gray_alt)
        _save_png(sample_dir / "rgb.png", rgb)
        _save_png(sample_dir / "rgb_alt.png", rgb_alt)
        _save_png(sample_dir / "shadow_mask.png", shadow)
        _save_png(sample_dir / "shadow_mask_alt.png", shadow_alt)
        if include_noon:
            noon_az = float(rng.uniform(0.0, 360.0))
            noon_el = float(rng.uniform(55.0, 75.0))
            rgb_noon, shadow_noon = _render_rgb(height, normals, noon_az, noon_el, render_cfg, rng, albedo=albedo)
            _save_png(sample_dir / "gray_noon.png", _rgb_to_luma(rgb_noon))
            _save_png(sample_dir / "shadow_mask_noon.png", shadow_noon)
        else:
            noon_az = 0.0
            noon_el = 0.0

        np.save(sample_dir / "height.npy", height.astype(np.float32))
        _save_png(sample_dir / "height_vis.png", height)
        np.save(sample_dir / "normal.npy", normals.astype(np.float32))

        camera_meta = {
            "sun_azimuth_deg": sunrise_az,
            "sun_elevation_deg": sunrise_el,
            "sun_azimuth_alt_deg": sunset_az,
            "sun_elevation_alt_deg": sunset_el,
            "sun_azimuth_noon_deg": noon_az,
            "sun_elevation_noon_deg": noon_el,
            "camera_azimuth_deg": float(rng.uniform(float(camera_cfg["azimuth_deg_min"]), float(camera_cfg["azimuth_deg_max"]))),
            "camera_pitch_deg": float(rng.uniform(float(camera_cfg["pitch_deg_min"]), float(camera_cfg["pitch_deg_max"]))),
            "camera_roll_deg": float(rng.uniform(float(camera_cfg["roll_deg_min"]), float(camera_cfg["roll_deg_max"]))),
            "camera_altitude_m": float(rng.uniform(float(camera_cfg["altitude_m_min"]), float(camera_cfg["altitude_m_max"]))),
            "camera_fov_deg": float(rng.uniform(float(camera_cfg["fov_deg_min"]), float(camera_cfg["fov_deg_max"]))),
            "timestamp": f"shadow_geometry_{index:05d}",
            "image_size": image_size,
            "random_albedo": albedo_mode == "random",
            "albedo_mode": albedo_mode,
            "lighting_setup": "sunrise_sunset_noon" if include_noon else "sunrise_sunset",
        }
        meta = _metadata_for_sample(sample_id, terrain_meta, camera_meta)
        with (sample_dir / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

        split = "train" if index < int(round(num_samples * train_ratio)) else "val"
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "sample_dir": f"samples/{sample_id}",
                "image_relpath": "rgb.png",
                "image_alt_relpath": "rgb_alt.png",
                "height_relpath": "height.npy",
                "shadow_relpath": "shadow_mask.png",
                "shadow_alt_relpath": "shadow_mask_alt.png",
                "meta_relpath": "meta.json",
            }
        )

    fieldnames = ["sample_id", "split", "sample_dir", "image_relpath", "image_alt_relpath", "height_relpath", "shadow_relpath", "shadow_alt_relpath", "meta_relpath"]
    _write_manifest_files(dataset_root, fieldnames, manifest_rows)
    print(f"Shadow geometry dataset generated at: {dataset_root}")


def generate_dataset(config: dict) -> None:
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)

    dataset_cfg = config["dataset"]
    terrain_cfg = config["terrain"]
    render_cfg = config["render"]
    camera_cfg = config["camera"]
    generation_cfg = config.get("generation", {})

    repo_root = Path(__file__).resolve().parents[2]
    output_root = resolve_repo_path(dataset_cfg["output_root"], repo_root)
    dataset_root = output_root / dataset_cfg["name"]
    samples_root = dataset_root / "samples"
    ensure_dir(samples_root)

    if str(dataset_cfg.get("dataset_mode", "")).lower() in {"shadow_geometry", "rgb_shadow_geometry"}:
        _generate_shadow_geometry_dataset(config, rng, dataset_root, samples_root)
        return

    manifest_rows: list[dict[str, str]] = []
    num_samples = int(dataset_cfg["num_samples"])
    image_size = int(dataset_cfg["image_size"])
    train_ratio = float(dataset_cfg["train_ratio"])
    write_alt_image = bool(dataset_cfg["write_alt_image"])

    variants_per_dem = 1
    if bool(generation_cfg.get("same_dem_multiple_albedos", False)):
        variants_per_dem *= 2
    if bool(generation_cfg.get("same_dem_multiple_suns", False)):
        variants_per_dem *= 2
    cached_height = None
    cached_terrain_meta = None

    for index in tqdm(range(num_samples), desc="Generating terrain dataset"):
        sample_id = f"sample_{index:05d}"
        sample_dir = samples_root / sample_id
        ensure_dir(sample_dir)

        if cached_height is None or index % variants_per_dem == 0:
            cached_height, cached_terrain_meta = _terrain_height(image_size, terrain_cfg, rng)
        height = cached_height.copy()
        terrain_meta = dict(cached_terrain_meta)
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

        albedo = _terrain_albedo(height, normals, generation_cfg, rng)
        rgb, shadow = _render_rgb(height, normals, sun_azimuth, sun_elevation, render_cfg, rng, generation_cfg, albedo=albedo)
        rgb_alt = None
        shadow_alt = None
        if write_alt_image:
            rgb_alt, shadow_alt = _render_rgb(
                height,
                normals,
                sun_azimuth_alt,
                sun_elevation_alt,
                render_cfg,
                rng,
                generation_cfg,
                albedo=albedo,
            )

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
        if shadow_alt is not None:
            _save_png(sample_dir / "shadow_mask_alt.png", shadow_alt)
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
                "shadow_alt_relpath": "shadow_mask_alt.png" if write_alt_image else "",
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
        "shadow_alt_relpath",
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
