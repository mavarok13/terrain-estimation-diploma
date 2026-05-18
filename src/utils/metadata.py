from __future__ import annotations

import numpy as np


METADATA_NORMALIZATION = {
    "sun_azimuth_deg": (180.0, 180.0),
    "sun_elevation_deg": (45.0, 45.0),
    "camera_azimuth_deg": (180.0, 180.0),
    "camera_pitch_deg": (15.0, 30.0),
    "camera_roll_deg": (0.0, 10.0),
    "camera_altitude_m": (1000.0, 1000.0),
    "camera_fov_deg": (45.0, 45.0),
}


def encode_metadata_value(key: str, value: float) -> float:
    center, scale = METADATA_NORMALIZATION.get(key, (0.0, 1.0))
    return float((value - center) / max(scale, 1e-6))


def encoded_metadata_size(keys: list[str]) -> int:
    size = 0
    for key in keys:
        size += 2 if key.endswith("_deg") else 1
    return size


def encode_metadata_vector(metadata: dict, keys: list[str]) -> np.ndarray:
    values = []
    for key in keys:
        if key not in metadata:
            raise KeyError(f"Missing metadata field: {key}")
        raw_value = float(metadata[key])
        if key.endswith("_deg"):
            radians = np.deg2rad(raw_value)
            values.append(float(np.sin(radians)))
            values.append(float(np.cos(radians)))
        else:
            values.append(encode_metadata_value(key, raw_value))
    return np.asarray(values, dtype=np.float32)
