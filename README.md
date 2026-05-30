# Terrain Height Estimation

Python MVP for monocular terrain height estimation from synthetic remote-sensing imagery.

This repository implements:

1. A fast procedural dataset generator.
2. A PyTorch training pipeline for dense height regression.
3. An inference script for predicting a height map from one image.

The MVP is intentionally practical:

1. Single-image inference is the main baseline.
2. The data format already supports an optional second illumination image.
3. Metadata-conditioned inference is supported through sun and camera parameters.
4. Shadow masks are generated and used for diagnostics and augmentation, but monocular shadow ambiguity is not considered solved.

## Repository Layout

```text
terrain_height_estimation/
  configs/
    generation/
    inference/
    training/
  data/
    generated/
    raw/
    splits/
  outputs/
  scripts/
    generate_dataset.py
    infer.py
    train.py
  src/
    dataset/
    generation/
    inference/
    models/
    training/
    utils/
  README.md
  requirements.txt
```

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

## 1. Generate A Dataset

Default config:

```bash
python scripts/generate_dataset.py --config configs/generation/mvp.yaml
```

This creates a dataset under `data/generated/mvp_dataset/` with one folder per sample.

Each sample contains:

1. `rgb.png`
2. `rgb_alt.png` if enabled
3. `height.npy`
4. `height_vis.png`
5. `normal.npy`
6. `shadow_mask.png`
7. `meta.json`

The generator also writes:

1. `manifest.csv`
2. `train.csv`
3. `val.csv`

## 2. Train The Baseline

```bash
python scripts/train.py --config configs/training/mvp.yaml
```

Outputs are written to `outputs/mvp_baseline/`:

1. `checkpoints/best.pt`
2. `checkpoints/last.pt`
3. `samples/epoch_xxx/`
4. `history.json`

## 3. Run Inference

```bash
python scripts/infer.py \
  --config configs/inference/default.yaml \
  --checkpoint outputs/mvp_baseline/checkpoints/best.pt \
  --image data/generated/mvp_dataset/samples/sample_00000/rgb.png \
  --metadata data/generated/mvp_dataset/samples/sample_00000/meta.json
```

Inference saves:

1. `pred_height.npy`
2. `pred_height.png`
3. `prediction_overview.png`

## Metadata Conditioning

The baseline can condition on metadata by tiling encoded metadata values into constant feature maps.

Angle fields are encoded as `sin/cos` pairs to avoid the `0 deg` / `360 deg` discontinuity.

Default metadata fields:

1. `sun_azimuth_deg`
2. `sun_elevation_deg`
3. `camera_azimuth_deg`
4. `camera_pitch_deg`
5. `camera_roll_deg`
6. `camera_altitude_m`
7. `camera_fov_deg`

## Shadow Handling In The MVP

The halo issue around shadow boundaries is treated explicitly:

1. Synthetic data includes a terrain-based shadow mask.
2. Sun direction is randomized aggressively.
3. Terrain albedo is randomized independently from shape.
4. Training includes gradient loss.
5. Validation reports MAE inside shadows and outside shadows.
6. Training can perturb shadow darkness to reduce over-reliance on dark pixels.

This reduces the risk of the network learning a simple darkness-to-height shortcut, but it does not remove the fundamental ambiguity of monocular hidden terrain.

## Planned Extensions

1. Better procedural terrain generation with stronger ridge and erosion priors.
2. True oblique rendering and more realistic camera models.
3. Two-image sunrise/sunset fusion.
4. Classification-regression head inspired by TSE-Net.
5. Teacher-student and pseudo-label filtering for semi-supervised learning.
