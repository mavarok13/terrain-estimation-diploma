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

Input modes can be selected with `training.input_mode`:

1. `rgb`: `rgb.png`, 3 channels.
2. `grayscale`: `rgb.png` converted to one grayscale channel, 1 channel.
3. `grayscale_shadow_sun`: grayscale image plus primary shadow mask plus normalized sun azimuth/elevation maps, 4 channels.
4. `grayscale_pair_shadow_sun`: primary/alternate grayscale plus primary shadow mask plus both sun directions, 7 channels.
5. `grayscale_pair_shadowmask_sun`: primary/alternate grayscale plus both shadow masks plus both sun directions, 8 channels.
6. `rgb_pair`: morning/evening or primary/alternate RGB concatenated, 6 channels.
7. `grayscale_pair`: morning/evening or primary/alternate grayscale concatenated, 2 channels.
8. `rgb_pair_metadata`: RGB pair plus normalized sun azimuth/elevation maps for both images, 10 channels.
9. `rgb_pair_full_metadata`: RGB pair plus all configured `dataset.metadata_keys`; this is the legacy-compatible metadata baseline.

Experiment commands:

```bash
python scripts/train.py --config configs/train.yaml training.input_mode=rgb training.output_dir=outputs/ablation_rgb
python scripts/train.py --config configs/train.yaml training.input_mode=grayscale training.output_dir=outputs/ablation_grayscale
python scripts/train.py --config configs/train.yaml training.input_mode=grayscale_shadow_sun training.output_dir=outputs/ablation_grayscale_shadow_sun
python scripts/train.py --config configs/train.yaml training.input_mode=grayscale_pair_shadow_sun training.output_dir=outputs/ablation_grayscale_pair_shadow_sun
python scripts/train.py --config configs/train.yaml training.input_mode=grayscale_pair_shadowmask_sun training.output_dir=outputs/ablation_grayscale_pair_shadowmask_sun
python scripts/train.py --config configs/train.yaml training.input_mode=rgb_pair training.output_dir=outputs/ablation_rgb_pair
python scripts/train.py --config configs/train.yaml training.input_mode=grayscale_pair training.output_dir=outputs/ablation_grayscale_pair
python scripts/train.py --config configs/train.yaml training.input_mode=rgb_pair_metadata training.output_dir=outputs/ablation_rgb_pair_metadata
python scripts/train.py --config configs/train.yaml training.input_mode=rgb_pair_full_metadata training.output_dir=outputs/mvp_baseline
```

Use a separate `training.output_dir` for each mode. Otherwise checkpoints and preview images from different experiments will overwrite each other and become hard to compare.

Resume from the last saved checkpoint:

```bash
python scripts/train.py --config configs/training/mvp.yaml --resume outputs/mvp_baseline/checkpoints/last.pt
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

For pair modes, pass `--image-alt`. For `rgb_pair_metadata`, also pass `--metadata` with `sun_azimuth_deg`, `sun_elevation_deg`, `sun_azimuth_alt_deg`, and `sun_elevation_alt_deg`. For shadow-mask modes, pass `--metadata` and `--shadow-mask`; `grayscale_pair_shadowmask_sun` also needs `--shadow-mask-alt`.

## Shadow Geometry Curriculum

Generate controlled shadow-to-height training data:

```bash
python scripts/generate_dataset.py --config configs/generation/shadow_geometry.yaml
```

Train on the strongest anti-shortcut mode:

```bash
python scripts/train.py --config configs/training/shadow_geometry.yaml
```

The `shadow_geometry` dataset mode writes grayscale sunrise/sunset renders, matching shadow masks, one shared `height.npy`, and metadata with both sun directions. Difficulty is controlled by `shadow_geometry.difficulty`:

1. `1`: one smooth hill, flat background, strong shadows, no intentional overlap.
2. `2`: two or three smooth hills with slight overlap allowed.
3. `3`: procedural terrain fallback for later curriculum stages.

Use overrides to advance curriculum stages, for example:

```bash
python scripts/generate_dataset.py --config configs/generation/shadow_geometry.yaml shadow_geometry.difficulty=2 dataset.name=shadow_geometry_d2
python scripts/generate_dataset.py --config configs/generation/shadow_geometry.yaml shadow_geometry.difficulty=3 dataset.name=shadow_geometry_d3
```

## Diagnostic Hill Dataset

Generate controlled single-hill samples:

```bash
python scripts/generate_hill_diagnostic_dataset.py --config configs/hill_diagnostic.yaml
```

The diagnostic dataset writes `data/generated/hill_diagnostic/` by default. Each sample contains:

1. `rgb.png`
2. `rgb_morning.png`
3. `rgb_evening.png`
4. `height.npy`
5. `height.png`
6. `shadow_mask.png`
7. `metadata.json`

The generator covers gaussian hills, cones/frustums, ridges, asymmetric hills, double hills, and crater/valley variants. Variants include same geometry with different colors, same color with different suns, morning/evening pairs, and different geometry with similar color.

Quick sanity training on the diagnostic dataset:

```bash
python scripts/train.py --config configs/train.yaml \
  dataset.root=data/generated/hill_diagnostic \
  dataset.train_manifest=data/generated/hill_diagnostic/train.csv \
  dataset.val_manifest=data/generated/hill_diagnostic/val.csv \
  dataset.image_size=256 \
  training.input_mode=rgb_pair_metadata \
  training.output_dir=outputs/hill_diagnostic_sanity \
  training.epochs=3
```

Anti-color-cheating generator options are available under `generation` in `configs/generation/mvp.yaml`:

1. `randomize_albedo_independent_of_height`
2. `same_dem_multiple_albedos`
3. `same_dem_multiple_suns`
4. `shuffle_palettes`
5. `disable_height_based_snow`
6. `albedo_noise_strength`

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
