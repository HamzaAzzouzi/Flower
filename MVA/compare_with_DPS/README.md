# Flower vs DPS comparison

This folder contains a reproducible script to compare Flower and DPS on the same AFHQ image subset for Gaussian deblurring.

## What this runs

- Flower:
  - dataset: `afhq_cat`
  - method: `flower`
  - problem: `gaussian_deblurring_FFT`
  - model: `ot`
- DPS:
  - task: Gaussian blur (`kernel_size=61`, `intensity=3.0`, `sigma=0.05`)
  - input images: AFHQ subset generated from `data/afhq_cat/test`

## Prerequisites

From project root:

1. Flower models:

```bash
bash download_models.sh
```

2. DPS checkpoint in:

`diffusion-posterior-sampling/models/ffhq_10m.pt`

3. Optional (better benchmark quality): AFHQ test images in `data/afhq_cat/test`.

If AFHQ test images are not available, `run_compare.sh` automatically bootstraps a tiny test set (few images only) from `flower_demo/` and runs the comparison on that subset.

## Run

From project root:

```bash
bash MVA/compare_with_DPS/run_compare.sh 0 10
```

Arguments:
- first: GPU id
- second: number of images to compare

## Outputs

- DPS reconstructions and metrics:
  - `MVA/compare_with_DPS/results/dps/gaussian_blur/...`
  - `MVA/compare_with_DPS/results/dps/dps_metrics.json`
- Combined summary:
  - `MVA/compare_with_DPS/results/summary.json`
