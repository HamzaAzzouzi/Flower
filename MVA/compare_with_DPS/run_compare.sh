#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash MVA/compare_with_DPS/run_compare.sh [gpu_id] [num_images]
# Example:
#   bash MVA/compare_with_DPS/run_compare.sh 0 10

GPU_ID="${1:-0}"
NUM_IMAGES="${2:-10}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPARE_DIR="$ROOT_DIR/MVA/compare_with_DPS"
DPS_DIR="$ROOT_DIR/diffusion-posterior-sampling"
SUBSET_DIR="$COMPARE_DIR/data/afhq_subset"
DPS_TASK_CFG="$COMPARE_DIR/configs/dps_gaussian_deblur_afhq_subset.yaml"
DPS_SAVE_DIR="$COMPARE_DIR/results/dps"
SUMMARY_JSON="$COMPARE_DIR/results/summary.json"

FLOWER_FINAL_PSNR="$ROOT_DIR/results/afhq_cat/ot/gaussian_deblurring_FFT/flower/test/final_psnr.txt"
FLOWER_FINAL_SSIM="$ROOT_DIR/results/afhq_cat/ot/gaussian_deblurring_FFT/flower/test/final_ssim.txt"
FLOWER_FINAL_LPIPS="$ROOT_DIR/results/afhq_cat/ot/gaussian_deblurring_FFT/flower/test/final_lpips.txt"

FLOWER_DATA_DIR="$ROOT_DIR/data/afhq_cat/test"
FLOWER_CAT_DIR="$ROOT_DIR/data/afhq_cat/test/cat"
FLOWER_VAL_CAT_DIR="$ROOT_DIR/data/afhq_cat/val/cat"
FLOWER_TRAIN_CAT_DIR="$ROOT_DIR/data/afhq_cat/train/cat"
SPLIT_FILE="$ROOT_DIR/data_splits/400_filenames.txt"
FLOWER_MODEL_DIR="$ROOT_DIR/model/afhq_cat/ot"
DPS_MODEL_PATH="$DPS_DIR/models/ffhq_10m.pt"
DEMO_IMG_DIR="$ROOT_DIR/flower_demo"

bootstrap_mini_testset() {
  mkdir -p "$FLOWER_CAT_DIR"
  mkdir -p "$FLOWER_VAL_CAT_DIR"
  mkdir -p "$FLOWER_TRAIN_CAT_DIR"

  ensure_train_val_from_test() {
    local existing
    mapfile -t existing < <(find "$FLOWER_CAT_DIR" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | sort | head -n "$NUM_IMAGES")
    if [[ "${#existing[@]}" -eq 0 ]]; then
      return
    fi
    local src
    for src in "${existing[@]}"; do
      local base
      base="$(basename "$src")"
      cp -f "$src" "$FLOWER_VAL_CAT_DIR/$base"
      cp -f "$src" "$FLOWER_TRAIN_CAT_DIR/$base"
    done
  }

  local current_count
  current_count=$(find "$FLOWER_CAT_DIR" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | wc -l)
  if [[ "$current_count" -ge "$NUM_IMAGES" ]]; then
    echo "Found $current_count local AFHQ-like test images, no bootstrap needed."
    ensure_train_val_from_test
    return
  fi

  echo "Local AFHQ test images are missing/insufficient."
  echo "Bootstrapping a tiny test set from flower_demo images (no full dataset download)."

  if [[ ! -f "$SPLIT_FILE" ]]; then
    echo "Missing split file: $SPLIT_FILE"
    exit 1
  fi

  if [[ ! -d "$DEMO_IMG_DIR" ]]; then
    echo "Missing demo image dir: $DEMO_IMG_DIR"
    echo "Cannot build a mini test set automatically."
    exit 1
  fi

  mapfile -t target_names < <(head -n "$NUM_IMAGES" "$SPLIT_FILE")
  mapfile -t source_images < <(find "$DEMO_IMG_DIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | sort)

  if [[ "${#target_names[@]}" -eq 0 ]]; then
    echo "No target names found in $SPLIT_FILE"
    exit 1
  fi
  if [[ "${#source_images[@]}" -eq 0 ]]; then
    echo "No source images found in $DEMO_IMG_DIR"
    exit 1
  fi

  local idx=0
  for name in "${target_names[@]}"; do
    src="${source_images[$((idx % ${#source_images[@]}))]}"
    cp -f "$src" "$FLOWER_CAT_DIR/$name"
    cp -f "$src" "$FLOWER_VAL_CAT_DIR/$name"
    cp -f "$src" "$FLOWER_TRAIN_CAT_DIR/$name"
    idx=$((idx + 1))
  done

  echo "Mini test set prepared in: $FLOWER_CAT_DIR"
  ensure_train_val_from_test
}

echo "[1/7] Checking prerequisites..."
if [[ ! -f "$FLOWER_MODEL_DIR/model_final.pt" ]]; then
  echo "Missing Flower model: $FLOWER_MODEL_DIR/model_final.pt"
  echo "Run: bash download_models.sh"
  exit 1
fi

bootstrap_mini_testset

if [[ ! -f "$DPS_MODEL_PATH" ]]; then
  echo "Missing DPS checkpoint: $DPS_MODEL_PATH"
  echo "Place ffhq_10m.pt there before running comparison."
  exit 1
fi

echo "[2/7] Building AFHQ subset with $NUM_IMAGES images..."
rm -rf "$SUBSET_DIR"
mkdir -p "$SUBSET_DIR"

mapfile -t images < <(find "$FLOWER_DATA_DIR" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | sort | head -n "$NUM_IMAGES")
if [[ "${#images[@]}" -eq 0 ]]; then
  echo "No images found in $FLOWER_DATA_DIR"
  exit 1
fi

idx=0
for src in "${images[@]}"; do
  printf -v name "%05d.png" "$idx"
  cp -f "$src" "$SUBSET_DIR/$name"
  idx=$((idx + 1))
done

echo "[3/7] Writing DPS task config for AFHQ subset..."
cat > "$DPS_TASK_CFG" <<EOF
conditioning:
  method: ps
  params:
    scale: 0.3

data:
  name: ffhq
  root: $SUBSET_DIR

measurement:
  operator:
    name: gaussian_blur
    kernel_size: 61
    intensity: 3.0

  noise:
    name: gaussian
    sigma: 0.05
EOF

echo "[4/7] Running Flower (gaussian_deblurring_FFT, method=flower)..."
cd "$ROOT_DIR"
python main.py --opts \
  dataset afhq_cat \
  eval_split test \
  model ot \
  problem gaussian_deblurring_FFT \
  method flower \
  num_samples 1 \
  max_batch "$NUM_IMAGES" \
  batch_size_ip 1 \
  steps 100 \
  device cuda:"$GPU_ID"

echo "[5/7] Running DPS (gaussian_blur)..."
cd "$DPS_DIR"
PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" python sample_condition.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config "$DPS_TASK_CFG" \
  --gpu "$GPU_ID" \
  --save_dir "$DPS_SAVE_DIR"

echo "[6/7] Evaluating DPS reconstructions..."
cd "$ROOT_DIR"
python MVA/compare_with_DPS/score_dps.py \
  --label_dir "$DPS_SAVE_DIR/gaussian_blur/label" \
  --recon_dir "$DPS_SAVE_DIR/gaussian_blur/recon" \
  --output_json "$DPS_SAVE_DIR/dps_metrics.json"

echo "[7/7] Aggregating Flower + DPS summary..."
python MVA/compare_with_DPS/summarize_results.py \
  --flower_psnr "$FLOWER_FINAL_PSNR" \
  --flower_ssim "$FLOWER_FINAL_SSIM" \
  --flower_lpips "$FLOWER_FINAL_LPIPS" \
  --dps_metrics "$DPS_SAVE_DIR/dps_metrics.json" \
  --output_json "$SUMMARY_JSON"

echo "Done. Summary written to: $SUMMARY_JSON"
