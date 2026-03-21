#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash MVA/compare_with_DPS/run_compare.sh [gpu_id]
# Example:
#   bash MVA/compare_with_DPS/run_compare.sh 0

GPU_ID="${1:-0}"
NUM_IMAGES=2

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
OOD_IMAGES=("JUUL.png" "HAAMZAA.png")

prepare_ood_testset() {
  mkdir -p "$FLOWER_CAT_DIR"
  mkdir -p "$FLOWER_VAL_CAT_DIR"
  mkdir -p "$FLOWER_TRAIN_CAT_DIR"

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

  if [[ "${#target_names[@]}" -eq 0 ]]; then
    echo "No target names found in $SPLIT_FILE"
    exit 1
  fi
  if [[ "${#OOD_IMAGES[@]}" -ne "$NUM_IMAGES" ]]; then
    echo "Internal configuration error: OOD image list and NUM_IMAGES mismatch"
    exit 1
  fi

  echo "Preparing OOD test set with images: ${OOD_IMAGES[*]}"

  local idx=0
  for name in "${target_names[@]}"; do
    local src="$DEMO_IMG_DIR/${OOD_IMAGES[$idx]}"
    if [[ ! -f "$src" ]]; then
      echo "Missing required OOD image: $src"
      exit 1
    fi

    # Keep Flower split filenames (expected by 400_filenames.txt), but overwrite contents with OOD images.
    SRC_IMG="$src" DST_TEST="$FLOWER_CAT_DIR/$name" DST_VAL="$FLOWER_VAL_CAT_DIR/$name" DST_TRAIN="$FLOWER_TRAIN_CAT_DIR/$name" python - <<'PY'
from PIL import Image
import os

src = os.environ["SRC_IMG"]
dst_test = os.environ["DST_TEST"]
dst_val = os.environ["DST_VAL"]
dst_train = os.environ["DST_TRAIN"]

img = Image.open(src).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
img.save(dst_test)
img.save(dst_val)
img.save(dst_train)
PY

    idx=$((idx + 1))
  done

  echo "OOD test set prepared in: $FLOWER_CAT_DIR"
}

echo "[1/8] Checking prerequisites..."
if [[ ! -f "$FLOWER_MODEL_DIR/model_final.pt" ]]; then
  echo "Missing Flower model: $FLOWER_MODEL_DIR/model_final.pt"
  echo "Run: bash download_models.sh"
  exit 1
fi

prepare_ood_testset

if [[ ! -f "$DPS_MODEL_PATH" ]]; then
  echo "Missing DPS checkpoint: $DPS_MODEL_PATH"
  echo "Place ffhq_10m.pt there before running comparison."
  exit 1
fi

echo "[2/8] Building OOD subset for DPS with $NUM_IMAGES images..."
rm -rf "$SUBSET_DIR"
mkdir -p "$SUBSET_DIR"

for i in 0 1; do
  src="$DEMO_IMG_DIR/${OOD_IMAGES[$i]}"
  printf -v name "%05d.png" "$i"
  SRC_IMG="$src" DST_IMG="$SUBSET_DIR/$name" python - <<'PY'
from PIL import Image
import os

src = os.environ["SRC_IMG"]
dst = os.environ["DST_IMG"]

# DPS FFHQ model expects 256x256 inputs.
img = Image.open(src).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
img.save(dst)
PY
done

echo "[3/8] Writing DPS task config for OOD subset..."
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

echo "[4/8] Running Flower (gaussian_deblurring_FFT, method=flower)..."
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

echo "[5/8] Running DPS (gaussian_blur)..."
rm -rf "$DPS_SAVE_DIR/gaussian_blur"
cd "$DPS_DIR"
PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" python sample_condition.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config "$DPS_TASK_CFG" \
  --gpu "$GPU_ID" \
  --save_dir "$DPS_SAVE_DIR"

echo "[6/8] Evaluating DPS reconstructions..."
cd "$ROOT_DIR"
python MVA/compare_with_DPS/score_dps.py \
  --label_dir "$DPS_SAVE_DIR/gaussian_blur/label" \
  --recon_dir "$DPS_SAVE_DIR/gaussian_blur/recon" \
  --output_json "$DPS_SAVE_DIR/dps_metrics.json"

echo "[7/8] Aggregating Flower + DPS summary..."
python MVA/compare_with_DPS/summarize_results.py \
  --flower_psnr "$FLOWER_FINAL_PSNR" \
  --flower_ssim "$FLOWER_FINAL_SSIM" \
  --flower_lpips "$FLOWER_FINAL_LPIPS" \
  --dps_metrics "$DPS_SAVE_DIR/dps_metrics.json" \
  --output_json "$SUMMARY_JSON"

echo "[8/8] Building paired same-image outputs..."
PAIRED_DIR="$COMPARE_DIR/results/paired"
rm -rf "$PAIRED_DIR"
mkdir -p "$PAIRED_DIR"
for i in 0 1; do
  d=$(printf "%s/%05d" "$PAIRED_DIR" "$i")
  mkdir -p "$d"
  cp -f "$DPS_SAVE_DIR/gaussian_blur/label/$(printf '%05d.png' "$i")" "$d/label.png"
  cp -f "$DPS_SAVE_DIR/gaussian_blur/input/$(printf '%05d.png' "$i")" "$d/dps_input.png"
  cp -f "$DPS_SAVE_DIR/gaussian_blur/recon/$(printf '%05d.png' "$i")" "$d/dps_recon.png"
  cp -f "$ROOT_DIR/results/afhq_cat/ot/gaussian_deblurring_FFT/flower/test/steps=100/num_samples=1/gaussian_deblurring_FFT_flower_batch${i}_final.png" "$d/flower_recon.png"
  cp -f "$ROOT_DIR/results/afhq_cat/ot/gaussian_deblurring_FFT/flower/test/steps=100/num_samples=1/gaussian_deblurring_FFT_noisy_batch${i}_final.png" "$d/flower_noisy.png"
  cp -f "$ROOT_DIR/results/afhq_cat/ot/gaussian_deblurring_FFT/flower/test/steps=100/num_samples=1/gaussian_deblurring_FFT_clean_batch${i}_final.png" "$d/flower_clean.png"
done

echo "Done. Summary written to: $SUMMARY_JSON"
echo "Paired outputs written to: $PAIRED_DIR"
