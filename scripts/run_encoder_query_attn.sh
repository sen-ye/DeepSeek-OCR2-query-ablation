#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Defaults (override via CLI args)
# Put model weights to WEIGHTS_DIR
WEIGHTS_DIR="${WEIGHTS_DIR:-./ckpt/DeepSeek-OCR-2}"
IMAGES_DIR="${IMAGES_DIR:-${ROOT_DIR}/test_images}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/encoder_query_attn}"
DEVICE="${DEVICE:-cuda:0}"
MODEL_IMAGE_SIZE="${MODEL_IMAGE_SIZE:-1024}"
DISPLAY_SIZE="${DISPLAY_SIZE:-512}"
LAST_N_LAYERS="${LAST_N_LAYERS:-2}"
LAYERS_MODE="${LAYERS_MODE:-all}" # last|all

python "${ROOT_DIR}/encoder_query_attn_ablation.py" \
  --weights_dir "${WEIGHTS_DIR}" \
  --images_dir "${IMAGES_DIR}" \
  --out_dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --model_image_size "${MODEL_IMAGE_SIZE}" \
  --display_size "${DISPLAY_SIZE}" \
  --last_n_layers "${LAST_N_LAYERS}" \
  --layers "${LAYERS_MODE}" \
  "$@"

