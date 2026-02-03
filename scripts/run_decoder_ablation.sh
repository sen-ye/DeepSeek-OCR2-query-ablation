#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Defaults (override via CLI args)
# Put model weights to WEIGHTS_DIR
WEIGHTS_DIR="${WEIGHTS_DIR:-./ckpt/DeepSeek-OCR-2}"
IMAGES_DIR="${IMAGES_DIR:-${ROOT_DIR}/test_images}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/decoder_ablation}"
DEVICE="${DEVICE:-cuda:0}"
MODEL_IMAGE_SIZE="${MODEL_IMAGE_SIZE:-1024}"
PREPROCESS_SIZE="${PREPROCESS_SIZE:-512}"
MAX_IMAGES="${MAX_IMAGES:-0}"          # 0 means all images
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SEED="${SEED:-0}"

# Default prompt for decoder ablation (user requested)
PROMPT="${PROMPT:-$'<image>\nFree OCR. '}"

python "${ROOT_DIR}/decoder_token_ablation.py" \
  --weights_dir "${WEIGHTS_DIR}" \
  --images_dir "${IMAGES_DIR}" \
  --out_dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --model_image_size "${MODEL_IMAGE_SIZE}" \
  --preprocess_size "${PREPROCESS_SIZE}" \
  --max_images "${MAX_IMAGES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --seed "${SEED}" \
  --prompt "${PROMPT}" \
  "$@"

