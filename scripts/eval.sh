#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_JSON="${DATASET_JSON:-/mnt/data/zruan/workspace_novel/zruan/GOAL/datasets/DCI_test_corrected_original.json}"
CKPT_PATH="${CKPT_PATH:-$ROOT_DIR/weights/DCI.pth}"
MODEL_ALIAS="${MODEL_ALIAS:-B}"
BATCH_SIZE="${BATCH_SIZE:-32}"

python -m structxlip.retrieval \
  --dataset "$DATASET_JSON" \
  --ckpt "$CKPT_PATH" \
  --model "$MODEL_ALIAS" \
  --eval_batch_size "$BATCH_SIZE"
