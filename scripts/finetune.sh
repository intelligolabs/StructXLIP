#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_JSON="${DATASET_JSON:-/mnt/data/zruan/workspace_novel/zruan/GOAL/datasets/DOCCI_train_nested_with_all_sketches.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/ckpts_docci_rsam2canny_edge_top2_ours}"
MODEL_NAME="${MODEL_NAME:-openai/clip-vit-base-patch16}"
WANDB_PROJECT="${WANDB_PROJECT:-INSECT_EDGE_TOP2_DEBUG}"

python -m structxlip.train \
  --dataset "$DATASET_JSON" \
  --model "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project "$WANDB_PROJECT" \
  --epochs 10 \
  --batch_size 16 \
  --init_lr 5e-6 \
  --min_lr 0 \
  --weight_decay 0.05 \
  --lambda_global 1.0 \
  --lambda_structure_centric 0.25 \
  --lambda_rgb_scribble_consistency 0.001 \
  --lambda_local_structure_centric 0.001 \
  --chunk_top_k 2 \
  --remove_colors
