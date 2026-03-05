#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE="${1:-1024}"
PRETRAIN_EPOCHS="${2:-3}"

cd /workspace/TFM_education_ai_analytics

echo "==> CPU benchmark"
EXECUTION_DEVICE=cpu python -m educational_ai_analytics.2_modeling.ae.train_autoencoder \
  --batch-size "$BATCH_SIZE" \
  --pretrain-epochs "$PRETRAIN_EPOCHS" \
  --joint-epochs 0

echo "==> GPU benchmark"
EXECUTION_DEVICE=gpu python -m educational_ai_analytics.2_modeling.ae.train_autoencoder \
  --batch-size "$BATCH_SIZE" \
  --use-mixed-precision \
  --pretrain-epochs "$PRETRAIN_EPOCHS" \
  --joint-epochs 0
