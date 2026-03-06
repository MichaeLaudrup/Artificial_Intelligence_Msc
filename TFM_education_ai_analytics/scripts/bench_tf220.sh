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

echo "==> Strict GPU compatibility check"
TFM_GPU_STRICT=1 EXECUTION_DEVICE=gpu python - <<'PY'
import tensorflow as tf
from loguru import logger
from educational_ai_analytics.tf_runtime import configure_tensorflow_runtime
print(configure_tensorflow_runtime(tf, 'gpu', logger))
PY
