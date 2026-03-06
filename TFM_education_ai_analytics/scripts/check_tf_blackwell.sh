#!/usr/bin/env bash
set -euo pipefail

cd /workspace/TFM_education_ai_analytics

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.12 || command -v python3 || command -v python)}"

TFM_GPU_STRICT=1 EXECUTION_DEVICE=gpu "${PYTHON_BIN}" - <<'PY'
import json

import tensorflow as tf
from loguru import logger

from educational_ai_analytics.tf_runtime import configure_tensorflow_runtime

print("tf_version", tf.__version__)
print("build_info", json.dumps(tf.sysconfig.get_build_info(), indent=2))
print("physical_gpus", tf.config.list_physical_devices("GPU"))

runtime_device = configure_tensorflow_runtime(tf, "gpu", logger)
print("runtime_device", runtime_device)

a = tf.random.normal([512, 512])
b = tf.random.normal([512, 512])
c = tf.matmul(a, b)
print("matmul_ok", c.shape, float(tf.reduce_mean(c)))
PY