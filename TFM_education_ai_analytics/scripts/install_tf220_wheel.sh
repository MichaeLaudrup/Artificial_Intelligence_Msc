#!/usr/bin/env bash
set -euo pipefail

cd /workspace/TFM_education_ai_analytics

pip uninstall -y tensorflow || true
pip install /workspace/TFM_education_ai_analytics/tensorflow-*.whl

python -c 'import json, tensorflow as tf; print(tf.__version__); print(json.dumps(tf.sysconfig.get_build_info(), indent=2)); print(tf.config.list_physical_devices("GPU"))'
