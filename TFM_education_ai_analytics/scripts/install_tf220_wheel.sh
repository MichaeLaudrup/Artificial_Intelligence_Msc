#!/usr/bin/env bash
set -euo pipefail

cd /workspace/TFM_education_ai_analytics

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.12 || command -v python3 || command -v python)}"
PIP_CMD=("${PYTHON_BIN}" -m pip)

find_built_wheel() {
	local wheel_path
	wheel_path="$(find \
		/workspace/TFM_education_ai_analytics \
		/tmp/tensorflow_src \
		/root/.cache/bazel \
		-type f \
		-name 'tensorflow-*.whl' \
		2>/dev/null | sort | tail -n 1 || true)"

	if [[ -z "${wheel_path}" ]]; then
		echo "ERROR: no encontré ninguna wheel de TensorFlow para instalar" >&2
		return 1
	fi

	printf '%s\n' "${wheel_path}"
}

WHEEL_PATH="$(find_built_wheel)"

"${PIP_CMD[@]}" uninstall -y tensorflow tf-nightly tf_nightly keras-nightly tf-keras-nightly tf_keras-nightly || true
"${PIP_CMD[@]}" install --force-reinstall 'numpy<2' 'packaging<=24.2' 'gast<=0.6.0' 'six==1.16.0' 'markdown-it-py<4'
"${PIP_CMD[@]}" install --force-reinstall --no-deps "${WHEEL_PATH}"
"${PIP_CMD[@]}" install --upgrade --force-reinstall 'protobuf>=6.31.1,<8.0.0' 'grpcio<2.0' 'h5py>=3.11,<3.15' 'ml_dtypes>=0.5.1,<1.0.0' 'tb-nightly~=2.20.0a' 'keras-nightly>=3.12.0.dev'
"${PIP_CMD[@]}" install --force-reinstall 'numpy<2' 'scipy<1.13' 'scikit-learn<1.5' 'pandas<2.3'

env -u TF_USE_LEGACY_KERAS "${PYTHON_BIN}" -c 'import json, tensorflow as tf; print(tf.__version__); print(json.dumps(tf.sysconfig.get_build_info(), indent=2)); print(tf.config.list_physical_devices("GPU"))'
