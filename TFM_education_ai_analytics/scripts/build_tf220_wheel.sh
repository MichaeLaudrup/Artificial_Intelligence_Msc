#!/usr/bin/env bash
set -euo pipefail

echo "==> Building TensorFlow r2.20 wheel (GPU)"
rm -rf /tmp/tensorflow_src
git clone --depth 1 --branch r2.20 https://github.com/tensorflow/tensorflow.git /tmp/tensorflow_src
cd /tmp/tensorflow_src

export TF_NEED_CUDA=1
export TF_CUDA_VERSION=12.5
export TF_CUDNN_VERSION=9.3
export TF_CUDA_COMPUTE_CAPABILITIES=12.0

PYTHON_BIN="$(command -v python3.13 || command -v python3)"
PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! ${PYTHON_BIN} -c 'import sys; raise SystemExit(0 if (sys.version_info.major == 3 and 9 <= sys.version_info.minor <= 13) else 1)'; then
  echo "ERROR: Python ${PYTHON_VERSION} no es compatible con TensorFlow 2.20 (se requiere 3.9-3.13)." >&2
  exit 1
fi

export PYTHON_BIN_PATH="${PYTHON_BIN}"
export USE_DEFAULT_PYTHON_LIB_PATH=1

CLANG_PATH="/usr/bin/clang-18"
if [[ ! -x "${CLANG_PATH}" ]]; then
  echo "ERROR: clang-18 no encontrado en ${CLANG_PATH}." >&2
  exit 1
fi

export TF_CUDA_CLANG=1
export TF_NEED_CLANG=1
export CLANG_CUDA_COMPILER_PATH="${CLANG_PATH}"
export CC="${CLANG_PATH}"
export BAZEL_COMPILER="${CLANG_PATH}"

echo "==> Python: ${PYTHON_VERSION}"
echo "==> Python bin: ${PYTHON_BIN}"
echo "==> Clang: ${CLANG_PATH}"

# Con pipefail activo, `yes` termina con SIGPIPE cuando ./configure cierra stdin.
# Eso no es un error real de configure, así que desactivamos pipefail solo aquí.
set +o pipefail
yes '' | ./configure
set -o pipefail

echo "==> Bazel version"
bazel --version

bazel build //tensorflow/tools/pip_package:wheel \
  --repo_env=USE_PYWRAP_RULES=1 \
  --repo_env=WHEEL_NAME=tensorflow \
  --config=cuda \
  --config=cuda_wheel \
  --config=opt

cp bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow-*.whl /workspace/TFM_education_ai_analytics/
ls -lh /workspace/TFM_education_ai_analytics/tensorflow-*.whl

echo "==> Done: wheel copied to project root"
