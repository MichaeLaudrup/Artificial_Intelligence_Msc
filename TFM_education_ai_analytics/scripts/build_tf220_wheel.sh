#!/usr/bin/env bash
set -euo pipefail

TF_REF="${TF_REF:-85ce1cf218901e11307cbef90cb46fe6234882f3}"
TF_SRC_DIR="${TF_SRC_DIR:-/tmp/tensorflow_src}"
TF_BASE_REF="${TF_BASE_REF:-master}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.12 || command -v python3)}"
if [[ -z "${CLANG_PATH:-}" ]]; then
  for candidate in /usr/bin/clang-23 /usr/bin/clang-22 /usr/bin/clang-21 /usr/bin/clang-20; do
    if [[ -x "${candidate}" ]]; then
      CLANG_PATH="${candidate}"
      break
    fi
  done
fi
CLANG_PATH="${CLANG_PATH:-/usr/bin/clang-20}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
HERMETIC_CUDA_VERSION="${HERMETIC_CUDA_VERSION:-13.0.0}"
HERMETIC_CUDNN_VERSION="${HERMETIC_CUDNN_VERSION:-9.15.1}"
TF_CUDA_COMPUTE_CAPABILITIES="${TF_CUDA_COMPUTE_CAPABILITIES:-12.0}"
LOCAL_CPU_RESOURCES="${LOCAL_CPU_RESOURCES:-8}"
LOCAL_JOBS="${LOCAL_JOBS:-8}"

find_built_wheel() {
  local wheel_path
  wheel_path="$(find \
    "${TF_SRC_DIR}" \
    "${HOME}/.cache/bazel" \
    -type f \
    -name 'tensorflow-*.whl' \
    2>/dev/null | sort | tail -n 1 || true)"

  if [[ -z "${wheel_path}" ]]; then
    echo "ERROR: no encontré ninguna wheel de TensorFlow construida" >&2
    return 1
  fi

  printf '%s\n' "${wheel_path}"
}

normalize_tf_configure_bazelrc() {
  local config_file="${TF_SRC_DIR}/.tf_configure.bazelrc"
  if [[ ! -f "${config_file}" ]]; then
    echo "ERROR: no existe ${config_file}" >&2
    return 1
  fi

  sed -i -E \
    -e "s|^build:cuda --repo_env HERMETIC_CUDA_VERSION=.*$|build:cuda --repo_env HERMETIC_CUDA_VERSION=\"${HERMETIC_CUDA_VERSION}\"|" \
    -e "s|^build:cuda --repo_env HERMETIC_CUDNN_VERSION=.*$|build:cuda --repo_env HERMETIC_CUDNN_VERSION=\"${HERMETIC_CUDNN_VERSION}\"|" \
    -e "s|^build:cuda --repo_env HERMETIC_CUDA_COMPUTE_CAPABILITIES=.*$|build:cuda --repo_env HERMETIC_CUDA_COMPUTE_CAPABILITIES=\"${TF_CUDA_COMPUTE_CAPABILITIES}\"|" \
    "${config_file}"

  if ! grep -q '^build:cuda --repo_env HERMETIC_CUDA_VERSION=' "${config_file}"; then
    echo "build:cuda --repo_env HERMETIC_CUDA_VERSION=\"${HERMETIC_CUDA_VERSION}\"" >> "${config_file}"
  fi
  if ! grep -q '^build:cuda --repo_env HERMETIC_CUDNN_VERSION=' "${config_file}"; then
    echo "build:cuda --repo_env HERMETIC_CUDNN_VERSION=\"${HERMETIC_CUDNN_VERSION}\"" >> "${config_file}"
  fi
  if ! grep -q '^build:cuda --repo_env HERMETIC_CUDA_COMPUTE_CAPABILITIES=' "${config_file}"; then
    echo "build:cuda --repo_env HERMETIC_CUDA_COMPUTE_CAPABILITIES=\"${TF_CUDA_COMPUTE_CAPABILITIES}\"" >> "${config_file}"
  fi
}

ensure_cuda_nvvm_layout() {
  local nvvm_repo
  local nvcc_repo
  nvvm_repo="$(find "${HOME}/.cache/bazel" -path '*/external/cuda_nvvm' -type d 2>/dev/null | head -n 1 || true)"
  nvcc_repo="$(find "${HOME}/.cache/bazel" -path '*/external/cuda_nvcc' -type d 2>/dev/null | head -n 1 || true)"

  if [[ -z "${nvvm_repo}" ]]; then
    echo "==> No se encontró external/cuda_nvvm todavía" >&2
    return 1
  fi
  if [[ -e "${nvvm_repo}/nvvm/bin/cicc" && -e "${nvvm_repo}/nvvm/libdevice/libdevice.10.bc" ]]; then
    echo "==> cuda_nvvm ya contiene cicc y libdevice"
    return 0
  fi

  if [[ -n "${nvcc_repo}" && -d "${nvcc_repo}/nvvm" ]]; then
    ln -sfn "${nvcc_repo}/nvvm" "${nvvm_repo}/nvvm"
  elif [[ -d "${CUDA_HOME}/nvvm" ]]; then
    ln -sfn "${CUDA_HOME}/nvvm" "${nvvm_repo}/nvvm"
  else
    echo "ERROR: no encontré un directorio nvvm utilizable ni en Bazel ni en ${CUDA_HOME}" >&2
    return 1
  fi

  if [[ ! -e "${nvvm_repo}/nvvm/bin/cicc" || ! -e "${nvvm_repo}/nvvm/libdevice/libdevice.10.bc" ]]; then
    echo "ERROR: cuda_nvvm sigue sin exponer cicc/libdevice tras el parche" >&2
    return 1
  fi

  echo "==> Reparado layout de cuda_nvvm en ${nvvm_repo}"
}

patch_cuda_redist_versions() {
  local redist_file
  local clang_major
  clang_major="$(basename "${CLANG_PATH}" | sed -E 's/[^0-9]*([0-9]+).*/\1/')"
  if [[ -z "${clang_major}" ]]; then
    echo "ERROR: no pude inferir la versión major de clang desde ${CLANG_PATH}" >&2
    return 1
  fi
  redist_file="$(find "${HOME}/.cache/bazel" -path '*/external/rules_ml_toolchain/gpu/cuda/cuda_redist_versions.bzl' 2>/dev/null | head -n 1 || true)"
  if [[ -z "${redist_file}" ]]; then
    echo "==> No se encontró cuda_redist_versions.bzl todavía; se reintentará tras el primer build." >&2
    return 1
  fi
    if grep -q "\"${clang_major}\": \"8.7\"" "${redist_file}"; then
    echo "==> PTX mapping para clang ${clang_major} ya presente en ${redist_file}"
    return 0
  fi
  python - <<PY
from pathlib import Path
path = Path(${redist_file@Q})
text = path.read_text()
clang_major = ${clang_major}
marker = '        "22": "8.7",\n'
if f'        "{clang_major}": "8.7",\n' in text:
    print(f"Mapping clang {clang_major} ya presente en {path}")
    raise SystemExit(0)
if marker not in text:
    raise SystemExit(f"No se encontró el bloque clang esperado en {path}")
text = text.replace(marker, marker + f'        "{clang_major}": "8.7",\n', 1)
path.write_text(text)
print(f"Patched {path}")
PY
}

echo "==> Building TensorFlow custom wheel for Blackwell"
echo "==> TF ref: ${TF_REF}"
rm -rf "${TF_SRC_DIR}"
git clone --depth 1 --filter=blob:none --branch "${TF_BASE_REF}" https://github.com/tensorflow/tensorflow.git "${TF_SRC_DIR}"
cd "${TF_SRC_DIR}"
git fetch --depth 1 --filter=blob:none origin "${TF_REF}"
git checkout --detach FETCH_HEAD

PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! ${PYTHON_BIN} -c 'import sys; raise SystemExit(0 if (sys.version_info.major == 3 and 9 <= sys.version_info.minor <= 13) else 1)'; then
  echo "ERROR: Python ${PYTHON_VERSION} no es compatible con TensorFlow 2.22." >&2
  exit 1
fi
if [[ ! -x "${CLANG_PATH}" ]]; then
  echo "ERROR: clang no encontrado en ${CLANG_PATH}." >&2
  exit 1
fi
if [[ ! -d "${CUDA_HOME}" ]]; then
  echo "ERROR: CUDA_HOME no existe: ${CUDA_HOME}" >&2
  exit 1
fi

export PYTHON_BIN_PATH="${PYTHON_BIN}"
export USE_DEFAULT_PYTHON_LIB_PATH=1
export TF_NEED_CUDA=1
export TF_NEED_CLANG=1
export TF_CUDA_CLANG=1
export TF_DOWNLOAD_CLANG=0
export CC="${CLANG_PATH}"
export BAZEL_COMPILER="${CLANG_PATH}"
export CLANG_CUDA_COMPILER_PATH="${CLANG_PATH}"
export TF_CUDA_COMPUTE_CAPABILITIES
export HERMETIC_CUDA_VERSION
export HERMETIC_CUDNN_VERSION
export HERMETIC_PYTHON_VERSION="${PYTHON_VERSION}"
export CUDA_TOOLKIT_PATH="${CUDA_HOME}"
export CUDNN_INSTALL_PATH="${CUDA_HOME}"
export PATH="$(dirname "${CLANG_PATH}"):${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

cat > .tf_blackwell.bazelrc <<EOF
build --repo_env=USE_HERMETIC_CC_TOOLCHAIN=0
build --action_env=TF_CUDA_CLANG=1
build --action_env=CLANG_CUDA_COMPILER_PATH=${CLANG_PATH}
build --action_env=PATH=$(dirname "${CLANG_PATH}"):${CUDA_HOME}/bin:/usr/local/bin:/usr/bin:/bin
build --action_env=LD_LIBRARY_PATH=${CUDA_HOME}/lib64
build --config=cuda
build --config=cuda_wheel
build --action_env=TF_CUDA_COMPUTE_CAPABILITIES=${TF_CUDA_COMPUTE_CAPABILITIES}
build --repo_env=HERMETIC_PYTHON_VERSION=${PYTHON_VERSION}
EOF

echo "==> Python: ${PYTHON_VERSION}"
echo "==> Python bin: ${PYTHON_BIN}"
echo "==> Clang: ${CLANG_PATH}"
echo "==> CUDA_HOME: ${CUDA_HOME}"
echo "==> Hermetic CUDA/cuDNN: ${HERMETIC_CUDA_VERSION} / ${HERMETIC_CUDNN_VERSION}"

set +o pipefail
yes '' | ./configure
set -o pipefail
normalize_tf_configure_bazelrc

echo "==> Bazel version"
bazel --version

echo "==> Primer build para poblar repositorios de Bazel"
if ! bazel --bazelrc=.tf_blackwell.bazelrc build \
  --local_cpu_resources="${LOCAL_CPU_RESOURCES}" \
  --jobs="${LOCAL_JOBS}" \
  --repo_env=USE_PYWRAP_RULES=1 \
  --repo_env=WHEEL_NAME=tensorflow \
  //tensorflow/tools/pip_package:wheel; then
  echo "==> Aplicando parches de compatibilidad CUDA/PTX"
  patch_cuda_redist_versions || true
  ensure_cuda_nvvm_layout
fi

echo "==> Build final"
bazel --bazelrc=.tf_blackwell.bazelrc build \
  --local_cpu_resources="${LOCAL_CPU_RESOURCES}" \
  --jobs="${LOCAL_JOBS}" \
  --repo_env=USE_PYWRAP_RULES=1 \
  --repo_env=WHEEL_NAME=tensorflow \
  --copt=-Wno-gnu-offsetof-extensions \
  --copt=-Wno-error \
  --copt=-Wno-c23-extensions \
  --copt=-Wno-macro-redefined \
  --verbose_failures \
  //tensorflow/tools/pip_package:wheel

WHEEL_PATH="$(find_built_wheel)"
cp "${WHEEL_PATH}" /workspace/TFM_education_ai_analytics/
ls -lh /workspace/TFM_education_ai_analytics/tensorflow-*.whl

echo "==> Done: wheel copied to project root"
