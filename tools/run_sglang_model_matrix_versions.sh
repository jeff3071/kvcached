#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

# Build a clean virtualenv per SGLang version and run the manual kvcached
# SGLang model/layout compatibility matrix in each one.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_WORK_DIR="${ROOT_DIR}/.tmp/sglang-model-matrix"
TEST_PATH="tests/test_sglang_model_compatibility.py"

python_bin="${PYTHON:-python3.11}"
work_dir="${KVCACHED_SGLANG_MATRIX_WORK_DIR:-${DEFAULT_WORK_DIR}}"
layouts="${KVCACHED_MODEL_MATRIX_LAYOUTS:-contiguous,non-contiguous}"
clean=1
run_tests=1
versions=()
pytest_args=()

usage() {
  cat <<USAGE
Usage: $0 --versions "0.5.10 0.5.15" [options] [-- pytest-args...]

Options:
  --version VERSION       Add one SGLang version to test. Can be repeated.
  --versions LIST         Space- or comma-separated SGLang versions.
  --layouts LIST          Layouts for the pytest matrix.
                          Default: contiguous,non-contiguous
  --python PYTHON         Python executable for new venvs. Default: python3.11
  --work-dir DIR          Directory for generated venvs/logs.
                          Default: .tmp/sglang-model-matrix
  --keep-env              Reuse an existing venv instead of deleting it first.
  --setup-only            Create/install venvs but do not run pytest.
  -h, --help              Show this help.

Examples:
  $0 --versions "0.5.10 0.5.15"
  $0 --version 0.5.9 --layouts non-contiguous -- -k Qwen -vv

Environment:
  KVCACHED_MODEL_MATRIX_TIMEOUT is passed through to pytest if set.
  HF_HOME, HF_TOKEN, CUDA_VISIBLE_DEVICES, and SGLANG_* variables are preserved.
USAGE
}

add_versions() {
  local raw="$1"
  raw="${raw//,/ }"
  local version
  for version in ${raw}; do
    [[ -n "${version}" ]] && versions+=("${version}")
  done
}

run_pip() {
  local venv_python="$1"
  shift
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${venv_python}" "$@"
  else
    "${venv_python}" -m pip install "$@"
  fi
}

create_venv() {
  local venv_dir="$1"

  if [[ "${clean}" == 1 && -d "${venv_dir}" ]]; then
    case "${venv_dir}" in
      ""|"/"|"${ROOT_DIR}")
        echo "Refusing to remove unsafe venv path: ${venv_dir}" >&2
        exit 2
        ;;
    esac
    rm -rf "${venv_dir}"
  fi

  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    mkdir -p "$(dirname "${venv_dir}")"
    if command -v uv >/dev/null 2>&1; then
      uv venv "${venv_dir}" --python "${python_bin}"
    else
      "${python_bin}" -m venv "${venv_dir}"
    fi
  fi
}

install_version() {
  local version="$1"
  local venv_dir="$2"
  local venv_python="${venv_dir}/bin/python"

  run_pip "${venv_python}" --upgrade pip
  run_pip "${venv_python}" -r "${ROOT_DIR}/requirements.txt"

  # Keep this compatibility pin aligned with engine_integration/scripts/setup.sh.
  if [[ "${version}" == "0.5.3" ]]; then
    run_pip "${venv_python}" transformers==4.57.0
  fi

  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${venv_python}" "sglang[all]==${version}" --prerelease=allow
    uv pip install --python "${venv_python}" -e "${ROOT_DIR}" \
      --no-build-isolation --no-cache-dir
  else
    "${venv_python}" -m pip install --pre "sglang[all]==${version}"
    "${venv_python}" -m pip install -e "${ROOT_DIR}" \
      --no-build-isolation --no-cache-dir
  fi

  "${venv_python}" "${ROOT_DIR}/tools/dev_copy_pth.py"
}

run_matrix() {
  local version="$1"
  local venv_dir="$2"
  local venv_python="${venv_dir}/bin/python"
  local log_dir="${work_dir}/logs"
  local log_path="${log_dir}/sglang-${version}.log"

  mkdir -p "${log_dir}"
  echo "[matrix] SGLang ${version}; layouts=${layouts}; log=${log_path}"
  (
    cd "${ROOT_DIR}"
    env \
      RUN_KVCACHED_MODEL_MATRIX=1 \
      KVCACHED_MODEL_MATRIX_LAYOUTS="${layouts}" \
      "${venv_python}" -m pytest -s "${TEST_PATH}" "${pytest_args[@]}"
  ) 2>&1 | tee "${log_path}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      versions+=("$2")
      shift 2
      ;;
    --versions)
      add_versions "$2"
      shift 2
      ;;
    --layouts)
      layouts="$2"
      shift 2
      ;;
    --python)
      python_bin="$2"
      shift 2
      ;;
    --work-dir)
      work_dir="$2"
      shift 2
      ;;
    --keep-env)
      clean=0
      shift
      ;;
    --setup-only)
      run_tests=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      pytest_args=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#versions[@]} -eq 0 ]]; then
  add_versions "${SGLANG_VERSIONS:-0.5.10 0.5.15}"
fi

mkdir -p "${work_dir}"

for version in "${versions[@]}"; do
  venv_dir="${work_dir}/sglang-${version}"
  echo "[setup] SGLang ${version}; venv=${venv_dir}"
  create_venv "${venv_dir}"
  install_version "${version}" "${venv_dir}"
  if [[ "${run_tests}" == 1 ]]; then
    run_matrix "${version}" "${venv_dir}"
  fi
done
