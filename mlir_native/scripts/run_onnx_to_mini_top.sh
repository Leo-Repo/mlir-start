#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LLAMA_PYTHON_DEFAULT="/home/jay/miniconda3/envs/llama/bin/python"
LLAMA_PYTHON="${LLAMA_PYTHON:-${LLAMA_PYTHON_DEFAULT}}"

exec /usr/bin/env bash "${SCRIPT_DIR}/with_mlir_python_env.sh" \
  "${LLAMA_PYTHON}" \
  "${REPO_ROOT}/mlir_native/python/onnx_to_mini_top.py" \
  "$@"
