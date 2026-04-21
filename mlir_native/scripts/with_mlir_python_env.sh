#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LLVM_BUILD_PY_DEFAULT="/home/jay/projs/llvm-project/build-py"
LLVM_BUILD_PY="${LLVM_BUILD_PY:-${LLVM_BUILD_PY_DEFAULT}}"
MLIR_PYTHON_ROOT="${MLIR_PYTHON_ROOT:-${LLVM_BUILD_PY}/tools/mlir/python_packages/mlir_core}"

export PYTHONPATH="${MLIR_PYTHON_ROOT}:${PYTHONPATH:-}"

if [[ $# -eq 0 ]]; then
  echo "MLIR Python environment configured."
  echo "  REPO_ROOT=${REPO_ROOT}"
  echo "  MLIR_PYTHON_ROOT=${MLIR_PYTHON_ROOT}"
  echo
  echo "Usage:"
  echo "  source ${BASH_SOURCE[0]}"
  echo "  ${BASH_SOURCE[0]} <command> [args...]"
  exit 0
fi

exec "$@"
