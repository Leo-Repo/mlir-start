#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT}/build/cuda"
mkdir -p "${OUT_DIR}"

NVCC="${NVCC:-/usr/local/cuda-12.4/bin/nvcc}"
if [[ ! -x "${NVCC}" ]]; then
  NVCC="$(command -v nvcc)"
fi

"${NVCC}" -std=c++17 -O2 -Xcompiler -fPIC \
  -c "${ROOT}/cuda/conv_silu_kernel.cu" \
  -o "${OUT_DIR}/conv_silu_kernel.o"

echo "Wrote ${OUT_DIR}/conv_silu_kernel.o"
