#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-14B}"
PORT="${PORT:-30000}"
HOST="0.0.0.0"
TP_SIZE="${TP_SIZE:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.83}"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"

export XDG_CACHE_HOME=/data1/zx57/.cache
export TORCH_EXTENSIONS_DIR=/data1/zx57/.cache/torch_extensions
export TRITON_CACHE_DIR=/data1/zx57/.cache/triton
export FLASHINFER_CACHE_DIR=/data1/zx57/.cache/flashinfer

mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$FLASHINFER_CACHE_DIR"

echo "[Compiler]"
echo "CC=$CC"
echo "CXX=$CXX"
$CC --version | head -1
$CXX --version | head -1

echo "[SGLang] model=${MODEL_PATH} host=${HOST} port=${PORT} tp_size=${TP_SIZE}"

ss -ltnp | grep ":${PORT}" || true

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}"