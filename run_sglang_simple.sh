#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-14B}"
PORT="${PORT:-30000}"

# Do not use variable name HOST.
# Conda/build environments may already define HOST=x86_64-conda-linux-gnu.
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"

TP_SIZE="${TP_SIZE:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.83}"

# Optional GPU selector.
# Example:
#   GPU_DEVICES=0,1,2,3 TP_SIZE=4 ./run_sglang_simple.sh
#   GPU_DEVICES=4,5,6,7 TP_SIZE=4 ./run_sglang_simple.sh
if [ -n "${GPU_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
fi

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"

export XDG_CACHE_HOME=/data1/zx57/.cache
export TORCH_EXTENSIONS_DIR=/data1/zx57/.cache/torch_extensions
export TRITON_CACHE_DIR=/data1/zx57/.cache/triton
export FLASHINFER_CACHE_DIR=/data1/zx57/.cache/flashinfer

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TRITON_CACHE_DIR" \
  "$FLASHINFER_CACHE_DIR"

echo "[Compiler]"
echo "CC=$CC"
echo "CXX=$CXX"
$CC --version | head -1
$CXX --version | head -1

echo "[GPU]"
echo "GPU_DEVICES=${GPU_DEVICES:-<not set>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "TP_SIZE=$TP_SIZE"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  VISIBLE_GPU_COUNT="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')"
  if [ "$TP_SIZE" -gt "$VISIBLE_GPU_COUNT" ]; then
    echo "[Error] TP_SIZE=$TP_SIZE is larger than visible GPU count=$VISIBLE_GPU_COUNT"
    echo "        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    exit 1
  fi
fi

echo "[SGLang]"
echo "model=${MODEL_PATH}"
echo "host=${SERVER_HOST}"
echo "port=${PORT}"
echo "tp_size=${TP_SIZE}"
echo "mem_fraction_static=${MEM_FRACTION_STATIC}"

echo "[Port check]"
ss -ltnp | grep ":${PORT}" || true

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${SERVER_HOST}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}"