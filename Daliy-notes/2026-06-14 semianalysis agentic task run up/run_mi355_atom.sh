#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8000}"
BACKGROUND="${2:-false}"
GPU_IDS="${3:-0}"
LOG_FILE="${LOG_FILE:-atom_server_${PORT}.log}"
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3.5-27B-FP8}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
ATTN_PREFILL_CHUNK_SIZE="${ATTN_PREFILL_CHUNK_SIZE:-16384}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
KV_TRANSFER_CONFIG="${KV_TRANSFER_CONFIG:-{\"kv_connector\":\"lmcache_offload\",\"kv_role\":\"offload\"}}"
LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-True}"
LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-312.5}"
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
OFFLOAD_PROFILE="${OFFLOAD_PROFILE:-1}"
OFFLOAD_MIN_LOAD_TOKENS="${OFFLOAD_MIN_LOAD_TOKENS:-8192}"
OFFLOAD_GPU_STAGING_CHUNKS="${OFFLOAD_GPU_STAGING_CHUNKS:-2}"

CMD=(
  python -m atom.entrypoints.openai_server
  --port "$PORT"
  --model "$MODEL_PATH"
  -tp "$TP_SIZE"
  --kv_cache_dtype "$KV_CACHE_DTYPE"
  --trust-remote-code
  --enable_prefix_caching
  --enable_chunked_prefill
  --attn-prefill-chunk-size "$ATTN_PREFILL_CHUNK_SIZE"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --block-size "$BLOCK_SIZE"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
)

export PYTHONPATH=/data/mesh-all/ATOM
export USE_ATOMESH_ENTRYPOINTS=0
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HIP_VISIBLE_DEVICES="$GPU_IDS"
export LMCACHE_LOCAL_CPU
export LMCACHE_MAX_LOCAL_CPU_SIZE
export LMCACHE_CHUNK_SIZE
export OFFLOAD_PROFILE
export OFFLOAD_MIN_LOAD_TOKENS
export OFFLOAD_GPU_STAGING_CHUNKS

echo "Usage: $0 [port] [background] [gpu_ids]"
echo "  background: true/false, yes/no, bg/fg, 1/0"
echo "  gpu_ids: GPU id(s) to expose, comma-separated for multiple GPUs, default: 0"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  TP_SIZE=${TP_SIZE}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}, MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}, ATTN_PREFILL_CHUNK_SIZE=${ATTN_PREFILL_CHUNK_SIZE}"
echo "  BLOCK_SIZE=${BLOCK_SIZE}, MAX_NUM_SEQS=${MAX_NUM_SEQS}, GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}, KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "  KV_TRANSFER_CONFIG=${KV_TRANSFER_CONFIG}"
echo "  LMCACHE_LOCAL_CPU=${LMCACHE_LOCAL_CPU}, LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_MAX_LOCAL_CPU_SIZE}, LMCACHE_CHUNK_SIZE=${LMCACHE_CHUNK_SIZE}"
echo "  OFFLOAD_PROFILE=${OFFLOAD_PROFILE}, OFFLOAD_MIN_LOAD_TOKENS=${OFFLOAD_MIN_LOAD_TOKENS}, OFFLOAD_GPU_STAGING_CHUNKS=${OFFLOAD_GPU_STAGING_CHUNKS}"

case "${BACKGROUND,,}" in
  1|true|yes|y|bg|background)
    echo "Starting ATOM in background on port ${PORT} using GPUs ${GPU_IDS}; chunked prefill enabled; log: ${LOG_FILE}"
    nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
    echo "ATOM pid: $!"
    ;;
  0|false|no|n|fg|foreground)
    echo "Starting ATOM in foreground on port ${PORT} using GPUs ${GPU_IDS}; chunked prefill enabled"
    exec "${CMD[@]}"
    ;;
  *)
    echo "Invalid background value: ${BACKGROUND}"
    exit 1
    ;;
esac