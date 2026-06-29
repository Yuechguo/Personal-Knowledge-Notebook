#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ATOMESH_HOST="${1:-127.0.0.1}"
ATOMESH_PORT="${2:-30000}"
MODEL="${3:-/data/models/Qwen3.5-27B-FP8}"
DATASET="${4:-semianalysis_cc_traces_weka_062126_256k}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${SCRIPT_DIR}/aiperf_artifacts}"

shift $(( $# < 4 ? $# : 4 ))

case "$DATASET" in
  062126|weka_062126|semianalysis_cc_traces_weka_062126|cc-traces-weka-062126|full)
    DATASET="semianalysis_cc_traces_weka_062126"
    ;;
  062126_256k|weka_062126_256k|semianalysis_cc_traces_weka_062126_256k|cc-traces-weka-062126-256k|256k)
    DATASET="semianalysis_cc_traces_weka_062126_256k"
    ;;
esac

export PYTHONPATH="${SCRIPT_DIR}/aiperf/src${PYTHONPATH:+:${PYTHONPATH}}"
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES:-0}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AIPERF_DATASET_CONFIGURATION_TIMEOUT:-1800}"
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT:-1800}"

if command -v aiperf >/dev/null 2>&1; then
  AIPERF_CMD=(aiperf)
else
  AIPERF_CMD=(python3 -m aiperf)
fi

mkdir -p "$ARTIFACT_DIR"

CMD=(
  "${AIPERF_CMD[@]}" profile
  --scenario inferencex-agentx-mvp
  --url "http://${ATOMESH_HOST}:${ATOMESH_PORT}"
  --endpoint /v1/chat/completions
  --endpoint-type chat
  --streaming
  --model "$MODEL"
  --concurrency "${CONCURRENCY:-16}"
  --benchmark-duration "${BENCHMARK_DURATION:-1800}"
  --random-seed "${RANDOM_SEED:-42}"
  --failed-request-threshold "${FAILED_REQUEST_THRESHOLD:-0.10}"
  --trajectory-start-min-ratio "${TRAJECTORY_START_MIN_RATIO:-0.25}"
  --trajectory-start-max-ratio "${TRAJECTORY_START_MAX_RATIO:-0.75}"
  --agentic-cache-warmup-duration "${AGENTIC_CACHE_WARMUP_DURATION:-600}"
  --use-server-token-count
  --no-gpu-telemetry
  --tokenizer-trust-remote-code
  --num-dataset-entries "${NUM_DATASET_ENTRIES:-393}"
  --slice-duration "${SLICE_DURATION:-1.0}"
  --output-artifact-dir "$ARTIFACT_DIR"
  --public-dataset "$DATASET"
  "$@"
)

echo "Usage: $0 [atomesh_host] [atomesh_port] [model] [dataset] [aiperf args...]"
echo "  atomesh_host: default 127.0.0.1"
echo "  atomesh_port: default 30000"
echo "  model: default /data/models/Qwen3-0.6B"
echo "  dataset: default semianalysis_cc_traces_weka_062126"
echo "           aliases: full, 256k, cc-traces-weka-062126, cc-traces-weka-062126-256k"
echo "Running AIPerf against http://${ATOMESH_HOST}:${ATOMESH_PORT}"
echo "AIPerf: ${AIPERF_CMD[*]}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Artifacts: ${ARTIFACT_DIR}"

exec "${CMD[@]}"
