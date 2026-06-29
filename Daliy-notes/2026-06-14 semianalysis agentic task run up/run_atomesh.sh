#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-30000}"
BACKGROUND="${2:-false}"
LOG_DIR="${LOG_DIR:-./atomesh_logs}"
LOG_FILE="${LOG_FILE:-atomesh_${PORT}.log}"
ATOMESH_BIN="${ATOMESH_BIN:-/data/mesh-all/ATOM/atom/mesh/target/release/atomesh}"
POLICY="${POLICY:-cache_aware}"

shift $(( $# < 2 ? $# : 2 ))

# Do not set worker information when launching ATOMesh.
CMD=(
  "$ATOMESH_BIN" launch
  --host 0.0.0.0
  --port "$PORT"
  --log-dir "$LOG_DIR"
  --log-level info
  --policy "$POLICY"
  "$@"
)

echo "Usage: $0 [port] [background] [atomesh args...]"
echo "  background: true/false, yes/no, bg/fg, 1/0"
echo "  POLICY=${POLICY}"
echo "  atomesh args: extra args appended after '${ATOMESH_BIN} launch'"

mkdir -p "$LOG_DIR"

case "${BACKGROUND,,}" in
  1|true|yes|y|bg|background)
    echo "Starting ATOMesh in background on port ${PORT}; policy: ${POLICY}; log: ${LOG_FILE}"
    nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
    echo "ATOMesh pid: $!"
    ;;
  0|false|no|n|fg|foreground)
    echo "Starting ATOMesh in foreground on port ${PORT}; policy: ${POLICY}"
    exec "${CMD[@]}"
    ;;
  *)
    echo "Invalid background value: ${BACKGROUND}"
    exit 1
    ;;
esac
