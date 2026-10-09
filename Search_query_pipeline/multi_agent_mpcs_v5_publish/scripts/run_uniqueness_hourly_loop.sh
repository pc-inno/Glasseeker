#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUN_DIR=${V2_FILTER_RUN_DIR:-$PROJECT_ROOT/data/runs_v20}
LOG_FILE="$PROJECT_ROOT/data/filter_data/collect_uniqueness_hourly.log"
COLLECTOR="$PROJECT_ROOT/scripts/collect_uniqueness_incremental.py"
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3 || command -v python)}

while true; do
  "$PYTHON_BIN" "$COLLECTOR" --run-dir "$RUN_DIR" >>"$LOG_FILE" 2>&1 || true
  now=$(/bin/date +%s)
  delay=$((3600 - now % 3600))
  /bin/sleep "$delay"
done
