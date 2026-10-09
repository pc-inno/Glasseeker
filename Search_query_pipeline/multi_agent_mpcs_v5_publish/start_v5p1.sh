#!/usr/bin/env bash
set -euo pipefail

# Launcher for the bundled Hermes checkout.
# Usage: ./start_v5p1.sh [--check | num_seeds run_id]
# Environment variables can override the defaults, for example:
#   NUM_SEEDS=100 SEED_CONCURRENCY=20 SOLVER_ROLLOUTS=6 SOLVER_CONCURRENCY=6 ./start_v5p1.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$SCRIPT_DIR
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [num_seeds=1] [run_id=v5_demo]"
  echo "       $0 --check   # import/config check; no model API call"
  echo "Setup: see README.md; copy .env_v5.example to .env_v5 and fill credentials."
  exit 0
fi
CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
  shift
fi
BASE_ENV=${ENV_FILE:-$PROJECT_ROOT/.env_v5}
HERMES_PROJECT_ROOT=${HERMES_PROJECT_ROOT:-$PROJECT_ROOT/hermes}
HERMES_VENV=${HERMES_VENV:-$HERMES_PROJECT_ROOT/.venv}
HERMES_HOME=${HERMES_HOME:-$PROJECT_ROOT/.hermes}
HERMES_COMMAND=$HERMES_PROJECT_ROOT/hermes
PYTHON_BIN=${PYTHON_BIN:-$HERMES_VENV/bin/python3}

NUM_SEEDS=${NUM_SEEDS:-${1:-1}}
RUN_ID=${RUN_ID:-${2:-v5_demo}}
SEED_CONCURRENCY=${SEED_CONCURRENCY:-1}
SEED_PREFETCH=${SEED_PREFETCH:-$SEED_CONCURRENCY}
SOLVER_ROLLOUTS=${SOLVER_ROLLOUTS:-3}
SOLVER_CONCURRENCY=${SOLVER_CONCURRENCY:-1}

if [[ ! -f "$BASE_ENV" ]]; then
  echo "missing environment file: $BASE_ENV" >&2
  exit 2
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Hermes Python is not executable: ${PYTHON_BIN:-<unset>}" >&2
  exit 2
fi
if [[ ! -x "$HERMES_COMMAND" ]]; then
  echo "Hermes command is not executable: $HERMES_COMMAND" >&2
  exit 2
fi
if [[ ! -f "$HERMES_HOME/config.yaml" ]]; then
  echo "missing Hermes config: $HERMES_HOME/config.yaml; see README.md" >&2
  exit 2
fi
if [[ ! "$NUM_SEEDS" =~ ^[0-9]+$ ]] || (( NUM_SEEDS < 1 || NUM_SEEDS > 100 )); then
  echo "num_seeds must be an integer in [1, 100]; got: $NUM_SEEDS" >&2
  exit 2
fi
if [[ ! "$SEED_CONCURRENCY" =~ ^[0-9]+$ ]] || (( SEED_CONCURRENCY < 1 )); then
  echo "SEED_CONCURRENCY must be a positive integer; got: $SEED_CONCURRENCY" >&2
  exit 2
fi
if [[ ! "$SEED_PREFETCH" =~ ^[0-9]+$ ]] || (( SEED_PREFETCH < SEED_CONCURRENCY )); then
  echo "SEED_PREFETCH must be >= SEED_CONCURRENCY; got: $SEED_PREFETCH" >&2
  exit 2
fi
if [[ ! "$SOLVER_ROLLOUTS" =~ ^[0-9]+$ ]] || (( SOLVER_ROLLOUTS < 3 )); then
  echo "SOLVER_ROLLOUTS must be >= 3; got: $SOLVER_ROLLOUTS" >&2
  exit 2
fi
if [[ ! "$SOLVER_CONCURRENCY" =~ ^[0-9]+$ ]] || (( SOLVER_CONCURRENCY < 1 )); then
  echo "SOLVER_CONCURRENCY must be a positive integer; got: $SOLVER_CONCURRENCY" >&2
  exit 2
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "run_id may contain only letters, digits, '.', '_' and '-'; got: $RUN_ID" >&2
  exit 2
fi

cd "$PROJECT_ROOT"

# config.load_dotenv intentionally reads the env file itself and overwrites the
# process environment. Append runtime overrides so the requested concurrency is
# guaranteed to win without mutating the checked-in .env_v5.
RUNTIME_ENV=$(mktemp "${TMPDIR:-/tmp}/hermes-v5p1-env.XXXXXX")
cleanup() {
  rm -f -- "$RUNTIME_ENV"
}
trap cleanup EXIT INT TERM
cp -- "$BASE_ENV" "$RUNTIME_ENV"
{
  printf '\n# Runtime overrides written by start_v5p1.sh\n'
  printf 'HERMES_VENV=%s\n' "$HERMES_VENV"
  printf 'HERMES_HOME=%s\n' "$HERMES_HOME"
  printf 'PYTHON_BIN=%s\n' "$PYTHON_BIN"
  printf 'REAL_HERMES_BIN=%s\n' "$HERMES_COMMAND"
  printf 'V2_HERMES_COMMAND=%s\n' "$HERMES_COMMAND"
  printf 'V2_HERMES_PATH=\n'
  printf 'V2_HERMES_SHARED_HOME=\n'
  printf 'V2_HERMES_SOLVER_WORKSPACE_ROOT=%s\n' "$PROJECT_ROOT/data/solver_workspaces"
  printf 'V2_SEED_CONCURRENCY=%s\n' "$SEED_CONCURRENCY"
  printf 'V2_SEED_PREFETCH=%s\n' "$SEED_PREFETCH"
  printf 'V2_SOLVER_ROLLOUTS=%s\n' "$SOLVER_ROLLOUTS"
  printf 'V2_SOLVER_CONCURRENCY=%s\n' "$SOLVER_CONCURRENCY"
  printf 'V2_OUTPUT_DIR=data/runs_%s\n' "$RUN_ID"
  printf 'V2_CONVERSATION_LOG_DIR=data/runs_%s/conversations\n' "$RUN_ID"
} >> "$RUNTIME_ENV"

if (( CHECK_ONLY )); then
  "$PYTHON_BIN" scripts/check_hermes_runtime.py --env "$RUNTIME_ENV"
  exit $?
fi

echo "Starting v5p1: seeds=$NUM_SEEDS seed_concurrency=$SEED_CONCURRENCY "\
  "solver_rollouts=$SOLVER_ROLLOUTS solver_concurrency=$SOLVER_CONCURRENCY "\
  "run_id=$RUN_ID"

"$PYTHON_BIN" scripts/run_v2_workflow.py \
  --env "$RUNTIME_ENV" \
  --run-id "$RUN_ID" \
  --auto-seed \
  --num-seeds "$NUM_SEEDS" \
  --resume
