#!/usr/bin/env bash
set -euo pipefail

# Production AGS launcher. Credentials are consumed by the
# evaluator through environment variables and are never included in the
# launch summary or an error message emitted by this wrapper.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HERMES_REPO="$(cd "${EVAL_ROOT}/../Evaluation_backend" && pwd)"
cd "${EVAL_ROOT}"

die() {
  echo "Error: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage:
  scripts/run_browsecomp_ags.sh --model MODEL --base-url URL \
    --dataset DATASET --data-path PATH [options]

Experiment-specific options:
  --model NAME                  Model name sent to Hermes
  --models CSV                  Models aligned with --api-key-envs
  --base-url URL                Single model endpoint
  --base-urls CSV               Model endpoint pool
  --key-base-urls CSV           Endpoints aligned with --api-key-envs
  --endpoint-registry PATH      Dynamic registry v1 JSON instead of manual endpoints
  --endpoint-registry-refresh-interval N  Registry reload interval (default: 30)
  --workers-per-endpoint N      Concurrent AGS runs per endpoint (default: 10)
  --api-key-env NAME            Common model API-key environment variable
  --api-key-envs CSV            Model API-key environment variable names
  --api-modes CSV               chat_completions/codex_responses per key
  --workers-per-key CSV         Optional concurrency limit per key
  --dataset NAME                Dataset/output organization name
  --data-path PATH              JSON/JSONL question file
  --save-name NAME              Run name below OUTPUT_ROOT
  --output-root PATH            Prediction root (default: output/preds)
  --save-path PATH              Alias for --output-root
  --resume-from PATH            Read existing results/attempt counts from another run root
  --limit N                     Only run the first N questions
  --repeats N                   Repeats per question (default: 1)
  --context-compression BOOL    Enable context compression: true/false (default: true)
  --compression-threshold N     Compress at this context ratio, 0 < N <= 1 (default: 0.5)
  --compressor NAME             Built-in compressor: default/v3 (default: default)

Environment controls:
  SUBAGENT_MAX_ITERATIONS       Per-subagent iteration cap (default: 50)
  SUBAGENT_TIMEOUT_SECONDS      Per-subagent wall timeout; 0 disables (default: 900)
  WEB_EXTRACT_CONTENT_MODE      legacy/prefix_overflow (default: legacy)
  --expected-questions N        Validate the complete dataset size
  --log-file PATH               Console log path (default: logs/<run>_<timestamp>.log)
  --no-log                      Do not save console output
  --dry-run                     Print resolved configuration without AGS creation
  --endpoint-watchdog           Enable model endpoint monitoring (default)
  --no-endpoint-watchdog        Disable model endpoint monitoring
  --endpoint-watchdog-interval N  Seconds between probes (default: 30)
  --endpoint-watchdog-timeout N   /v1/models timeout (default: 5)
  --endpoint-watchdog-failures N  Failures before quarantine (default: 3)
  --endpoint-watchdog-recovery-successes N  Successes before re-entry (default: 2)
  --all-endpoints-down-timeout N  Continuous all-down seconds before exit 75 (default: 300)
  --rerun-failed               Rerun existing failed records
  --force                      Rerun all existing records
  -h, --help                   Show this help

Credentials stay in environment variables: E2B_API_KEY, HERMES_API_KEY (or
the names supplied via --api-key-envs), SEARCH_SERVER_ENDPOINT, and
SEARCH_SERVER_API_KEY. Fixed AGS/evaluation policy values have safe defaults
in this launcher and can still be overridden through uppercase environment
variables when an experiment explicitly requires it.
EOF
}

need_value() {
  [[ "$#" -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer"
}

nonnegative_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || die "${name} must be a non-negative integer"
}

EVAL_PYTHON="${EVAL_PYTHON:-${EVAL_ROOT}/.venv/bin/python}"
HERMES_API_KEY="${HERMES_API_KEY:-}"
export HERMES_API_KEY

E2B_DOMAIN="${E2B_DOMAIN:-}"
E2B_API_KEY="${E2B_API_KEY:-}"
MODEL="${MODEL:-}"
MODELS="${MODELS:-}"
BASE_URL="${BASE_URL:-${HERMES_BASE_URL:-}}"
BASE_URLS="${BASE_URLS:-${HERMES_BASE_URLS:-}}"
KEY_BASE_URLS="${KEY_BASE_URLS:-}"
API_KEY_ENVS="${API_KEY_ENVS:-}"
API_KEY_ENV="${API_KEY_ENV:-HERMES_API_KEY}"
API_MODES="${API_MODES:-}"
PROVIDER="${PROVIDER:-custom}"
DATASET="${DATASET:-}"
DATA_PATH="${DATA_PATH:-}"
SAVE_NAME="${SAVE_NAME:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/preds}"
RESUME_FROM="${RESUME_FROM:-}"
EXPECTED_QUESTIONS="${EXPECTED_QUESTIONS:-}"
LOG_FILE="${LOG_FILE:-}"
AUTO_LOG="${AUTO_LOG:-1}"
SKIP_STATUSES="${SKIP_STATUSES:-success,policy_violation}"
SEARCH_MODE="${SEARCH_MODE:-external}"
TOOL_WHITELIST="${TOOL_WHITELIST:-web,code_execution,delegation,vision}"
NUM_WORKERS="${NUM_WORKERS:-10}"
WORKERS_PER_ENDPOINT="${WORKERS_PER_ENDPOINT:-10}"
WORKERS_PER_KEY="${WORKERS_PER_KEY:-}"
REPEATS="${REPEATS:-1}"
MAX_RETRIES="${MAX_RETRIES:-3}"
MAX_ROUNDS="${MAX_ROUNDS:-500}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
CONTEXT_COMPRESSION="${CONTEXT_COMPRESSION:-true}"
COMPRESSION_THRESHOLD="${COMPRESSION_THRESHOLD:-0.5}"
CONTEXT_COMPRESSOR="${CONTEXT_COMPRESSOR:-default}"
WEB_EXTRACT_CONTENT_MODE="${WEB_EXTRACT_CONTENT_MODE:-legacy}"
SUBAGENT_MAX_ITERATIONS="${SUBAGENT_MAX_ITERATIONS:-50}"
SUBAGENT_TIMEOUT_SECONDS="${SUBAGENT_TIMEOUT_SECONDS:-900}"
export WEB_EXTRACT_CONTENT_MODE SUBAGENT_MAX_ITERATIONS SUBAGENT_TIMEOUT_SECONDS
REASONING_EFFORT="${REASONING_EFFORT:-}"
QUESTION_MATCH_MODE="${QUESTION_MATCH_MODE:-evaluation}"
ANTIHACK_ENABLED="${ANTIHACK_ENABLED:-1}"
MAX_TOKENS="${MAX_TOKENS:-}"
TEMPERATURE="${TEMPERATURE:-}"
LIMIT="${LIMIT:-}"
FORCE="${FORCE:-0}"
RERUN_FAILED="${RERUN_FAILED:-0}"
NUMBERED_OUTPUT="${NUMBERED_OUTPUT:-1}"
HERMES_QUIET="${HERMES_QUIET:-1}"
ENDPOINT_WATCHDOG="${ENDPOINT_WATCHDOG:-1}"
ENDPOINT_WATCHDOG_INTERVAL="${ENDPOINT_WATCHDOG_INTERVAL:-30}"
ENDPOINT_WATCHDOG_TIMEOUT="${ENDPOINT_WATCHDOG_TIMEOUT:-5}"
ENDPOINT_WATCHDOG_FAILURES="${ENDPOINT_WATCHDOG_FAILURES:-3}"
ENDPOINT_WATCHDOG_RECOVERY_SUCCESSES="${ENDPOINT_WATCHDOG_RECOVERY_SUCCESSES:-2}"
ALL_ENDPOINTS_DOWN_TIMEOUT="${ALL_ENDPOINTS_DOWN_TIMEOUT:-300}"
ENDPOINT_REGISTRY="${ENDPOINT_REGISTRY:-}"
ENDPOINT_REGISTRY_REFRESH_INTERVAL="${ENDPOINT_REGISTRY_REFRESH_INTERVAL:-30}"

AGS_TEMPLATE="${AGS_TEMPLATE:-}"
AGS_COMMAND_TIMEOUT="${AGS_COMMAND_TIMEOUT:-8400}"
AGS_LIFETIME_TIMEOUT="${AGS_LIFETIME_TIMEOUT:-21600}"
AGS_CREATE_REQUEST_TIMEOUT="${AGS_CREATE_REQUEST_TIMEOUT:-120}"
AGS_OPERATION_REQUEST_TIMEOUT="${AGS_OPERATION_REQUEST_TIMEOUT:-60}"
AGS_KILL_REQUEST_TIMEOUT="${AGS_KILL_REQUEST_TIMEOUT:-30}"
AGS_PYTHON_BIN="${AGS_PYTHON_BIN:-python3}"
AGS_HERMES_BIN="${AGS_HERMES_BIN:-/home/user/.local/bin/hermes}"
AGS_WORKSPACE_ROOT="${AGS_WORKSPACE_ROOT:-/home/user/workspace}"
AGS_UPLOAD_HERMES_SOURCE="${AGS_UPLOAD_HERMES_SOURCE:-1}"
AGS_FETCH_SERVER_SOURCE_DIR="${AGS_FETCH_SERVER_SOURCE_DIR:-${FETCH_SERVER_SOURCE_DIR:-}}"
AGS_FETCH_SERVER_INSTALL_DEPENDENCIES="${AGS_FETCH_SERVER_INSTALL_DEPENDENCIES:-1}"
AGS_FETCH_SERVER_PORT="${AGS_FETCH_SERVER_PORT:-18081}"
AGS_FETCH_SERVER_STARTUP_TIMEOUT="${AGS_FETCH_SERVER_STARTUP_TIMEOUT:-120}"
AGS_FETCH_SERVER_INSTALL_TIMEOUT="${AGS_FETCH_SERVER_INSTALL_TIMEOUT:-600}"
AGS_FETCH_SERVER_PROBE_URL="${AGS_FETCH_SERVER_PROBE_URL:-}"
AGS_FORWARD_HOST_PROXY="${AGS_FORWARD_HOST_PROXY:-0}"

passthrough_args=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --model)
      need_value "$@"; MODEL="$2"; shift 2 ;;
    --models)
      need_value "$@"; MODELS="$2"; shift 2 ;;
    --base-url)
      need_value "$@"; BASE_URL="$2"; BASE_URLS=""; KEY_BASE_URLS=""; shift 2 ;;
    --base-urls)
      need_value "$@"; BASE_URLS="$2"; BASE_URL=""; KEY_BASE_URLS=""; shift 2 ;;
    --key-base-urls)
      need_value "$@"; KEY_BASE_URLS="$2"; BASE_URL=""; BASE_URLS=""; shift 2 ;;
    --endpoint-registry)
      need_value "$@"; ENDPOINT_REGISTRY="$2"; shift 2 ;;
    --endpoint-registry-refresh-interval)
      need_value "$@"; ENDPOINT_REGISTRY_REFRESH_INTERVAL="$2"; shift 2 ;;
    --workers-per-endpoint)
      need_value "$@"; WORKERS_PER_ENDPOINT="$2"; shift 2 ;;
    --api-key-env)
      need_value "$@"; API_KEY_ENV="$2"; shift 2 ;;
    --api-key-envs)
      need_value "$@"; API_KEY_ENVS="$2"; shift 2 ;;
    --api-modes)
      need_value "$@"; API_MODES="$2"; shift 2 ;;
    --workers-per-key)
      need_value "$@"; WORKERS_PER_KEY="$2"; shift 2 ;;
    --dataset)
      need_value "$@"; DATASET="$2"; shift 2 ;;
    --data-path)
      need_value "$@"; DATA_PATH="$2"; shift 2 ;;
    --save-name)
      need_value "$@"; SAVE_NAME="$2"; shift 2 ;;
    --output-root|--save-path)
      need_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --resume-from)
      need_value "$@"; RESUME_FROM="$2"; shift 2 ;;
    --limit)
      need_value "$@"; LIMIT="$2"; shift 2 ;;
    --repeats)
      need_value "$@"; REPEATS="$2"; shift 2 ;;
    --context-compression)
      need_value "$@"; CONTEXT_COMPRESSION="$2"; shift 2 ;;
    --compression-threshold)
      need_value "$@"; COMPRESSION_THRESHOLD="$2"; shift 2 ;;
    --compressor)
      need_value "$@"; CONTEXT_COMPRESSOR="$2"; shift 2 ;;
    --expected-questions)
      need_value "$@"; EXPECTED_QUESTIONS="$2"; shift 2 ;;
    --log-file)
      need_value "$@"; LOG_FILE="$2"; AUTO_LOG=1; shift 2 ;;
    --no-log)
      AUTO_LOG=0; shift ;;
    --dry-run)
      passthrough_args+=(--dry-run); shift ;;
    --endpoint-watchdog)
      ENDPOINT_WATCHDOG=1; shift ;;
    --no-endpoint-watchdog)
      ENDPOINT_WATCHDOG=0; shift ;;
    --endpoint-watchdog-interval)
      need_value "$@"; ENDPOINT_WATCHDOG_INTERVAL="$2"; shift 2 ;;
    --endpoint-watchdog-timeout)
      need_value "$@"; ENDPOINT_WATCHDOG_TIMEOUT="$2"; shift 2 ;;
    --endpoint-watchdog-failures)
      need_value "$@"; ENDPOINT_WATCHDOG_FAILURES="$2"; shift 2 ;;
    --endpoint-watchdog-recovery-successes)
      need_value "$@"; ENDPOINT_WATCHDOG_RECOVERY_SUCCESSES="$2"; shift 2 ;;
    --all-endpoints-down-timeout)
      need_value "$@"; ALL_ENDPOINTS_DOWN_TIMEOUT="$2"; shift 2 ;;
    --rerun-failed)
      RERUN_FAILED=1; shift ;;
    --force)
      FORCE=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift; passthrough_args+=("$@"); break ;;
    *)
      passthrough_args+=("$1"); shift ;;
  esac
done

PRIMARY_MODEL="${MODEL:-${MODELS%%,*}}"
[[ -n "${PRIMARY_MODEL}" ]] || die "--model or --models is required"
if [[ -n "${ENDPOINT_REGISTRY}" ]]; then
  [[ -z "${BASE_URL}" && -z "${BASE_URLS}" && -z "${KEY_BASE_URLS}" ]] \
    || die "--endpoint-registry cannot be combined with --base-url, --base-urls, or --key-base-urls"
  [[ -z "${MODELS}" ]] || die "--endpoint-registry cannot be combined with --models"
  [[ -z "${API_KEY_ENVS}" ]] || die "--endpoint-registry cannot be combined with --api-key-envs"
  [[ -z "${WORKERS_PER_KEY}" ]] || die "--endpoint-registry cannot be combined with --workers-per-key"
  [[ "${ENDPOINT_WATCHDOG}" == 1 ]] || die "--endpoint-registry requires endpoint watchdog"
  if [[ ! "${ENDPOINT_REGISTRY}" = /* ]]; then
    ENDPOINT_REGISTRY="${EVAL_ROOT}/${ENDPOINT_REGISTRY}"
  fi
else
  [[ -n "${BASE_URL}" || -n "${BASE_URLS}" || -n "${KEY_BASE_URLS}" ]] \
    || die "--base-url, --base-urls, --key-base-urls, or --endpoint-registry is required"
fi
[[ -n "${DATASET}" ]] || die "--dataset is required"
[[ -n "${DATA_PATH}" ]] || die "--data-path is required"
[[ -n "${E2B_API_KEY}" ]] || die "set E2B_API_KEY"
[[ -n "${E2B_DOMAIN}" ]] || die "set E2B_DOMAIN"
[[ -n "${AGS_TEMPLATE}" ]] || die "set AGS_TEMPLATE"
SAVE_NAME="${SAVE_NAME:-${PRIMARY_MODEL}_ags}"
if [[ ! "${DATA_PATH}" = /* ]]; then
  DATA_PATH="${EVAL_ROOT}/${DATA_PATH}"
fi

if [[ "${AUTO_LOG}" == 1 ]]; then
  if [[ -z "${LOG_FILE}" ]]; then
    log_run_name="${SAVE_NAME//\//_}"
    log_run_name="$(printf '%s' "${log_run_name}" | tr -cs '[:alnum:]_.-' '_')"
    log_dataset="${DATASET//\//_}"
    log_dataset="$(printf '%s' "${log_dataset}" | tr -cs '[:alnum:]_.-' '_')"
    LOG_FILE="logs/${log_run_name}_${log_dataset}_$(date '+%Y%m%d_%H%M%S').log"
  fi
  if [[ ! "${LOG_FILE}" = /* ]]; then
    LOG_FILE="${EVAL_ROOT}/${LOG_FILE}"
  fi
  mkdir -p "$(dirname "${LOG_FILE}")"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  echo "Console log: ${LOG_FILE}"
fi

positive_integer NUM_WORKERS "${NUM_WORKERS}"
positive_integer WORKERS_PER_ENDPOINT "${WORKERS_PER_ENDPOINT}"
positive_integer REPEATS "${REPEATS}"
nonnegative_integer MAX_RETRIES "${MAX_RETRIES}"
positive_integer MAX_ROUNDS "${MAX_ROUNDS}"
positive_integer TIMEOUT_SECONDS "${TIMEOUT_SECONDS}"
positive_integer SUBAGENT_MAX_ITERATIONS "${SUBAGENT_MAX_ITERATIONS}"
nonnegative_integer SUBAGENT_TIMEOUT_SECONDS "${SUBAGENT_TIMEOUT_SECONDS}"
positive_integer AGS_COMMAND_TIMEOUT "${AGS_COMMAND_TIMEOUT}"
positive_integer AGS_LIFETIME_TIMEOUT "${AGS_LIFETIME_TIMEOUT}"
positive_integer AGS_CREATE_REQUEST_TIMEOUT "${AGS_CREATE_REQUEST_TIMEOUT}"
positive_integer AGS_OPERATION_REQUEST_TIMEOUT "${AGS_OPERATION_REQUEST_TIMEOUT}"
positive_integer AGS_KILL_REQUEST_TIMEOUT "${AGS_KILL_REQUEST_TIMEOUT}"
positive_integer AGS_FETCH_SERVER_PORT "${AGS_FETCH_SERVER_PORT}"
positive_integer AGS_FETCH_SERVER_STARTUP_TIMEOUT "${AGS_FETCH_SERVER_STARTUP_TIMEOUT}"
positive_integer AGS_FETCH_SERVER_INSTALL_TIMEOUT "${AGS_FETCH_SERVER_INSTALL_TIMEOUT}"
positive_integer ENDPOINT_WATCHDOG_FAILURES "${ENDPOINT_WATCHDOG_FAILURES}"
positive_integer ENDPOINT_WATCHDOG_RECOVERY_SUCCESSES "${ENDPOINT_WATCHDOG_RECOVERY_SUCCESSES}"
if [[ -n "${CONTEXT_LENGTH}" ]]; then positive_integer CONTEXT_LENGTH "${CONTEXT_LENGTH}"; fi
if [[ -n "${MAX_TOKENS}" ]]; then positive_integer MAX_TOKENS "${MAX_TOKENS}"; fi
if [[ -n "${TEMPERATURE}" && ! "${TEMPERATURE}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  die "TEMPERATURE must be a non-negative number"
fi
if [[ -n "${LIMIT}" ]]; then positive_integer LIMIT "${LIMIT}"; fi
if [[ -n "${EXPECTED_QUESTIONS}" ]]; then
  positive_integer EXPECTED_QUESTIONS "${EXPECTED_QUESTIONS}"
fi
(( TIMEOUT_SECONDS < AGS_COMMAND_TIMEOUT )) \
  || die "TIMEOUT_SECONDS must be less than AGS_COMMAND_TIMEOUT"
(( AGS_COMMAND_TIMEOUT < AGS_LIFETIME_TIMEOUT )) \
  || die "AGS_COMMAND_TIMEOUT must be less than AGS_LIFETIME_TIMEOUT"
[[ "${FORCE}" == 0 || "${FORCE}" == 1 ]] || die "FORCE must be 0 or 1"
[[ "${RERUN_FAILED}" == 0 || "${RERUN_FAILED}" == 1 ]] || die "RERUN_FAILED must be 0 or 1"
[[ "${NUMBERED_OUTPUT}" == 0 || "${NUMBERED_OUTPUT}" == 1 ]] || die "NUMBERED_OUTPUT must be 0 or 1"
[[ "${HERMES_QUIET}" == 0 || "${HERMES_QUIET}" == 1 ]] || die "HERMES_QUIET must be 0 or 1"
[[ "${AUTO_LOG}" == 0 || "${AUTO_LOG}" == 1 ]] || die "AUTO_LOG must be 0 or 1"
[[ "${ANTIHACK_ENABLED}" == 0 || "${ANTIHACK_ENABLED}" == 1 ]] || die "ANTIHACK_ENABLED must be 0 or 1"
[[ "${ENDPOINT_WATCHDOG}" == 0 || "${ENDPOINT_WATCHDOG}" == 1 ]] || die "ENDPOINT_WATCHDOG must be 0 or 1"
[[ "${API_KEY_ENV}" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]] || die "invalid API_KEY_ENV: ${API_KEY_ENV}"
if [[ -z "${API_KEY_ENVS}" ]]; then
  [[ -n "${!API_KEY_ENV:-}" ]] || die "API key environment variable is not set: ${API_KEY_ENV}"
fi
for pair in \
  "ENDPOINT_WATCHDOG_INTERVAL:${ENDPOINT_WATCHDOG_INTERVAL}" \
  "ENDPOINT_WATCHDOG_TIMEOUT:${ENDPOINT_WATCHDOG_TIMEOUT}" \
  "ENDPOINT_REGISTRY_REFRESH_INTERVAL:${ENDPOINT_REGISTRY_REFRESH_INTERVAL}" \
  "ALL_ENDPOINTS_DOWN_TIMEOUT:${ALL_ENDPOINTS_DOWN_TIMEOUT}"; do
  label="${pair%%:*}"
  value="${pair#*:}"
  [[ "${value}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ && "${value}" =~ [1-9] ]] \
    || die "${label} must be a positive number"
done
if [[ "${ENDPOINT_WATCHDOG}" == 1 ]] && ! awk \
  -v timeout="${ENDPOINT_WATCHDOG_TIMEOUT}" \
  -v interval="${ENDPOINT_WATCHDOG_INTERVAL}" \
  'BEGIN { exit !(timeout < interval) }'; then
  die "ENDPOINT_WATCHDOG_TIMEOUT must be less than ENDPOINT_WATCHDOG_INTERVAL"
fi
[[ "${CONTEXT_COMPRESSION}" == true || "${CONTEXT_COMPRESSION}" == false ]] \
  || die "CONTEXT_COMPRESSION must be true or false"
[[ "${CONTEXT_COMPRESSOR}" == default || "${CONTEXT_COMPRESSOR}" == v3 ]] \
  || die "CONTEXT_COMPRESSOR must be default or v3"
if [[ ! "${COMPRESSION_THRESHOLD}" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]]; then
  die "COMPRESSION_THRESHOLD must be greater than 0 and at most 1"
fi
[[ "${WEB_EXTRACT_CONTENT_MODE}" == legacy \
  || "${WEB_EXTRACT_CONTENT_MODE}" == prefix_overflow ]] \
  || die "WEB_EXTRACT_CONTENT_MODE must be legacy or prefix_overflow"
[[ "${QUESTION_MATCH_MODE}" == evaluation || "${QUESTION_MATCH_MODE}" == off ]] \
  || die "QUESTION_MATCH_MODE must be evaluation or off"
[[ -z "${REASONING_EFFORT}" || "${REASONING_EFFORT}" =~ ^(low|medium|high|max)$ ]] \
  || die "REASONING_EFFORT must be low, medium, high, or max"
[[ -z "${WORKERS_PER_KEY}" || "${WORKERS_PER_KEY}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] \
  || die "WORKERS_PER_KEY must be one or more comma-separated positive integers"
[[ "${AGS_UPLOAD_HERMES_SOURCE}" == 0 || "${AGS_UPLOAD_HERMES_SOURCE}" == 1 ]] \
  || die "AGS_UPLOAD_HERMES_SOURCE must be 0 or 1"
[[ "${AGS_FETCH_SERVER_INSTALL_DEPENDENCIES}" == 0 || "${AGS_FETCH_SERVER_INSTALL_DEPENDENCIES}" == 1 ]] \
  || die "AGS_FETCH_SERVER_INSTALL_DEPENDENCIES must be 0 or 1"
[[ "${AGS_FORWARD_HOST_PROXY}" == 0 || "${AGS_FORWARD_HOST_PROXY}" == 1 ]] \
  || die "AGS_FORWARD_HOST_PROXY must be 0 or 1"
[[ -n "${AGS_FETCH_SERVER_SOURCE_DIR}" && -f "${AGS_FETCH_SERVER_SOURCE_DIR}/server.py" ]] \
  || die "AGS_FETCH_SERVER_SOURCE_DIR must contain server.py"
[[ -f "${AGS_FETCH_SERVER_SOURCE_DIR}/requirements.lock.txt" \
  || -f "${AGS_FETCH_SERVER_SOURCE_DIR}/requirements.txt" ]] \
  || die "AGS_FETCH_SERVER_SOURCE_DIR must contain requirements.lock.txt or requirements.txt"
[[ "${AGS_UPLOAD_HERMES_SOURCE}" == 1 ]] \
  || die "the per-sandbox Fetch Server requires AGS_UPLOAD_HERMES_SOURCE=1"
[[ -f "${DATA_PATH}" ]] || die "dataset not found: ${DATA_PATH}"
[[ -x "${EVAL_PYTHON}" ]] || die "evaluation Python not found or not executable: ${EVAL_PYTHON}"

question_count="$(awk 'NF {count++} END {print count+0}' "${DATA_PATH}")"
if [[ -n "${EXPECTED_QUESTIONS}" && "${question_count}" != "${EXPECTED_QUESTIONS}" ]]; then
  die "expected ${EXPECTED_QUESTIONS} questions, found ${question_count}: ${DATA_PATH}"
fi

if [[ "${SEARCH_MODE}" == external ]]; then
  SEARCH_SERVER_ENDPOINT="${SEARCH_SERVER_ENDPOINT:-${EXTERNAL_SEARCH_ENDPOINT:-${SEARCH_ENDPOINT:-}}}"
  SEARCH_SERVER_API_KEY="${SEARCH_SERVER_API_KEY:-${EXTERNAL_SEARCH_API_KEY:-${SEARCH_API_KEY:-}}}"
  [[ -n "${SEARCH_SERVER_ENDPOINT}" ]] \
    || die "set SEARCH_SERVER_ENDPOINT for external search"
  [[ -n "${SEARCH_SERVER_API_KEY}" ]] \
    || die "set SEARCH_SERVER_API_KEY for external search"
  export SEARCH_SERVER_ENDPOINT SEARCH_SERVER_API_KEY
fi

export HERMES_CONTEXT_COMPRESSOR="${CONTEXT_COMPRESSOR}"

# A localhost proxy on the trusted host is not reachable from a remote AGS
# sandbox and can silently hijack private vLLM/Search traffic. VPC-direct is
# the production default; forwarding host proxy settings requires opt-in.
if [[ "${AGS_FORWARD_HOST_PROXY}" == 0 ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
  unset http_proxy https_proxy all_proxy no_proxy
fi

args=(
  "${EVAL_PYTHON}" -m browse_comp_eval.run
  --dataset "${DATASET}"
  --data-path "${DATA_PATH}"
  --provider "${PROVIDER}"
  --save-name "${SAVE_NAME}"
  --output-root "${OUTPUT_ROOT}"
  --skip-statuses "${SKIP_STATUSES}"
  --repeats "${REPEATS}"
  --num-workers "${NUM_WORKERS}"
  --max-rounds "${MAX_ROUNDS}"
  --max-retries "${MAX_RETRIES}"
  --timeout-seconds "${TIMEOUT_SECONDS}"
  --context-length "${CONTEXT_LENGTH}"
  --context-compression "${CONTEXT_COMPRESSION}"
  --compression-threshold "${COMPRESSION_THRESHOLD}"
  --tool-whitelist "${TOOL_WHITELIST}"
  --endpoint-watchdog-interval "${ENDPOINT_WATCHDOG_INTERVAL}"
  --endpoint-watchdog-timeout "${ENDPOINT_WATCHDOG_TIMEOUT}"
  --endpoint-watchdog-failures "${ENDPOINT_WATCHDOG_FAILURES}"
  --endpoint-watchdog-recovery-successes "${ENDPOINT_WATCHDOG_RECOVERY_SUCCESSES}"
  --all-endpoints-down-timeout "${ALL_ENDPOINTS_DOWN_TIMEOUT}"
  --endpoint-registry-refresh-interval "${ENDPOINT_REGISTRY_REFRESH_INTERVAL}"
  --execution-backend ags
  --search-mode "${SEARCH_MODE}"
  --question-match-mode "${QUESTION_MATCH_MODE}"
  --ignore-rules
  --ags-api-key-env E2B_API_KEY
  --ags-domain "${E2B_DOMAIN}"
  --ags-template "${AGS_TEMPLATE}"
  --ags-lifetime-timeout "${AGS_LIFETIME_TIMEOUT}"
  --ags-command-timeout "${AGS_COMMAND_TIMEOUT}"
  --ags-create-request-timeout "${AGS_CREATE_REQUEST_TIMEOUT}"
  --ags-operation-request-timeout "${AGS_OPERATION_REQUEST_TIMEOUT}"
  --ags-kill-request-timeout "${AGS_KILL_REQUEST_TIMEOUT}"
  --ags-container-python "${AGS_PYTHON_BIN}"
  --ags-container-hermes "${AGS_HERMES_BIN}"
  --ags-workspace "${AGS_WORKSPACE_ROOT}"
  --ags-fetch-server-source-dir "${AGS_FETCH_SERVER_SOURCE_DIR}"
  --ags-fetch-server-port "${AGS_FETCH_SERVER_PORT}"
  --ags-fetch-server-startup-timeout "${AGS_FETCH_SERVER_STARTUP_TIMEOUT}"
  --ags-fetch-server-install-timeout "${AGS_FETCH_SERVER_INSTALL_TIMEOUT}"
  --ags-upload-bundle
)

if [[ "${ENDPOINT_WATCHDOG}" == 1 ]]; then
  args+=(--endpoint-watchdog)
else
  args+=(--no-endpoint-watchdog)
fi

if [[ -n "${RESUME_FROM}" ]]; then args+=(--resume-from "${RESUME_FROM}"); fi

if [[ -n "${MODEL}" ]]; then args+=(--model "${MODEL}"); fi
if [[ -n "${MODELS}" ]]; then args+=(--models "${MODELS}"); fi
if [[ -n "${ENDPOINT_REGISTRY}" ]]; then
  args+=(--endpoint-registry "${ENDPOINT_REGISTRY}" --workers-per-endpoint "${WORKERS_PER_ENDPOINT}")
elif [[ -n "${KEY_BASE_URLS}" ]]; then
  args+=(--key-base-urls "${KEY_BASE_URLS}" --workers-per-endpoint "${WORKERS_PER_ENDPOINT}")
elif [[ -n "${BASE_URLS}" ]]; then
  args+=(--base-urls "${BASE_URLS}" --workers-per-endpoint "${WORKERS_PER_ENDPOINT}")
else
  args+=(--base-url "${BASE_URL}")
fi
if [[ -n "${API_KEY_ENVS}" ]]; then
  args+=(--api-key-envs "${API_KEY_ENVS}")
else
  args+=(--api-key-env "${API_KEY_ENV}")
fi
if [[ -n "${API_MODES}" ]]; then args+=(--api-modes "${API_MODES}"); fi
if [[ -n "${WORKERS_PER_KEY}" ]]; then args+=(--workers-per-key "${WORKERS_PER_KEY}"); fi
if [[ -n "${REASONING_EFFORT}" ]]; then args+=(--reasoning-effort "${REASONING_EFFORT}"); fi

if [[ -n "${AGS_FETCH_SERVER_PROBE_URL}" ]]; then
  args+=(--ags-fetch-server-probe-url "${AGS_FETCH_SERVER_PROBE_URL}")
fi

if [[ "${AGS_FETCH_SERVER_INSTALL_DEPENDENCIES}" == 1 ]]; then
  args+=(--ags-fetch-server-install-dependencies)
else
  args+=(--no-ags-fetch-server-install-dependencies)
fi

if [[ "${AGS_UPLOAD_HERMES_SOURCE}" == 1 ]]; then
  args+=(--ags-upload-hermes-source)
else
  args+=(--no-ags-upload-hermes-source)
fi
if [[ -n "${MAX_TOKENS}" ]]; then args+=(--max-tokens "${MAX_TOKENS}"); fi
if [[ -n "${TEMPERATURE}" ]]; then args+=(--temperature "${TEMPERATURE}"); fi
if [[ -n "${LIMIT}" ]]; then args+=(--limit "${LIMIT}"); fi
[[ "${FORCE}" == 1 ]] && args+=(--force)
[[ "${RERUN_FAILED}" == 1 ]] && args+=(--rerun-failed)
[[ "${ANTIHACK_ENABLED}" == 0 ]] && args+=(--no-antihack)
if [[ "${NUMBERED_OUTPUT}" == 1 ]]; then
  args+=(--numbered-output)
else
  args+=(--no-numbered-output)
fi
[[ "${HERMES_QUIET}" == 0 ]] && args+=(--no-quiet)

if [[ -n "${ENDPOINT_REGISTRY}" ]]; then
  printf 'backend=ags search=%s dataset=%s model=%s compressor=%s workers=dynamic(%s/endpoint) registry=%s\n' \
    "${SEARCH_MODE}" "${DATASET}" "${PRIMARY_MODEL}" "${CONTEXT_COMPRESSOR}" "${WORKERS_PER_ENDPOINT}" "${ENDPOINT_REGISTRY}"
else
  printf 'backend=ags search=%s dataset=%s model=%s compressor=%s workers=%s\n' \
    "${SEARCH_MODE}" "${DATASET}" "${PRIMARY_MODEL}" "${CONTEXT_COMPRESSOR}" "${NUM_WORKERS}"
fi

# Additional evaluator flags are intentionally passed through unchanged;
# command-line values appear after the defaults and therefore can override
# them using argparse's normal last-value-wins behaviour.
if [[ "${#passthrough_args[@]}" -gt 0 ]]; then
  exec "${args[@]}" "${passthrough_args[@]}"
else
  exec "${args[@]}"
fi
