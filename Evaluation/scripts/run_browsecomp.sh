#!/usr/bin/env bash
# Generic launcher for BrowseComp evaluations against one or more
# OpenAI-compatible model endpoints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HERMES_REPO="$(cd "${EVAL_ROOT}/../Evaluation_backend" && pwd)"
WORKSPACE="${EVAL_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_browsecomp.sh [options]

Required (or set the corresponding uppercase environment variable):
  --model NAME                  Model name sent to Hermes
  --models NAMES                Models aligned one-to-one with --api-key-envs
  --base-urls URLS              Comma-separated OpenAI-compatible endpoints
  --endpoint-registry PATH      Dynamic registry v1 JSON instead of manual endpoints
  --dataset NAME                Dataset/output organization name
  --data-path PATH              JSON or JSONL question file

Common options:
  --base-url URL                Add one endpoint; may be specified repeatedly
  --key-base-urls URLS          URLs aligned one-to-one with --api-key-envs
  --api-modes MODES             API modes aligned with keys (chat_completions/codex_responses)
  --workers-per-endpoint N      Concurrent runs per endpoint (default: 10)
  --endpoint-registry-refresh-interval N  Seconds between registry reloads (default: 30)
  --workers-per-key LIMITS      One limit for all keys, or aligned CSV limits
  --save-name NAME              Output run name (default: safe model name)
  --repeats N                   Runs per question (default: 1)
  --expected-questions N        Validate the dataset size; omitted = any size
  --provider NAME               Hermes provider (default: custom)
  --api-key KEY                 API key (prefer setting the key environment variable)
  --api-key-env NAME            API key environment variable (default: HERMES_API_KEY)
  --api-key-envs NAMES          Comma-separated API key environment variables
  --output-root PATH            Prediction output root (default: output/preds)
  --workspace-namespace NAME    Batch prefix for workspace directories (for example RUN_TAG)
  --enable-custom-system-prompt BOOL
                                Replace the main Agent System Prompt: true/false (default: false)
  --custom-system-prompt-file PATH
                                UTF-8 System Prompt file; required only when the gate is true
  --enable-subagent-custom-system-prompt BOOL
                                Replace the Subagent Hermes core System Prompt: true/false (default: false)
  --subagent-custom-system-prompt-file PATH
                                UTF-8 Subagent core System Prompt file; required only when the gate is true
  --resume-from PATH            Read existing results/attempt counts from another run root
  --log-file PATH               Console log path (default: logs/<run>_<timestamp>.log)
  --no-log                      Do not save the console output
  --hermes-bin PATH             Hermes executable
  --eval-python PATH            Python used to run the evaluator

Run controls:
  --max-rounds N                Maximum agent rounds (default: 500)
  --max-retries N               Maximum attempts for each run (default: 3)
  --timeout-seconds N           Per-attempt timeout (default: 3600)
  --context-length N            Model context length (default: 131072)
  --context-compression BOOL    Enable context compression: true/false (default: true)
  --compression-threshold N     Compress at this context ratio, 0 < N <= 1 (default: 0.5)
  --reasoning-effort LEVEL      Optional: low, medium, high, or max (default: API/model default)
  --max-tokens N                Optional output-token cap (default: model config)
  --temperature N               Optional temperature (default: model config)
  --endpoint-watchdog           Enable model endpoint monitoring (default)
  --no-endpoint-watchdog        Disable model endpoint monitoring
  --endpoint-watchdog-interval N  Seconds between probes (default: 30)
  --endpoint-watchdog-timeout N   /v1/models timeout (default: 5)
  --endpoint-watchdog-failures N  Failures before quarantine (default: 3)
  --endpoint-watchdog-recovery-successes N  Successes before re-entry (default: 2)
  --all-endpoints-down-timeout N  Continuous all-down seconds before exit 75 (default: 300)
  --tool-whitelist CSV          Enabled Hermes toolsets
  --skill-whitelist CSV         Preloaded Hermes skills
  --enable-browser              Enable browser tool and BROWSER_CDP_URL
  --browser-cdp-url URL         Browser CDP endpoint
  --limit N                     Only evaluate the first N questions
  --rerun-failed               Rerun existing failed results (default: skip them)
  --force                       Overwrite/resume by rerunning existing results
  --numbered-output             Name conv paths by dataset order (q000001, q000002, ...)
  --no-antihack                 Disable the BrowseComp anti-cheat guard plugin
  --no-quiet                    Show Hermes subprocess output
  --fetch-server-url URL        Fetch server used by Hermes and the watchdog
  --fetch-watchdog-url URL      Watchdog URL; must match Hermes fetch server
  --fetch-watchdog-interval N   Seconds between functional probes (default: 300)
  --fetch-watchdog-timeout N    Functional per-target timeout (default: 30)
  --fetch-watchdog-failures N   Functional failures before abort (default: 2)
  --fetch-watchdog-health-interval N  Seconds between /health probes (default: 10)
  --fetch-watchdog-health-timeout N   /health timeout in seconds (default: 3)
  --fetch-watchdog-health-failures N  /health failures before abort (default: 3)
  --fetch-watchdog-target URLS  Comma-separated URLs for functional /fetch probes
  --fetch-watchdog-health-only  Check /health without functional fetch probes
  --no-fetch-watchdog           Disable fetch-server monitoring
  --dry-run                     Let the evaluator print its plan without running
  --prepare-only                Validate launcher inputs without invoking evaluator
  -h, --help                    Show this help

Environment variables with the same uppercase names remain supported. For example:
  MODEL=Qwen3.5-9B HERMES_BASE_URLS=http://127.0.0.1:8000/v1 \
    DATASET=browsecomp_subset_200 DATA_PATH=data/browsecomp_subset_200/questions_200.jsonl \
    scripts/run_browsecomp.sh

Example:
  scripts/run_browsecomp.sh \
    --model SenseNova-Flash-Lite \
    --base-urls http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1 \
    --workers-per-endpoint 10 \
    --dataset browsecomp_subset_200 \
    --data-path data/browsecomp_subset_200/questions_200.jsonl \
    --expected-questions 200 --repeats 3

SEAL-0:
  scripts/run_browsecomp.sh \
    --model MODEL --base-url http://127.0.0.1:8000/v1 \
    --save-name MODEL_seal_0 \
    --dataset seal_0 --data-path data/seal_0/questions.jsonl \
    --expected-questions 111

WideSearch:
  scripts/run_browsecomp.sh \
    --model MODEL --base-url http://127.0.0.1:8000/v1 \
    --save-name MODEL_widesearch \
    --dataset widesearch --data-path data/widesearch/questions.jsonl \
    --expected-questions 200 --repeats 4
EOF
}

die() {
  echo "Error: $*" >&2
  exit 2
}

need_value() {
  [[ "$#" -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

MODEL="${MODEL:-}"
MODELS="${MODELS:-}"
KEY_BASE_URLS="${KEY_BASE_URLS:-}"
API_MODES="${API_MODES:-}"
PROVIDER="${PROVIDER:-custom}"
SAVE_NAME="${SAVE_NAME:-}"
DATASET="${DATASET:-}"
DATA_PATH="${DATA_PATH:-}"
EXPECTED_QUESTIONS="${EXPECTED_QUESTIONS:-}"
REPEATS="${REPEATS:-1}"
WORKERS_PER_ENDPOINT="${WORKERS_PER_ENDPOINT:-10}"
WORKERS_PER_KEY="${WORKERS_PER_KEY:-}"
MAX_ROUNDS="${MAX_ROUNDS:-500}"
MAX_RETRIES="${MAX_RETRIES:-3}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
CONTEXT_COMPRESSION="${CONTEXT_COMPRESSION:-true}"
COMPRESSION_THRESHOLD="${COMPRESSION_THRESHOLD:-0.5}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
MAX_TOKENS="${MAX_TOKENS:-}"
TEMPERATURE="${TEMPERATURE:-}"
ENDPOINT_WATCHDOG="${ENDPOINT_WATCHDOG:-1}"
ENDPOINT_WATCHDOG_INTERVAL="${ENDPOINT_WATCHDOG_INTERVAL:-30}"
ENDPOINT_WATCHDOG_TIMEOUT="${ENDPOINT_WATCHDOG_TIMEOUT:-5}"
ENDPOINT_WATCHDOG_FAILURES="${ENDPOINT_WATCHDOG_FAILURES:-3}"
ENDPOINT_WATCHDOG_RECOVERY_SUCCESSES="${ENDPOINT_WATCHDOG_RECOVERY_SUCCESSES:-2}"
ALL_ENDPOINTS_DOWN_TIMEOUT="${ALL_ENDPOINTS_DOWN_TIMEOUT:-300}"
ENDPOINT_REGISTRY="${ENDPOINT_REGISTRY:-}"
ENDPOINT_REGISTRY_REFRESH_INTERVAL="${ENDPOINT_REGISTRY_REFRESH_INTERVAL:-30}"
HERMES_QUIET="${HERMES_QUIET:-1}"
ANTIHACK_ENABLED="${ANTIHACK_ENABLED:-1}"
FORCE="${FORCE:-0}"
RERUN_FAILED="${RERUN_FAILED:-0}"
NUMBERED_OUTPUT="${NUMBERED_OUTPUT:-1}"
ENABLE_BROWSER="${ENABLE_BROWSER:-0}"
SKILL_WHITELIST="${SKILL_WHITELIST:-}"
ENABLE_CUSTOM_SYSTEM_PROMPT="${ENABLE_CUSTOM_SYSTEM_PROMPT:-false}"
CUSTOM_SYSTEM_PROMPT_FILE="${CUSTOM_SYSTEM_PROMPT_FILE:-}"
ENABLE_SUBAGENT_CUSTOM_SYSTEM_PROMPT="${ENABLE_SUBAGENT_CUSTOM_SYSTEM_PROMPT:-false}"
SUBAGENT_CUSTOM_SYSTEM_PROMPT_FILE="${SUBAGENT_CUSTOM_SYSTEM_PROMPT_FILE:-}"
WORKSPACE_NAMESPACE="${WORKSPACE_NAMESPACE:-}"
API_KEY_ENV="${API_KEY_ENV:-HERMES_API_KEY}"
API_KEY_ENVS="${API_KEY_ENVS:-}"
HERMES_BIN="${HERMES_BIN:-${HERMES_REPO}/.venv/bin/hermes}"
EVAL_PYTHON="${EVAL_PYTHON:-${EVAL_ROOT}/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/preds}"
RESUME_FROM="${RESUME_FROM:-}"
LOG_FILE="${LOG_FILE:-}"
AUTO_LOG="${AUTO_LOG:-1}"
BROWSER_CDP_URL="${BROWSER_CDP_URL:-http://127.0.0.1:9222}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-}"
FETCH_WATCHDOG="${FETCH_WATCHDOG:-auto}"
FETCH_SERVER_BASE_URL="${FETCH_SERVER_BASE_URL:-}"
FETCH_WATCHDOG_URL="${FETCH_WATCHDOG_URL:-}"
FETCH_WATCHDOG_INTERVAL="${FETCH_WATCHDOG_INTERVAL:-300}"
FETCH_WATCHDOG_TIMEOUT="${FETCH_WATCHDOG_TIMEOUT:-30}"
FETCH_WATCHDOG_FAILURES="${FETCH_WATCHDOG_FAILURES:-2}"
FETCH_WATCHDOG_HEALTH_INTERVAL="${FETCH_WATCHDOG_HEALTH_INTERVAL:-10}"
FETCH_WATCHDOG_HEALTH_TIMEOUT="${FETCH_WATCHDOG_HEALTH_TIMEOUT:-3}"
FETCH_WATCHDOG_HEALTH_FAILURES="${FETCH_WATCHDOG_HEALTH_FAILURES:-3}"
FETCH_WATCHDOG_DEFAULT_TARGETS="https://example.com,https://www.baidu.com,https://en.wikipedia.org/wiki/Main_Page"
FETCH_WATCHDOG_TARGET_URL="${FETCH_WATCHDOG_TARGET_URL-${FETCH_WATCHDOG_DEFAULT_TARGETS}}"
FETCH_WATCHDOG_STOP_GRACE="${FETCH_WATCHDOG_STOP_GRACE:-10}"
FETCH_WATCHDOG_HELPER="${FETCH_WATCHDOG_HELPER:-${SCRIPT_DIR}/fetch_watchdog.py}"

endpoint_args=()
endpoint_cli_seen=0
API_KEY_VALUE=""
if [[ -n "${HERMES_BASE_URLS:-}" ]]; then
  endpoint_args+=("${HERMES_BASE_URLS}")
elif [[ -n "${HERMES_BASE_URL:-}" ]]; then
  endpoint_args+=("${HERMES_BASE_URL}")
fi

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --model|--models|--provider|--save-name|--dataset|--data-path|--expected-questions)
      need_value "$@"
      name="${1#--}"
      name="${name//-/_}"
      printf -v "${name^^}" '%s' "$2"
      shift 2
      ;;
    --workers-per-endpoint|--workers-per-key|--repeats|--max-rounds|--max-retries|--timeout-seconds|--context-length|--compression-threshold|--max-tokens|--temperature|--endpoint-watchdog-interval|--endpoint-watchdog-timeout|--endpoint-watchdog-failures|--endpoint-watchdog-recovery-successes|--all-endpoints-down-timeout|--endpoint-registry-refresh-interval)
      need_value "$@"
      name="${1#--}"
      name="${name//-/_}"
      printf -v "${name^^}" '%s' "$2"
      shift 2
      ;;
    --reasoning-effort)
      need_value "$@"; REASONING_EFFORT="$2"; shift 2 ;;
    --context-compression)
      need_value "$@"; CONTEXT_COMPRESSION="$2"; shift 2 ;;
    --tool-whitelist)
      need_value "$@"; TOOL_WHITELIST="$2"; shift 2 ;;
    --skill-whitelist)
      need_value "$@"; SKILL_WHITELIST="$2"; shift 2 ;;
    --enable-custom-system-prompt)
      need_value "$@"; ENABLE_CUSTOM_SYSTEM_PROMPT="$2"; shift 2 ;;
    --custom-system-prompt-file)
      need_value "$@"; CUSTOM_SYSTEM_PROMPT_FILE="$2"; shift 2 ;;
    --enable-subagent-custom-system-prompt)
      need_value "$@"; ENABLE_SUBAGENT_CUSTOM_SYSTEM_PROMPT="$2"; shift 2 ;;
    --subagent-custom-system-prompt-file)
      need_value "$@"; SUBAGENT_CUSTOM_SYSTEM_PROMPT_FILE="$2"; shift 2 ;;
    --workspace-namespace)
      need_value "$@"; WORKSPACE_NAMESPACE="$2"; shift 2 ;;
    --api-key-env)
      need_value "$@"; API_KEY_ENV="$2"; shift 2 ;;
    --api-key-envs)
      need_value "$@"; API_KEY_ENVS="$2"; shift 2 ;;
    --key-base-urls)
      need_value "$@"; KEY_BASE_URLS="$2"; endpoint_args=(); endpoint_cli_seen=1; shift 2 ;;
    --api-modes)
      need_value "$@"; API_MODES="$2"; shift 2 ;;
    --api-key)
      need_value "$@"; API_KEY_VALUE="$2"; shift 2 ;;
    --endpoint-registry)
      need_value "$@"; ENDPOINT_REGISTRY="$2"; shift 2 ;;
    --hermes-bin)
      need_value "$@"; HERMES_BIN="$2"; shift 2 ;;
    --eval-python)
      need_value "$@"; EVAL_PYTHON="$2"; shift 2 ;;
    --output-root)
      need_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --resume-from)
      need_value "$@"; RESUME_FROM="$2"; shift 2 ;;
    --log-file)
      need_value "$@"; LOG_FILE="$2"; AUTO_LOG=1; shift 2 ;;
    --browser-cdp-url)
      need_value "$@"; BROWSER_CDP_URL="$2"; shift 2 ;;
    --fetch-server-url)
      need_value "$@"
      FETCH_SERVER_BASE_URL="${2%/}"
      FETCH_WATCHDOG_URL="${2%/}"
      FETCH_WATCHDOG=1
      shift 2 ;;
    --fetch-watchdog-url)
      need_value "$@"; FETCH_WATCHDOG_URL="${2%/}"; FETCH_WATCHDOG=1; shift 2 ;;
    --fetch-watchdog-interval)
      need_value "$@"; FETCH_WATCHDOG_INTERVAL="$2"; shift 2 ;;
    --fetch-watchdog-timeout)
      need_value "$@"; FETCH_WATCHDOG_TIMEOUT="$2"; shift 2 ;;
    --fetch-watchdog-failures)
      need_value "$@"; FETCH_WATCHDOG_FAILURES="$2"; shift 2 ;;
    --fetch-watchdog-health-interval)
      need_value "$@"; FETCH_WATCHDOG_HEALTH_INTERVAL="$2"; shift 2 ;;
    --fetch-watchdog-health-timeout)
      need_value "$@"; FETCH_WATCHDOG_HEALTH_TIMEOUT="$2"; shift 2 ;;
    --fetch-watchdog-health-failures)
      need_value "$@"; FETCH_WATCHDOG_HEALTH_FAILURES="$2"; shift 2 ;;
    --fetch-watchdog-target)
      need_value "$@"; FETCH_WATCHDOG_TARGET_URL="$2"; shift 2 ;;
    --limit)
      need_value "$@"; LIMIT="$2"; shift 2 ;;
    --base-url)
      need_value "$@"
      if [[ "${endpoint_cli_seen}" == "0" ]]; then endpoint_args=(); endpoint_cli_seen=1; fi
      endpoint_args+=("$2"); shift 2 ;;
    --base-urls)
      need_value "$@"
      if [[ "${endpoint_cli_seen}" == "0" ]]; then endpoint_args=(); endpoint_cli_seen=1; fi
      endpoint_args+=("$2"); shift 2 ;;
    --enable-browser)
      ENABLE_BROWSER=1; shift ;;
    --force)
      FORCE=1; shift ;;
    --rerun-failed)
      RERUN_FAILED=1; shift ;;
    --numbered-output)
      NUMBERED_OUTPUT=1; shift ;;
    --no-antihack)
      ANTIHACK_ENABLED=0; shift ;;
    --no-quiet)
      HERMES_QUIET=0; shift ;;
    --no-log)
      AUTO_LOG=0; shift ;;
    --no-fetch-watchdog)
      FETCH_WATCHDOG=0; shift ;;
    --endpoint-watchdog)
      ENDPOINT_WATCHDOG=1; shift ;;
    --no-endpoint-watchdog)
      ENDPOINT_WATCHDOG=0; shift ;;
    --fetch-watchdog-health-only)
      FETCH_WATCHDOG_TARGET_URL=""; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --prepare-only)
      PREPARE_ONLY=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      [[ "$#" -eq 0 ]] || die "unexpected positional arguments: $*"
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
done

[[ -n "${MODEL}" || -n "${MODELS}" ]] || die "--model or --models is required"
[[ -n "${DATASET}" ]] || die "--dataset is required"
[[ -n "${DATA_PATH}" ]] || die "--data-path is required"
[[ "${API_KEY_ENV}" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]] || die "invalid --api-key-env: ${API_KEY_ENV}"

for pair in \
  "workers-per-endpoint:${WORKERS_PER_ENDPOINT}" "repeats:${REPEATS}" \
  "max-rounds:${MAX_ROUNDS}" "max-retries:${MAX_RETRIES}" \
  "timeout-seconds:${TIMEOUT_SECONDS}" "context-length:${CONTEXT_LENGTH}" \
  "endpoint-watchdog-failures:${ENDPOINT_WATCHDOG_FAILURES}" \
  "endpoint-watchdog-recovery-successes:${ENDPOINT_WATCHDOG_RECOVERY_SUCCESSES}" \
  "fetch-watchdog-interval:${FETCH_WATCHDOG_INTERVAL}" \
  "fetch-watchdog-timeout:${FETCH_WATCHDOG_TIMEOUT}" \
  "fetch-watchdog-failures:${FETCH_WATCHDOG_FAILURES}" \
  "fetch-watchdog-health-interval:${FETCH_WATCHDOG_HEALTH_INTERVAL}" \
  "fetch-watchdog-health-timeout:${FETCH_WATCHDOG_HEALTH_TIMEOUT}" \
  "fetch-watchdog-health-failures:${FETCH_WATCHDOG_HEALTH_FAILURES}" \
  "fetch-watchdog-stop-grace:${FETCH_WATCHDOG_STOP_GRACE}"; do
  label="${pair%%:*}"
  value="${pair#*:}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "--${label} must be a positive integer"
done
if [[ -n "${MAX_TOKENS}" ]]; then
  [[ "${MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]] || die "--max-tokens must be a positive integer"
fi
if [[ -n "${TEMPERATURE}" ]]; then
  [[ "${TEMPERATURE}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || die "--temperature must be a non-negative number"
fi
[[ "${FETCH_WATCHDOG}" =~ ^(auto|0|1)$ ]] \
  || die "FETCH_WATCHDOG must be auto, 0, or 1"
[[ "${ENDPOINT_WATCHDOG}" =~ ^(0|1)$ ]] \
  || die "ENDPOINT_WATCHDOG must be 0 or 1"
for pair in \
  "endpoint-watchdog-interval:${ENDPOINT_WATCHDOG_INTERVAL}" \
  "endpoint-watchdog-timeout:${ENDPOINT_WATCHDOG_TIMEOUT}" \
  "endpoint-registry-refresh-interval:${ENDPOINT_REGISTRY_REFRESH_INTERVAL}" \
  "all-endpoints-down-timeout:${ALL_ENDPOINTS_DOWN_TIMEOUT}"; do
  label="${pair%%:*}"
  value="${pair#*:}"
  [[ "${value}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ && "${value}" =~ [1-9] ]] \
    || die "--${label} must be a positive number"
done
if [[ "${ENDPOINT_WATCHDOG}" == "1" ]] && ! awk \
  -v timeout="${ENDPOINT_WATCHDOG_TIMEOUT}" \
  -v interval="${ENDPOINT_WATCHDOG_INTERVAL}" \
  'BEGIN { exit !(timeout < interval) }'; then
  die "--endpoint-watchdog-timeout must be less than --endpoint-watchdog-interval"
fi
[[ "${ANTIHACK_ENABLED}" =~ ^(0|1)$ ]] \
  || die "ANTIHACK_ENABLED must be 0 or 1"
if [[ -n "${FETCH_SERVER_BASE_URL}" ]]; then
  [[ "${FETCH_SERVER_BASE_URL}" =~ ^https?:// ]] \
    || die "invalid fetch server URL: ${FETCH_SERVER_BASE_URL}"
  export FETCH_SERVER_BASE_URL
fi
if [[ -n "${WORKERS_PER_KEY}" && ! "${WORKERS_PER_KEY}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
  die "--workers-per-key must be a positive integer or comma-separated positive integers"
fi
if [[ -n "${EXPECTED_QUESTIONS}" ]]; then
  [[ "${EXPECTED_QUESTIONS}" =~ ^[1-9][0-9]*$ ]] || die "--expected-questions must be a positive integer"
fi
if [[ -n "${LIMIT}" ]]; then
  [[ "${LIMIT}" =~ ^[1-9][0-9]*$ ]] || die "--limit must be a positive integer"
fi
if [[ -n "${REASONING_EFFORT}" && ! "${REASONING_EFFORT}" =~ ^(low|medium|high|max)$ ]]; then
  die "--reasoning-effort must be one of: low, medium, high, max"
fi
[[ "${ENABLE_CUSTOM_SYSTEM_PROMPT}" =~ ^(true|false)$ ]] \
  || die "--enable-custom-system-prompt must be true or false"
if [[ "${ENABLE_CUSTOM_SYSTEM_PROMPT}" == "true" ]]; then
  [[ -n "${CUSTOM_SYSTEM_PROMPT_FILE}" ]] \
    || die "--enable-custom-system-prompt true requires --custom-system-prompt-file"
else
  [[ -z "${CUSTOM_SYSTEM_PROMPT_FILE}" ]] \
    || die "--custom-system-prompt-file requires --enable-custom-system-prompt true"
fi
[[ "${ENABLE_SUBAGENT_CUSTOM_SYSTEM_PROMPT}" =~ ^(true|false)$ ]] \
  || die "--enable-subagent-custom-system-prompt must be true or false"
if [[ "${ENABLE_SUBAGENT_CUSTOM_SYSTEM_PROMPT}" == "true" ]]; then
  [[ -n "${SUBAGENT_CUSTOM_SYSTEM_PROMPT_FILE}" ]] \
    || die "--enable-subagent-custom-system-prompt true requires --subagent-custom-system-prompt-file"
else
  [[ -z "${SUBAGENT_CUSTOM_SYSTEM_PROMPT_FILE}" ]] \
    || die "--subagent-custom-system-prompt-file requires --enable-subagent-custom-system-prompt true"
fi
if [[ ! "${CONTEXT_COMPRESSION}" =~ ^(true|false)$ ]]; then
  die "--context-compression must be true or false"
fi
if [[ ! "${COMPRESSION_THRESHOLD}" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]]; then
  die "--compression-threshold must be greater than 0 and at most 1"
fi
if [[ -n "${MODELS}" ]]; then
  [[ -n "${API_KEY_ENVS}" ]] || die "--models requires --api-key-envs"
  IFS=',' read -r -a model_names <<<"${MODELS}"
  for index in "${!model_names[@]}"; do
    model_name="${model_names[$index]}"
    model_name="${model_name#"${model_name%%[![:space:]]*}"}"
    model_name="${model_name%"${model_name##*[![:space:]]}"}"
    [[ -n "${model_name}" ]] || die "--models contains an empty model name"
    model_names[$index]="${model_name}"
  done
  MODELS="$(IFS=,; echo "${model_names[*]}")"
  MODEL="${MODEL:-${model_names[0]}}"
fi

ENDPOINTS=()
if [[ -n "${ENDPOINT_REGISTRY}" ]]; then
  [[ "${#endpoint_args[@]}" -eq 0 && -z "${KEY_BASE_URLS}" ]] \
    || die "--endpoint-registry cannot be combined with --base-url, --base-urls, or --key-base-urls"
  [[ -z "${MODELS}" ]] || die "--endpoint-registry cannot be combined with --models"
  [[ -z "${API_KEY_ENVS}" ]] || die "--endpoint-registry cannot be combined with --api-key-envs"
  [[ -z "${WORKERS_PER_KEY}" ]] || die "--endpoint-registry cannot be combined with --workers-per-key"
  [[ "${ENDPOINT_WATCHDOG}" == "1" ]] \
    || die "--endpoint-registry requires endpoint watchdog"
  if [[ ! "${ENDPOINT_REGISTRY}" = /* ]]; then
    ENDPOINT_REGISTRY="${EVAL_ROOT}/${ENDPOINT_REGISTRY}"
  fi
  HERMES_BASE_URLS=""
  HERMES_BASE_URL=""
  export HERMES_BASE_URLS HERMES_BASE_URL
else
  raw_endpoints="$(IFS=,; echo "${endpoint_args[*]}")"
  IFS=',' read -r -a candidates <<<"${raw_endpoints}"
  for endpoint in "${candidates[@]}"; do
    endpoint="${endpoint#"${endpoint%%[![:space:]]*}"}"
    endpoint="${endpoint%"${endpoint##*[![:space:]]}"}"
    [[ -n "${endpoint}" ]] || continue
    [[ "${endpoint}" =~ ^https?:// ]] || die "invalid endpoint: ${endpoint}"
    seen=0
    for existing in "${ENDPOINTS[@]}"; do
      [[ "${endpoint}" == "${existing}" ]] && seen=1 && break
    done
    [[ "${seen}" == "1" ]] || ENDPOINTS+=("${endpoint}")
  done
  if [[ -n "${KEY_BASE_URLS}" ]]; then
    [[ "${#ENDPOINTS[@]}" -eq 0 ]] || die "--key-base-urls cannot be combined with --base-urls or --base-url"
    IFS=',' read -r -a ENDPOINTS <<<"${KEY_BASE_URLS}"
    for index in "${!ENDPOINTS[@]}"; do
      endpoint="${ENDPOINTS[$index]}"
      endpoint="${endpoint#"${endpoint%%[![:space:]]*}"}"
      endpoint="${endpoint%"${endpoint##*[![:space:]]}"}"
      [[ "${endpoint}" =~ ^https?:// ]] || die "invalid key base URL: ${endpoint}"
      ENDPOINTS[$index]="${endpoint}"
    done
    KEY_BASE_URLS="$(IFS=,; echo "${ENDPOINTS[*]}")"
    HERMES_BASE_URLS=""
  else
    [[ "${#ENDPOINTS[@]}" -gt 0 ]] || die "--base-urls, --base-url, --key-base-urls, or --endpoint-registry is required"
    HERMES_BASE_URLS="$(IFS=,; echo "${ENDPOINTS[*]}")"
  fi
  HERMES_BASE_URL="${ENDPOINTS[0]}"
  export HERMES_BASE_URLS HERMES_BASE_URL
fi

cd "${EVAL_ROOT}"
if [[ ! "${DATA_PATH}" = /* ]]; then
  DATA_PATH="${EVAL_ROOT}/${DATA_PATH}"
fi

if [[ "${AUTO_LOG}" == "1" ]]; then
  if [[ -z "${LOG_FILE}" ]]; then
    log_run_name="${SAVE_NAME:-${MODEL}}"
    log_run_name="${log_run_name//\//_}"
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

[[ -f "${DATA_PATH}" ]] || die "dataset not found: ${DATA_PATH}"
if [[ "${ENABLE_CUSTOM_SYSTEM_PROMPT}" == "true" ]]; then
  [[ -f "${CUSTOM_SYSTEM_PROMPT_FILE}" ]] \
    || die "custom System Prompt file not found: ${CUSTOM_SYSTEM_PROMPT_FILE}"
  [[ -r "${CUSTOM_SYSTEM_PROMPT_FILE}" ]] \
    || die "custom System Prompt file is not readable: ${CUSTOM_SYSTEM_PROMPT_FILE}"
fi
if [[ "${ENABLE_SUBAGENT_CUSTOM_SYSTEM_PROMPT}" == "true" ]]; then
  [[ -f "${SUBAGENT_CUSTOM_SYSTEM_PROMPT_FILE}" ]] \
    || die "subagent custom System Prompt file not found: ${SUBAGENT_CUSTOM_SYSTEM_PROMPT_FILE}"
  [[ -r "${SUBAGENT_CUSTOM_SYSTEM_PROMPT_FILE}" ]] \
    || die "subagent custom System Prompt file is not readable: ${SUBAGENT_CUSTOM_SYSTEM_PROMPT_FILE}"
fi
[[ -x "${HERMES_BIN}" ]] || die "Hermes executable not found or not executable: ${HERMES_BIN}"
[[ -x "${EVAL_PYTHON}" ]] || die "evaluation Python not found or not executable: ${EVAL_PYTHON}"

question_count="$(awk 'NF {count++} END {print count+0}' "${DATA_PATH}")"
if [[ -n "${EXPECTED_QUESTIONS}" && "${question_count}" != "${EXPECTED_QUESTIONS}" ]]; then
  die "expected ${EXPECTED_QUESTIONS} questions, found ${question_count}: ${DATA_PATH}"
fi

export HERMES_HOME="${HERMES_HOME:-${HERMES_REPO}/.hermes}"
export HERMES_SESSION_STORAGE_ROOT="${HERMES_SESSION_STORAGE_ROOT:-${HERMES_HOME}/sessions}"
if [[ -n "${API_KEY_VALUE}" && -n "${API_KEY_ENVS}" ]]; then
  die "--api-key cannot be combined with --api-key-envs"
elif [[ -n "${API_KEY_VALUE}" ]]; then
  printf -v "${API_KEY_ENV}" '%s' "${API_KEY_VALUE}"
  export "${API_KEY_ENV}"
elif [[ -n "${API_KEY_ENVS}" ]]; then
  IFS=',' read -r -a api_key_env_names <<<"${API_KEY_ENVS}"
  for api_key_env_name in "${api_key_env_names[@]}"; do
    api_key_env_name="${api_key_env_name#"${api_key_env_name%%[![:space:]]*}"}"
    api_key_env_name="${api_key_env_name%"${api_key_env_name##*[![:space:]]}"}"
    [[ "${api_key_env_name}" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]] \
      || die "invalid API key environment variable: ${api_key_env_name}"
    [[ -n "${!api_key_env_name:-}" ]] \
      || die "API key environment variable is not set: ${api_key_env_name}"
  done
elif [[ -z "${!API_KEY_ENV:-}" ]]; then
  die "API key environment variable is not set: ${API_KEY_ENV}"
fi
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ "${ENABLE_BROWSER}" == "1" ]]; then
  DEFAULT_TOOL_WHITELIST="browser,web,code_execution,delegation,vision"
  export BROWSER_CDP_URL
else
  DEFAULT_TOOL_WHITELIST="web,code_execution,delegation,vision"
  unset BROWSER_CDP_URL
fi
TOOL_WHITELIST="${TOOL_WHITELIST:-${DEFAULT_TOOL_WHITELIST}}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
MODEL_HOSTS=()
for endpoint in "${ENDPOINTS[@]}"; do
  model_host="${endpoint#*://}"
  model_host="${model_host%%/*}"
  model_host="${model_host%%:*}"
  MODEL_HOSTS+=("${model_host}")
done
MODEL_HOST_CSV="$(IFS=,; echo "${MODEL_HOSTS[*]}")"
if [[ -n "${MODEL_HOST_CSV}" ]]; then
  export NO_PROXY="${MODEL_HOST_CSV},localhost,127.0.0.1,::1"
else
  export NO_PROXY="localhost,127.0.0.1,::1"
fi
export no_proxy="${NO_PROXY}"

API_KEY_COUNT=1
if [[ -n "${API_KEY_ENVS}" ]]; then
  IFS=',' read -r -a api_key_env_names <<<"${API_KEY_ENVS}"
  API_KEY_COUNT="${#api_key_env_names[@]}"
fi
if [[ -n "${MODELS}" && "${#model_names[@]}" -ne "${API_KEY_COUNT}" ]]; then
  die "--models and --api-key-envs must contain the same number of values"
fi
if [[ -n "${KEY_BASE_URLS}" && "${#ENDPOINTS[@]}" -ne "${API_KEY_COUNT}" ]]; then
  die "--key-base-urls and --api-key-envs must contain the same number of values"
fi
if [[ -n "${API_MODES}" ]]; then
  IFS=',' read -r -a api_modes <<<"${API_MODES}"
  [[ "${#api_modes[@]}" -eq "${API_KEY_COUNT}" ]] \
    || die "--api-modes and --api-key-envs must contain the same number of values"
  for api_mode in "${api_modes[@]}"; do
    [[ "${api_mode}" =~ ^(chat_completions|codex_responses)$ ]] \
      || die "--api-modes values must be chat_completions or codex_responses"
  done
fi
if [[ -n "${WORKERS_PER_KEY}" ]]; then
  IFS=',' read -r -a worker_limits <<<"${WORKERS_PER_KEY}"
  if [[ "${#worker_limits[@]}" -eq 1 ]]; then
    worker_limit="${worker_limits[0]}"
    worker_limits=()
    for ((index = 0; index < API_KEY_COUNT; index++)); do
      worker_limits+=("${worker_limit}")
    done
  elif [[ "${#worker_limits[@]}" -ne "${API_KEY_COUNT}" ]]; then
    die "--workers-per-key must contain one value or one value per API key"
  fi
else
  worker_limits=()
  for ((index = 0; index < API_KEY_COUNT; index++)); do
    worker_limits+=("${WORKERS_PER_ENDPOINT}")
  done
fi
WORKER_LIMIT_SUM=0
for worker_limit in "${worker_limits[@]}"; do
  WORKER_LIMIT_SUM="$((WORKER_LIMIT_SUM + worker_limit))"
done
WORKER_LIMIT_CSV="$(IFS=,; echo "${worker_limits[*]}")"
if [[ -n "${ENDPOINT_REGISTRY}" ]]; then
  NUM_WORKERS="${WORKERS_PER_ENDPOINT}"
elif [[ -n "${KEY_BASE_URLS}" ]]; then
  NUM_WORKERS="${WORKER_LIMIT_SUM}"
else
  NUM_WORKERS="$(( ${#ENDPOINTS[@]} * WORKER_LIMIT_SUM ))"
fi
planned_questions="${question_count}"
if [[ -n "${LIMIT}" && "${LIMIT}" -lt "${planned_questions}" ]]; then
  planned_questions="${LIMIT}"
fi
echo "Dataset validation passed: ${question_count} questions"
echo "Model: ${MODEL}"
if [[ -n "${MODELS}" ]]; then
  echo "Model/key routes:"
  for index in "${!model_names[@]}"; do
    route_details="model=${model_names[$index]} api_key_env=${api_key_env_names[$index]} workers=${worker_limits[$index]}"
    if [[ -n "${KEY_BASE_URLS}" ]]; then
      route_details+=" base_url=${ENDPOINTS[$index]}"
    fi
    if [[ -n "${API_MODES}" ]]; then
      route_details+=" api_mode=${api_modes[$index]}"
    fi
    echo "  ${route_details}"
  done
fi
if [[ -n "${ENDPOINT_REGISTRY}" ]]; then
  echo "Endpoint registry: ${ENDPOINT_REGISTRY} (refresh=${ENDPOINT_REGISTRY_REFRESH_INTERVAL}s)"
  echo "Model endpoints: dynamic; workers per endpoint=${WORKERS_PER_ENDPOINT}"
else
  echo "Model endpoints (${#ENDPOINTS[@]}):"
  printf '  %s\n' "${ENDPOINTS[@]}"
fi
echo "Dataset: ${DATASET} (${DATA_PATH})"
echo "Workspace namespace: ${WORKSPACE_NAMESPACE:-<none>}"
echo "Output name: ${SAVE_NAME:-<derived from model>}"
echo "Runs: ${planned_questions} questions x ${REPEATS} repeats"
echo "Workers=${NUM_WORKERS}"
if [[ -n "${ENDPOINT_REGISTRY}" ]]; then
  LANE_COUNT="dynamic"
elif [[ -n "${KEY_BASE_URLS}" ]]; then
  LANE_COUNT="${API_KEY_COUNT}"
else
  LANE_COUNT="$(( ${#ENDPOINTS[@]} * API_KEY_COUNT ))"
fi
echo "API keys=${API_KEY_COUNT}; endpoint/key lanes=${LANE_COUNT}; workers per key=${WORKER_LIMIT_CSV}"
echo "Rounds=${MAX_ROUNDS} retries=${MAX_RETRIES}"
echo "Rerun failed: ${RERUN_FAILED}"
echo "Timeout=${TIMEOUT_SECONDS}s context=${CONTEXT_LENGTH}"
echo "Max tokens=${MAX_TOKENS:-<model default>} temperature=${TEMPERATURE:-<model default>}"
echo "Context compression=${CONTEXT_COMPRESSION} threshold=${COMPRESSION_THRESHOLD}"
echo "Reasoning effort=${REASONING_EFFORT:-<API default>}"
echo "Endpoint watchdog: ${ENDPOINT_WATCHDOG} (interval=${ENDPOINT_WATCHDOG_INTERVAL}s timeout=${ENDPOINT_WATCHDOG_TIMEOUT}s failures=${ENDPOINT_WATCHDOG_FAILURES} recovery_successes=${ENDPOINT_WATCHDOG_RECOVERY_SUCCESSES} all_down=${ALL_ENDPOINTS_DOWN_TIMEOUT}s)"
echo "Tool whitelist: ${TOOL_WHITELIST}"
echo "Numbered output: ${NUMBERED_OUTPUT}"
if [[ "${ANTIHACK_ENABLED}" == "1" ]]; then
  echo "Antihack: enabled"
else
  echo "Antihack: disabled"
fi

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "Preparation complete; evaluation not started."
  exit 0
fi

optional_args=()
[[ -n "${SAVE_NAME}" ]] && optional_args+=(--save-name "${SAVE_NAME}")
[[ -n "${RESUME_FROM}" ]] && optional_args+=(--resume-from "${RESUME_FROM}")
[[ "${HERMES_QUIET}" == "0" ]] && optional_args+=(--no-quiet)
[[ "${FORCE}" == "1" ]] && optional_args+=(--force)
[[ "${RERUN_FAILED}" == "1" ]] && optional_args+=(--rerun-failed)
[[ "${NUMBERED_OUTPUT}" == "1" ]] && optional_args+=(--numbered-output)
[[ "${ANTIHACK_ENABLED}" == "0" ]] && optional_args+=(--no-antihack)
[[ -n "${LIMIT}" ]] && optional_args+=(--limit "${LIMIT}")
[[ "${DRY_RUN}" == "1" ]] && optional_args+=(--dry-run)
[[ -n "${API_KEY_ENVS}" ]] && optional_args+=(--api-key-envs "${API_KEY_ENVS}")
[[ -n "${MODELS}" ]] && optional_args+=(--models "${MODELS}")
[[ -n "${KEY_BASE_URLS}" ]] && optional_args+=(--key-base-urls "${KEY_BASE_URLS}")
[[ -n "${API_MODES}" ]] && optional_args+=(--api-modes "${API_MODES}")
[[ -n "${HERMES_BASE_URLS}" ]] && optional_args+=(--base-urls "${HERMES_BASE_URLS}")
[[ -n "${ENDPOINT_REGISTRY}" ]] && optional_args+=(--endpoint-registry "${ENDPOINT_REGISTRY}")
optional_args+=(--endpoint-registry-refresh-interval "${ENDPOINT_REGISTRY_REFRESH_INTERVAL}")
[[ -n "${WORKERS_PER_KEY}" ]] && optional_args+=(--workers-per-key "${WORKERS_PER_KEY}")
[[ -n "${REASONING_EFFORT}" ]] && optional_args+=(--reasoning-effort "${REASONING_EFFORT}")
[[ -n "${MAX_TOKENS}" ]] && optional_args+=(--max-tokens "${MAX_TOKENS}")
[[ -n "${TEMPERATURE}" ]] && optional_args+=(--temperature "${TEMPERATURE}")
[[ -n "${WORKSPACE_NAMESPACE}" ]] && optional_args+=(--workspace-namespace "${WORKSPACE_NAMESPACE}")
optional_args+=(--enable-custom-system-prompt "${ENABLE_CUSTOM_SYSTEM_PROMPT}")
[[ -n "${CUSTOM_SYSTEM_PROMPT_FILE}" ]] && optional_args+=(--custom-system-prompt-file "${CUSTOM_SYSTEM_PROMPT_FILE}")
optional_args+=(--enable-subagent-custom-system-prompt "${ENABLE_SUBAGENT_CUSTOM_SYSTEM_PROMPT}")
[[ -n "${SUBAGENT_CUSTOM_SYSTEM_PROMPT_FILE}" ]] && optional_args+=(--subagent-custom-system-prompt-file "${SUBAGENT_CUSTOM_SYSTEM_PROMPT_FILE}")
optional_args+=(--context-compression "${CONTEXT_COMPRESSION}")
optional_args+=(--compression-threshold "${COMPRESSION_THRESHOLD}")
if [[ "${ENDPOINT_WATCHDOG}" == "1" ]]; then
  optional_args+=(--endpoint-watchdog)
else
  optional_args+=(--no-endpoint-watchdog)
fi
optional_args+=(
  --endpoint-watchdog-interval "${ENDPOINT_WATCHDOG_INTERVAL}"
  --endpoint-watchdog-timeout "${ENDPOINT_WATCHDOG_TIMEOUT}"
  --endpoint-watchdog-failures "${ENDPOINT_WATCHDOG_FAILURES}"
  --endpoint-watchdog-recovery-successes "${ENDPOINT_WATCHDOG_RECOVERY_SUCCESSES}"
  --all-endpoints-down-timeout "${ALL_ENDPOINTS_DOWN_TIMEOUT}"
)

eval_cmd=("${EVAL_PYTHON}" -m browse_comp_eval.run
  --dataset "${DATASET}" \
  --data-path "${DATA_PATH}" \
  --model "${MODEL}" \
  --provider "${PROVIDER}" \
  --workers-per-endpoint "${WORKERS_PER_ENDPOINT}" \
  --api-key-env "${API_KEY_ENV}" \
  --hermes-bin "${HERMES_BIN}" \
  --repeats "${REPEATS}" \
  --num-workers "${NUM_WORKERS}" \
  --max-rounds "${MAX_ROUNDS}" \
  --max-retries "${MAX_RETRIES}" \
  --timeout-seconds "${TIMEOUT_SECONDS}" \
  --context-length "${CONTEXT_LENGTH}" \
  --tool-whitelist "${TOOL_WHITELIST}" \
  --skill-whitelist "${SKILL_WHITELIST}" \
  --output-root "${OUTPUT_ROOT}" \
  --ignore-rules \
  "${optional_args[@]}")

if [[ "${FETCH_WATCHDOG}" != "0" ]]; then
  [[ -f "${FETCH_WATCHDOG_HELPER}" ]] \
    || die "fetch watchdog helper not found: ${FETCH_WATCHDOG_HELPER}"
  HERMES_FETCH_SERVER_URL="$("${EVAL_PYTHON}" "${FETCH_WATCHDOG_HELPER}" resolve-url)"
  if [[ -z "${FETCH_WATCHDOG_URL}" ]]; then
    FETCH_WATCHDOG_URL="${HERMES_FETCH_SERVER_URL}"
  fi
  FETCH_WATCHDOG_URL="${FETCH_WATCHDOG_URL%/}"
  if [[ -z "${HERMES_FETCH_SERVER_URL}" ]]; then
    if [[ "${FETCH_WATCHDOG}" == "1" ]]; then
      die "fetch watchdog enabled but Hermes has no fetch server URL configured; use --fetch-server-url"
    fi
    FETCH_WATCHDOG=0
    echo "Fetch watchdog: disabled (Hermes has no fetch server URL configured)"
  elif [[ "${FETCH_WATCHDOG_URL}" != "${HERMES_FETCH_SERVER_URL}" ]]; then
    die "fetch watchdog URL (${FETCH_WATCHDOG_URL}) does not match Hermes fetch server (${HERMES_FETCH_SERVER_URL}); use --fetch-server-url"
  else
    FETCH_WATCHDOG=1
  fi
fi

if [[ "${FETCH_WATCHDOG}" == "0" ]]; then
  exec "${eval_cmd[@]}"
fi

echo "Fetch watchdog: enabled"
echo "  server=${FETCH_WATCHDOG_URL}"
echo "  health: interval=${FETCH_WATCHDOG_HEALTH_INTERVAL}s timeout=${FETCH_WATCHDOG_HEALTH_TIMEOUT}s consecutive_failures=${FETCH_WATCHDOG_HEALTH_FAILURES}"
echo "  functional: interval=${FETCH_WATCHDOG_INTERVAL}s timeout=${FETCH_WATCHDOG_TIMEOUT}s consecutive_failures=${FETCH_WATCHDOG_FAILURES}"
echo "  functional_targets=${FETCH_WATCHDOG_TARGET_URL:-<health-only>}"

if ! "${EVAL_PYTHON}" "${FETCH_WATCHDOG_HELPER}" probe \
  --base-url "${FETCH_WATCHDOG_URL}" \
  --timeout "${FETCH_WATCHDOG_TIMEOUT}" \
  --target-url "${FETCH_WATCHDOG_TARGET_URL}"; then
  die "fetch watchdog preflight failed: ${FETCH_WATCHDOG_URL}"
fi
echo "Fetch watchdog preflight passed."

# Run the evaluator and all Hermes descendants in an isolated process group so
# a watchdog trip cannot leave orphaned workers behind.
setsid "${eval_cmd[@]}" &
eval_pid=$!
watchdog_run_ids="$(mktemp "${TMPDIR:-/tmp}/browsecomp-fetch-watchdog.XXXXXX")"
watchdog_signal=""

terminate_eval_group() {
  local signal_name="$1"
  kill "-${signal_name}" -- "-${eval_pid}" 2>/dev/null || true
}

handle_watchdog_signal() {
  watchdog_signal="$1"
  echo "Received ${watchdog_signal}; terminating evaluator process group ${eval_pid}." >&2
  terminate_eval_group TERM
}
trap 'handle_watchdog_signal INT' INT
trap 'handle_watchdog_signal TERM' TERM
trap 'handle_watchdog_signal HUP' HUP
trap 'rm -f "${watchdog_run_ids}"' EXIT

health_failures=0
functional_failures=0
last_probe_error=""
last_health_probe="${SECONDS}"
last_functional_probe="${SECONDS}"
while kill -0 "${eval_pid}" 2>/dev/null; do
  sleep 1 &
  wait $! || true
  [[ -z "${watchdog_signal}" ]] || break
  kill -0 "${eval_pid}" 2>/dev/null || break

  watchdog_tripped=0
  if ((SECONDS - last_health_probe >= FETCH_WATCHDOG_HEALTH_INTERVAL)); then
    last_health_probe="${SECONDS}"
    probe_output="$(
      "${EVAL_PYTHON}" "${FETCH_WATCHDOG_HELPER}" probe \
        --base-url "${FETCH_WATCHDOG_URL}" \
        --timeout "${FETCH_WATCHDOG_HEALTH_TIMEOUT}" 2>&1
    )" && probe_ok=1 || probe_ok=0
    if [[ "${probe_ok}" == "1" ]]; then
      if [[ "${health_failures}" -gt 0 ]]; then
        echo "Fetch watchdog /health recovered after ${health_failures} failed probe(s)."
      fi
      health_failures=0
    else
      health_failures="$((health_failures + 1))"
      echo "Fetch watchdog /health failed (${health_failures}/${FETCH_WATCHDOG_HEALTH_FAILURES}): ${probe_output:-probe failed without an error message}" >&2
      if [[ "${health_failures}" -ge "${FETCH_WATCHDOG_HEALTH_FAILURES}" ]]; then
        last_probe_error="/health: ${probe_output:-probe failed without an error message}"
        watchdog_tripped=1
      fi
    fi
  fi

  if [[ "${watchdog_tripped}" == "0" && -n "${FETCH_WATCHDOG_TARGET_URL}" ]] \
    && ((SECONDS - last_functional_probe >= FETCH_WATCHDOG_INTERVAL)); then
    last_functional_probe="${SECONDS}"
    probe_output="$(
      "${EVAL_PYTHON}" "${FETCH_WATCHDOG_HELPER}" probe \
        --base-url "${FETCH_WATCHDOG_URL}" \
        --timeout "${FETCH_WATCHDOG_TIMEOUT}" \
        --target-url "${FETCH_WATCHDOG_TARGET_URL}" 2>&1
    )" && probe_ok=1 || probe_ok=0
    if [[ "${probe_ok}" == "1" ]]; then
      if [[ "${functional_failures}" -gt 0 ]]; then
        echo "Fetch watchdog functional probe recovered after ${functional_failures} failed round(s)."
      fi
      functional_failures=0
    else
      functional_failures="$((functional_failures + 1))"
      echo "Fetch watchdog functional probe failed (${functional_failures}/${FETCH_WATCHDOG_FAILURES}): ${probe_output:-probe failed without an error message}" >&2
      if [[ "${functional_failures}" -ge "${FETCH_WATCHDOG_FAILURES}" ]]; then
        last_probe_error="functional fetch: ${probe_output:-probe failed without an error message}"
        watchdog_tripped=1
      fi
    fi
  fi

  if [[ "${watchdog_tripped}" == "0" ]]; then
    continue
  fi

  echo "Fetch watchdog tripped; freezing evaluator process group ${eval_pid}." >&2
  terminate_eval_group STOP
  "${EVAL_PYTHON}" "${FETCH_WATCHDOG_HELPER}" active-run-ids \
    --process-group "${eval_pid}" >"${watchdog_run_ids}" || true
  active_count="$(awk 'NF {count++} END {print count+0}' "${watchdog_run_ids}")"
  echo "Fetch watchdog captured ${active_count} active run(s); terminating the evaluation." >&2

  terminate_eval_group TERM
  terminate_eval_group CONT
  for ((wait_index = 0; wait_index < FETCH_WATCHDOG_STOP_GRACE; wait_index++)); do
    kill -0 -- "-${eval_pid}" 2>/dev/null || break
    sleep 1
  done
  if kill -0 -- "-${eval_pid}" 2>/dev/null; then
    echo "Evaluator process group did not stop within ${FETCH_WATCHDOG_STOP_GRACE}s; sending SIGKILL." >&2
    terminate_eval_group KILL
  fi
  wait "${eval_pid}" 2>/dev/null || true

  mark_args=(
    --run-ids-file "${watchdog_run_ids}"
    --data-path "${DATA_PATH}"
    --output-root "${OUTPUT_ROOT}"
    --project-root "${EVAL_ROOT}"
    --save-name "${SAVE_NAME}"
    --model "${MODEL}"
    --dataset "${DATASET}"
    --fetch-server-url "${FETCH_WATCHDOG_URL}"
    --reason "${last_probe_error}"
  )
  [[ "${NUMBERED_OUTPUT}" == "1" ]] && mark_args+=(--numbered-output)
  "${EVAL_PYTHON}" "${FETCH_WATCHDOG_HELPER}" mark-failed "${mark_args[@]}"
  echo "Evaluation aborted because the fetch server was unhealthy. Rerun with --rerun-failed after recovery." >&2
  exit 75
done

if [[ -n "${watchdog_signal}" ]]; then
  wait "${eval_pid}" 2>/dev/null || true
  case "${watchdog_signal}" in
    INT) exit 130 ;;
    TERM) exit 143 ;;
    HUP) exit 129 ;;
  esac
fi

set +e
wait "${eval_pid}"
eval_status=$?
set -e
exit "${eval_status}"
