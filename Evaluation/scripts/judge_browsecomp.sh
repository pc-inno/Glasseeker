#!/usr/bin/env bash
# Generic launcher for judging BrowseComp prediction outputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HERMES_REPO="$(cd "${EVAL_ROOT}/../Evaluation_backend" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/judge_browsecomp.sh [options]

Prediction input (choose one form):
  --pred-root PATH              Prediction root containing conv/
  --save-name NAME             Evaluation save name (requires --dataset)
  --dataset NAME               Dataset name used with --save-name
  --output-root PATH            Prediction output root (default: output/preds)
  --resume-from PATH            Previous run root; reuse unchanged judge results

Judge API:
  --judge-model NAME            Judge model/deployment name (default: gpt-5-mini)
  --judge-template NAME         browsecomp, sealqa, gaia-text-103,
                                xbench-deepsearch-2510, or widesearch
                                (default: browsecomp)
  --prompt-version NAME         BrowseComp prompt: hermes or official
                                (default: official)
  --judge-base-url URL          OpenAI-compatible base URL or Azure endpoint
  --judge-provider NAME         openai or azure (default: openai)
  --judge-api-key-env NAME      Environment variable holding the API key
                                (default: JUDGE_API_KEY)
  --judge-api-key KEY           API key value (prefer --judge-api-key-env)
  --judge-api-version VERSION   Azure API version

Run controls:
  --num-workers N               Concurrent judge requests (default: 4)
  --max-retries N               Retries per request (default: 3)
  --timeout-seconds N           Request timeout (default: 60)
  --max-tokens N                Maximum judge output tokens (default: 2048)
  --temperature N               Sampling temperature (default: 1.0)
  --top-p N                     Nucleus sampling value (default: 1.0)
  --token-param NAME            max_completion_tokens or max_tokens
  --expected-predictions N      Require exactly N prediction JSON files
  --allow-partial               Allow a count different from the expected count
  --include-failed              Send failed inference outputs to the judge
  --no-json-schema              Disable response_format=json_schema
  --force                       Rejudge existing outputs
  --dry-run                     Print the judge plan without API requests

Output controls:
  --judge-output-dir PATH       Per-record judge output directory
                                (default depends on prompt version)
  --summary-path PATH           Judge summary JSONL path (versioned default)
  --metrics-path PATH           Judge metrics JSON path (versioned default)
  --eval-python PATH            Python used to run the evaluator
  --log-file PATH               Console log path
  --no-log                      Do not save console output
  -h, --help                    Show this help

All options also support their uppercase environment variable equivalents.

Example:
  export JUDGE_API_KEY="..."
  scripts/judge_browsecomp.sh \
    --save-name Qwen3.5-9B_browsecomp_200_r3 \
    --dataset browsecomp_subset_200 \
    --expected-predictions 600 \
    --judge-model gpt-5-mini \
    --judge-base-url https://example.com/v1 \
    --num-workers 10

BrowseComp with the official OpenAI grading prompt:
  scripts/judge_browsecomp.sh \
    --save-name MODEL_browsecomp --dataset browsecomp_full \
    --prompt-version official \
    --judge-model gpt-5-mini --judge-base-url https://example.com/v1

SEAL-0 (the published protocol uses gpt-4o-mini at temperature 0):
  scripts/judge_browsecomp.sh \
    --save-name MODEL_seal_0 --dataset seal_0 \
    --expected-predictions 111 --judge-template sealqa \
    --judge-model gpt-4o-mini --temperature 0 \
    --token-param max_tokens --judge-base-url https://api.openai.com/v1

WideSearch (defaults to the official GPT-4.1 judge configuration):
  scripts/judge_browsecomp.sh \
    --save-name MODEL_widesearch --dataset widesearch \
    --expected-predictions 800 --judge-template widesearch \
    --judge-base-url https://api.openai.com/v1
EOF
}

die() {
  echo "Error: $*" >&2
  exit 2
}

need_value() {
  [[ "$#" -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

PRED_ROOT="${PRED_ROOT:-}"
RESUME_FROM="${RESUME_FROM:-}"
SAVE_NAME="${SAVE_NAME:-}"
DATASET="${DATASET:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/preds}"
EXPECTED_PREDICTIONS="${EXPECTED_PREDICTIONS:-}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"

JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_TEMPLATE="${JUDGE_TEMPLATE:-browsecomp}"
JUDGE_PROMPT_VERSION="${JUDGE_PROMPT_VERSION:-official}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-JUDGE_API_KEY}"
JUDGE_API_KEY_VALUE=""
JUDGE_API_VERSION="${JUDGE_API_VERSION:-2024-12-01-preview}"
JUDGE_NUM_WORKERS="${JUDGE_NUM_WORKERS:-${NUM_WORKERS:-4}}"
JUDGE_MAX_RETRIES="${JUDGE_MAX_RETRIES:-${MAX_RETRIES:-3}}"
JUDGE_TIMEOUT_SECONDS="${JUDGE_TIMEOUT_SECONDS:-${TIMEOUT_SECONDS:-60}}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-${MAX_TOKENS:-}}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-${TEMPERATURE:-}}"
JUDGE_TOP_P="${JUDGE_TOP_P:-${TOP_P:-1.0}}"
JUDGE_TOKEN_PARAM="${JUDGE_TOKEN_PARAM:-${TOKEN_PARAM:-}}"

JUDGE_OUTPUT_DIR="${JUDGE_OUTPUT_DIR:-}"
SUMMARY_PATH="${SUMMARY_PATH:-}"
METRICS_PATH="${METRICS_PATH:-}"
EVAL_PYTHON="${EVAL_PYTHON:-${EVAL_ROOT}/.venv/bin/python}"
LOG_FILE="${LOG_FILE:-}"
AUTO_LOG="${AUTO_LOG:-1}"
FORCE="${FORCE:-0}"
INCLUDE_FAILED="${INCLUDE_FAILED:-0}"
NO_JSON_SCHEMA="${NO_JSON_SCHEMA:-0}"
DRY_RUN="${DRY_RUN:-0}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --pred-root) need_value "$@"; PRED_ROOT="$2"; shift 2 ;;
    --resume-from) need_value "$@"; RESUME_FROM="$2"; shift 2 ;;
    --save-name) need_value "$@"; SAVE_NAME="$2"; shift 2 ;;
    --dataset) need_value "$@"; DATASET="$2"; shift 2 ;;
    --output-root) need_value "$@"; OUTPUT_ROOT="$2"; shift 2 ;;
    --expected-predictions) need_value "$@"; EXPECTED_PREDICTIONS="$2"; shift 2 ;;
    --judge-provider) need_value "$@"; JUDGE_PROVIDER="$2"; shift 2 ;;
    --judge-model) need_value "$@"; JUDGE_MODEL="$2"; shift 2 ;;
    --judge-template) need_value "$@"; JUDGE_TEMPLATE="$2"; shift 2 ;;
    --prompt-version) need_value "$@"; JUDGE_PROMPT_VERSION="$2"; shift 2 ;;
    --judge-base-url) need_value "$@"; JUDGE_BASE_URL="$2"; shift 2 ;;
    --judge-api-key-env) need_value "$@"; JUDGE_API_KEY_ENV="$2"; shift 2 ;;
    --judge-api-key) need_value "$@"; JUDGE_API_KEY_VALUE="$2"; shift 2 ;;
    --judge-api-version) need_value "$@"; JUDGE_API_VERSION="$2"; shift 2 ;;
    --num-workers) need_value "$@"; JUDGE_NUM_WORKERS="$2"; shift 2 ;;
    --max-retries) need_value "$@"; JUDGE_MAX_RETRIES="$2"; shift 2 ;;
    --timeout-seconds) need_value "$@"; JUDGE_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --max-tokens) need_value "$@"; JUDGE_MAX_TOKENS="$2"; shift 2 ;;
    --temperature) need_value "$@"; JUDGE_TEMPERATURE="$2"; shift 2 ;;
    --top-p) need_value "$@"; JUDGE_TOP_P="$2"; shift 2 ;;
    --token-param) need_value "$@"; JUDGE_TOKEN_PARAM="$2"; shift 2 ;;
    --judge-output-dir) need_value "$@"; JUDGE_OUTPUT_DIR="$2"; shift 2 ;;
    --summary-path) need_value "$@"; SUMMARY_PATH="$2"; shift 2 ;;
    --metrics-path) need_value "$@"; METRICS_PATH="$2"; shift 2 ;;
    --eval-python) need_value "$@"; EVAL_PYTHON="$2"; shift 2 ;;
    --log-file) need_value "$@"; LOG_FILE="$2"; AUTO_LOG=1; shift 2 ;;
    --allow-partial) ALLOW_PARTIAL=1; shift ;;
    --include-failed) INCLUDE_FAILED=1; shift ;;
    --no-json-schema) NO_JSON_SCHEMA=1; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-log) AUTO_LOG=0; shift ;;
    -h|--help) usage; exit 0 ;;
    --)
      shift
      [[ "$#" -eq 0 ]] || die "unexpected positional arguments: $*"
      ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

if [[ "${JUDGE_TEMPLATE}" == "widesearch" ]]; then
  JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-2025-04-14}"
  JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-10240}"
  JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0}"
  JUDGE_TOKEN_PARAM="${JUDGE_TOKEN_PARAM:-max_tokens}"
else
  JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini}"
  JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-2048}"
  JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-1.0}"
  JUDGE_TOKEN_PARAM="${JUDGE_TOKEN_PARAM:-max_completion_tokens}"
fi

[[ "${JUDGE_PROVIDER}" == "openai" || "${JUDGE_PROVIDER}" == "azure" ]] \
  || die "--judge-provider must be openai or azure"
[[ "${JUDGE_TEMPLATE}" =~ ^(browsecomp|sealqa|gaia-text-103|xbench-deepsearch-2510|widesearch)$ ]] \
  || die "--judge-template must be browsecomp, sealqa, gaia-text-103, xbench-deepsearch-2510, or widesearch"
[[ "${JUDGE_PROMPT_VERSION}" =~ ^(hermes|official)$ ]] \
  || die "--prompt-version must be hermes or official"
if [[ "${JUDGE_TEMPLATE}" != "browsecomp" && "${JUDGE_PROMPT_VERSION}" != "hermes" ]]; then
  die "--prompt-version official requires --judge-template browsecomp"
fi
[[ "${JUDGE_API_KEY_ENV}" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]] \
  || die "invalid --judge-api-key-env: ${JUDGE_API_KEY_ENV}"
[[ "${JUDGE_TOKEN_PARAM}" == "max_completion_tokens" || "${JUDGE_TOKEN_PARAM}" == "max_tokens" ]] \
  || die "--token-param must be max_completion_tokens or max_tokens"

for pair in \
  "num-workers:${JUDGE_NUM_WORKERS}" "max-retries:${JUDGE_MAX_RETRIES}" \
  "max-tokens:${JUDGE_MAX_TOKENS}"; do
  label="${pair%%:*}"
  value="${pair#*:}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "--${label} must be a positive integer"
done
[[ "${JUDGE_TIMEOUT_SECONDS}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
  || die "--timeout-seconds must be a positive number"
[[ "${JUDGE_TIMEOUT_SECONDS}" != "0" && "${JUDGE_TIMEOUT_SECONDS}" != "0.0" ]] \
  || die "--timeout-seconds must be greater than zero"
if [[ -n "${EXPECTED_PREDICTIONS}" ]]; then
  [[ "${EXPECTED_PREDICTIONS}" =~ ^[1-9][0-9]*$ ]] \
    || die "--expected-predictions must be a positive integer"
fi

if [[ -z "${PRED_ROOT}" ]]; then
  [[ -n "${SAVE_NAME}" ]] || die "--pred-root or --save-name is required"
  [[ -n "${DATASET}" ]] || die "--dataset is required when using --save-name"
  PRED_ROOT="${OUTPUT_ROOT}/${SAVE_NAME}/${DATASET}"
elif [[ -n "${SAVE_NAME}" || -n "${DATASET}" ]]; then
  die "--pred-root cannot be combined with --save-name or --dataset"
fi

cd "${EVAL_ROOT}"
[[ "${PRED_ROOT}" = /* ]] || PRED_ROOT="${EVAL_ROOT}/${PRED_ROOT}"
[[ -d "${PRED_ROOT}/conv" ]] || die "prediction directory not found: ${PRED_ROOT}/conv"
[[ -x "${EVAL_PYTHON}" ]] || die "evaluation Python not found or not executable: ${EVAL_PYTHON}"

prediction_count="$(find "${PRED_ROOT}/conv" -type f -name '*.json' -print | wc -l | tr -d '[:space:]')"
if [[ -n "${EXPECTED_PREDICTIONS}" && "${ALLOW_PARTIAL}" != "1" \
      && "${prediction_count}" != "${EXPECTED_PREDICTIONS}" ]]; then
  die "expected ${EXPECTED_PREDICTIONS} predictions, found ${prediction_count}: ${PRED_ROOT}/conv (use --allow-partial to continue)"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -n "${JUDGE_BASE_URL}" ]] || die "--judge-base-url is required"
  if [[ -n "${JUDGE_API_KEY_VALUE}" ]]; then
    printf -v "${JUDGE_API_KEY_ENV}" '%s' "${JUDGE_API_KEY_VALUE}"
    export "${JUDGE_API_KEY_ENV}"
  else
    [[ -n "${!JUDGE_API_KEY_ENV:-}" ]] \
      || die "API key environment variable is not set: ${JUDGE_API_KEY_ENV}"
  fi
fi

if [[ -n "${JUDGE_BASE_URL}" ]]; then
  JUDGE_HOST="${JUDGE_BASE_URL#*://}"
  JUDGE_HOST="${JUDGE_HOST%%/*}"
  JUDGE_HOST="${JUDGE_HOST%%:*}"
  INHERITED_NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
  JUDGE_NO_PROXY="${JUDGE_HOST},localhost,127.0.0.1,::1"
  [[ -z "${INHERITED_NO_PROXY}" ]] || JUDGE_NO_PROXY="${JUDGE_NO_PROXY},${INHERITED_NO_PROXY}"
  export NO_PROXY="${JUDGE_NO_PROXY}"
  export no_proxy="${JUDGE_NO_PROXY}"
fi
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ "${AUTO_LOG}" == "1" ]]; then
  if [[ -z "${LOG_FILE}" ]]; then
    log_name="${SAVE_NAME:-$(basename "$(dirname "${PRED_ROOT}")")_$(basename "${PRED_ROOT}")}"
    log_name="$(printf '%s' "${log_name}" | tr -cs '[:alnum:]_.-' '_')"
    [[ "${JUDGE_PROMPT_VERSION}" != "official" ]] || log_name="${log_name}_official"
    LOG_FILE="${EVAL_ROOT}/logs/judge_${log_name}_$(date '+%Y%m%d_%H%M%S').log"
  elif [[ ! "${LOG_FILE}" = /* ]]; then
    LOG_FILE="${EVAL_ROOT}/${LOG_FILE}"
  fi
  mkdir -p "$(dirname "${LOG_FILE}")"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  echo "Console log: ${LOG_FILE}"
fi

echo "Hermes benchmark judge launcher"
echo "Prediction root: ${PRED_ROOT}"
echo "Predictions: ${prediction_count}${EXPECTED_PREDICTIONS:+ (expected ${EXPECTED_PREDICTIONS})}"
echo "Resume from: ${RESUME_FROM:-<none>}"
echo "Judge model: ${JUDGE_MODEL}"
echo "Judge template: ${JUDGE_TEMPLATE}"
echo "Judge prompt version: ${JUDGE_PROMPT_VERSION}"
echo "Judge provider: ${JUDGE_PROVIDER}"
echo "Judge endpoint: ${JUDGE_BASE_URL:-<not required for dry-run>}"
echo "Workers=${JUDGE_NUM_WORKERS} retries=${JUDGE_MAX_RETRIES} timeout=${JUDGE_TIMEOUT_SECONDS}s"

args=(
  "${EVAL_PYTHON}" -m browse_comp_eval.judge
  --pred-root "${PRED_ROOT}"
  --judge-provider "${JUDGE_PROVIDER}"
  --judge-model "${JUDGE_MODEL}"
  --judge-template "${JUDGE_TEMPLATE}"
  --prompt-version "${JUDGE_PROMPT_VERSION}"
  --judge-base-url "${JUDGE_BASE_URL}"
  --judge-api-key-env "${JUDGE_API_KEY_ENV}"
  --judge-api-version "${JUDGE_API_VERSION}"
  --num-workers "${JUDGE_NUM_WORKERS}"
  --max-retries "${JUDGE_MAX_RETRIES}"
  --timeout-seconds "${JUDGE_TIMEOUT_SECONDS}"
  --max-tokens "${JUDGE_MAX_TOKENS}"
  --temperature "${JUDGE_TEMPERATURE}"
  --top-p "${JUDGE_TOP_P}"
  --token-param "${JUDGE_TOKEN_PARAM}"
)
[[ -z "${JUDGE_OUTPUT_DIR}" ]] || args+=(--judge-output-dir "${JUDGE_OUTPUT_DIR}")
[[ -z "${SUMMARY_PATH}" ]] || args+=(--summary-path "${SUMMARY_PATH}")
[[ -z "${METRICS_PATH}" ]] || args+=(--metrics-path "${METRICS_PATH}")
[[ -z "${RESUME_FROM}" ]] || args+=(--resume-from "${RESUME_FROM}")
[[ "${INCLUDE_FAILED}" == "1" ]] && args+=(--include-failed)
[[ "${NO_JSON_SCHEMA}" == "1" ]] && args+=(--no-json-schema)
[[ "${FORCE}" == "1" ]] && args+=(--force)
[[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run)

exec "${args[@]}"
