from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
from pathlib import Path

from .dataset import expand_repeat, load_dataset, safe_id
from .endpoint_health import (
    EndpointWatchdogConfig,
    TEMPORARY_ENDPOINT_FAILURE_EXIT_CODE,
)
from .hermes_client import HermesClient, HermesConfig, MultiEndpointHermesClient
from .endpoint_registry import EndpointRegistryError, load_registry, model_entries
from .runner import BrowseCompRunner, RunnerConfig
from .whitelist import (
    DEFAULT_TOOL_WHITELIST,
    normalize_skill_whitelist,
    parse_csv,
    resolve_toolsets,
    validate_tool_whitelist,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or parsed != parsed or parsed in (float("inf"), float("-inf")):
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _explicit_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("must be true or false")


def load_custom_system_prompt(
    *,
    enabled: bool,
    file_path: str | None,
    project_root: Path,
    enable_option: str = "--enable-custom-system-prompt",
    file_option: str = "--custom-system-prompt-file",
) -> str | None:
    if enabled and not file_path:
        raise ValueError(f"{enable_option} true requires {file_option}")
    if not enabled and file_path:
        raise ValueError(f"{file_option} requires {enable_option} true")
    if not enabled:
        return None

    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    try:
        return path.read_bytes().decode("utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read custom System Prompt file {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"custom System Prompt file must be valid UTF-8: {path}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone parallel Hermes runner for Browse Comp evals")
    parser.add_argument("--dataset", required=True, help="Dataset name used for data/output organization")
    parser.add_argument("--data-path", help="JSONL or JSON data file. Defaults to data/{dataset}/questions.jsonl")
    parser.add_argument("--limit", type=int, help="Only run the first N loaded questions")
    parser.add_argument("--model", help="Hermes model name")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names aligned with --api-key-envs",
    )
    parser.add_argument("--provider", default=os.getenv("HERMES_PROVIDER", "custom"), help="Hermes provider")
    parser.add_argument("--base-url", default=os.getenv("HERMES_BASE_URL", ""), help="Model API base URL")
    parser.add_argument(
        "--base-urls",
        default="",
        help="Comma-separated model API base URLs for least-loaded per-run scheduling",
    )
    parser.add_argument(
        "--key-base-urls",
        default="",
        help="Comma-separated base URLs aligned one-to-one with --api-key-envs",
    )
    parser.add_argument(
        "--api-modes",
        default="",
        help="Comma-separated API modes aligned with --api-key-envs: chat_completions or codex_responses",
    )
    parser.add_argument(
        "--workers-per-endpoint",
        type=int,
        default=None,
        help="Maximum concurrent runs per endpoint when --base-urls is used",
    )
    parser.add_argument(
        "--endpoint-registry",
        default=os.getenv("ENDPOINT_REGISTRY", ""),
        help="Registry v1 JSON file used instead of static endpoint arguments",
    )
    parser.add_argument(
        "--endpoint-registry-refresh-interval",
        type=_positive_float,
        default=os.getenv("ENDPOINT_REGISTRY_REFRESH_INTERVAL", "30"),
        help="Seconds between live registry reloads (default: 30)",
    )
    parser.add_argument(
        "--workers-per-key",
        default=None,
        help="One limit for every key, or comma-separated limits aligned with --api-key-envs",
    )
    parser.add_argument("--api-key-env", default="HERMES_API_KEY", help="Environment variable containing the API key")
    parser.add_argument(
        "--api-key-envs",
        default="",
        help="Comma-separated environment variable names containing API keys",
    )
    parser.add_argument("--api-key", default=None, help="API key override. Prefer --api-key-env for scripts.")
    parser.add_argument("--save-name", help="Output run name. Defaults to a filesystem-safe model name")
    parser.add_argument("--repeats", type=int, default=1, help="How many times to run each question")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel Hermes subprocesses")
    parser.add_argument("--max-rounds", type=int, default=60, help="Hermes max turns per run")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per run when Hermes fails")
    parser.add_argument("--timeout-seconds", type=int, help="Per-attempt subprocess timeout. Defaults to no subprocess timeout")
    parser.add_argument("--context-length", type=int, help="Explicit model context length to write into each Hermes profile")
    parser.add_argument(
        "--context-compression",
        choices=("true", "false"),
        default="true",
        help="Enable Hermes automatic context compression (default: true)",
    )
    parser.add_argument(
        "--compression-threshold",
        type=float,
        default=0.5,
        help="Compress at this fraction of the model context window (default: 0.5)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "max"),
        default=None,
        help="Optional reasoning effort: low, medium, high, or max. Omitted means use the API/model default.",
    )
    parser.add_argument("--tool-whitelist", default=DEFAULT_TOOL_WHITELIST, help="Comma-separated Hermes toolsets to enable")
    parser.add_argument("--skill-whitelist", default="", help="Comma-separated Hermes skills to preload")
    parser.add_argument(
        "--enable-custom-system-prompt",
        type=_explicit_bool,
        default=False,
        metavar="BOOL",
        help="Replace the main Agent System Prompt with a file: true/false (default: false)",
    )
    parser.add_argument(
        "--custom-system-prompt-file",
        help="UTF-8 System Prompt file; valid only when --enable-custom-system-prompt is true",
    )
    parser.add_argument(
        "--enable-subagent-custom-system-prompt",
        type=_explicit_bool,
        default=False,
        metavar="BOOL",
        help=(
            "Replace the Subagent Hermes core System Prompt with a file: "
            "true/false (default: false)"
        ),
    )
    parser.add_argument(
        "--subagent-custom-system-prompt-file",
        help=(
            "UTF-8 Subagent core System Prompt file; valid only when "
            "--enable-subagent-custom-system-prompt is true"
        ),
    )
    parser.add_argument("--output-root", default="output/preds", help="Root output directory")
    parser.add_argument(
        "--workspace-namespace",
        default="",
        help=(
            "Optional batch prefix for workspace directories. Use a unique run tag "
            "when multiple batches share the workspace root."
        ),
    )
    parser.add_argument(
        "--resume-from",
        help=(
            "Existing run root (the directory containing conv/) or existing conv directory. "
            "Failed records are resumed automatically and completed attempts count toward --max-retries."
        ),
    )
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--hermes-bin", default=os.getenv("HERMES_BIN", "hermes"), help="Hermes executable")
    parser.add_argument("--skip-statuses", default="success,policy_violation", help="Existing statuses to skip on resume")
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Rerun existing failed results. Failed results are skipped by default.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun even if a result file already exists")
    parser.add_argument(
        "--numbered-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Name conv directories/files by dataset order (for example q000001/q000001__r1.json)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned run count and exit")
    parser.add_argument("--no-quiet", action="store_true", help="Do not pass --quiet to Hermes chat")
    parser.add_argument("--accept-hooks", action="store_true", help="Pass --accept-hooks to Hermes")
    parser.add_argument("--ignore-rules", action="store_true", help="Pass --ignore-rules to Hermes")
    parser.add_argument(
        "--no-antihack",
        action="store_true",
        help="Do not enable the browse_comp_guard plugin for evaluation profiles",
    )
    parser.add_argument(
        "--question-match-mode",
        choices=("evaluation", "off"),
        default=os.getenv("BROWSE_COMP_QUESTION_MATCH_MODE", "evaluation"),
        help=(
            "Question-overlap filtering mode. Defaults to evaluation; use off "
            "for non-evaluation runs."
        ),
    )
    parser.add_argument(
        "--execution-backend",
        choices=("local", "ags"),
        default="local",
        help="Run Hermes locally or inside an AGS sandbox",
    )
    parser.add_argument(
        "--search-mode",
        choices=("external", "mock", "disabled"),
        default="external",
        help="Search provider mode inside each Hermes profile",
    )
    parser.add_argument("--max-tokens", type=_positive_int, help="Optional output-token cap")
    parser.add_argument(
        "--temperature",
        type=_nonnegative_float,
        help="Optional non-negative sampling temperature",
    )
    parser.add_argument(
        "--endpoint-watchdog",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Monitor model endpoints and fail over interrupted attempts",
    )
    parser.add_argument("--endpoint-watchdog-interval", type=_positive_float, default=30.0)
    parser.add_argument("--endpoint-watchdog-timeout", type=_positive_float, default=5.0)
    parser.add_argument("--endpoint-watchdog-failures", type=_positive_int, default=3)
    parser.add_argument(
        "--endpoint-watchdog-recovery-successes", type=_positive_int, default=2
    )
    parser.add_argument(
        "--all-endpoints-down-timeout", type=_positive_float, default=300.0
    )
    parser.add_argument("--ags-api-key-env", default="E2B_API_KEY")
    parser.add_argument("--ags-domain", default=os.getenv("E2B_DOMAIN", ""))
    parser.add_argument("--ags-template", default="node-python-hermes-26-7-1")
    parser.add_argument("--ags-lifetime-timeout", type=float, default=4200)
    parser.add_argument("--ags-command-timeout", type=float, default=3900)
    parser.add_argument("--ags-container-python", default="python3")
    parser.add_argument("--ags-container-hermes", default="/home/user/.local/bin/hermes")
    parser.add_argument("--ags-workspace", default="/home/user/workspace")
    parser.add_argument(
        "--ags-upload-bundle", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ags-upload-hermes-source", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ags-create-request-timeout", type=float, default=120)
    parser.add_argument("--ags-operation-request-timeout", type=float, default=60)
    parser.add_argument("--ags-kill-request-timeout", type=float, default=30)
    parser.add_argument("--ags-fetch-server-source-dir", type=Path)
    parser.add_argument(
        "--ags-fetch-server-install-dependencies",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--ags-fetch-server-port", type=int, default=18081)
    parser.add_argument("--ags-fetch-server-startup-timeout", type=float, default=30)
    parser.add_argument("--ags-fetch-server-install-timeout", type=float, default=300)
    parser.add_argument("--ags-fetch-server-probe-url", default="")
    return parser


def merge_stats(total: dict[str, int], batch: dict[str, int]) -> dict[str, int]:
    for key, value in batch.items():
        total[key] = total.get(key, 0) + value
    return total


def main(argv: list[str] | None = None) -> int:
    argv_tokens = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv_tokens)
    if not math.isfinite(args.compression_threshold) or not 0 < args.compression_threshold <= 1:
        print("--compression-threshold must be greater than 0 and at most 1", file=sys.stderr)
        return 2
    project_root = Path.cwd()
    try:
        custom_system_prompt = load_custom_system_prompt(
            enabled=args.enable_custom_system_prompt,
            file_path=args.custom_system_prompt_file,
            project_root=project_root,
        )
        subagent_custom_system_prompt = load_custom_system_prompt(
            enabled=args.enable_subagent_custom_system_prompt,
            file_path=args.subagent_custom_system_prompt_file,
            project_root=project_root,
            enable_option="--enable-subagent-custom-system-prompt",
            file_option="--subagent-custom-system-prompt-file",
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    workspace_namespace = (
        safe_id(args.workspace_namespace) if args.workspace_namespace else ""
    )

    def option_present(name: str) -> bool:
        return any(token == name or token.startswith(name + "=") for token in argv_tokens)

    try:
        args.endpoint_registry_refresh_interval = _positive_float(
            str(args.endpoint_registry_refresh_interval)
        )
    except (argparse.ArgumentTypeError, ValueError):
        print(
            "--endpoint-registry-refresh-interval must be a finite positive number",
            file=sys.stderr,
        )
        return 2

    if args.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        return 2
    if args.execution_backend == "ags":
        for option, value in (
            ("--ags-lifetime-timeout", args.ags_lifetime_timeout),
            ("--ags-command-timeout", args.ags_command_timeout),
            ("--ags-create-request-timeout", args.ags_create_request_timeout),
            ("--ags-operation-request-timeout", args.ags_operation_request_timeout),
            ("--ags-kill-request-timeout", args.ags_kill_request_timeout),
            ("--ags-fetch-server-startup-timeout", args.ags_fetch_server_startup_timeout),
            ("--ags-fetch-server-install-timeout", args.ags_fetch_server_install_timeout),
        ):
            if value <= 0:
                print(f"{option} must be positive", file=sys.stderr)
                return 2
        if not 1 <= args.ags_fetch_server_port <= 65535:
            print("--ags-fetch-server-port must be between 1 and 65535", file=sys.stderr)
            return 2
        if args.ags_fetch_server_source_dir is not None and not args.ags_upload_hermes_source:
            print(
                "--ags-fetch-server-source-dir requires --ags-upload-hermes-source",
                file=sys.stderr,
            )
            return 2
    base_urls = list(dict.fromkeys(parse_csv(args.base_urls)))
    key_base_urls = parse_csv(args.key_base_urls)
    api_modes = parse_csv(args.api_modes)
    endpoint_registry = Path(args.endpoint_registry) if args.endpoint_registry else None
    if endpoint_registry is not None and not endpoint_registry.is_absolute():
        endpoint_registry = (project_root / endpoint_registry).resolve()
    registry_mode = endpoint_registry is not None
    valid_api_modes = {"chat_completions", "codex_responses"}
    if any(mode not in valid_api_modes for mode in api_modes):
        print("--api-modes values must be chat_completions or codex_responses", file=sys.stderr)
        return 2
    if args.workers_per_endpoint is not None and args.workers_per_endpoint < 1:
        print("--workers-per-endpoint must be >= 1", file=sys.stderr)
        return 2
    if (
        args.workers_per_endpoint is not None
        and not base_urls
        and not key_base_urls
        and not registry_mode
    ):
        print(
            "--workers-per-endpoint requires --base-urls, --key-base-urls, or --endpoint-registry",
            file=sys.stderr,
        )
        return 2
    if key_base_urls and base_urls:
        print("--key-base-urls cannot be combined with --base-urls", file=sys.stderr)
        return 2
    if args.workers_per_key is not None and not args.api_key_envs:
        print("--workers-per-key requires --api-key-envs", file=sys.stderr)
        return 2

    data_path = Path(args.data_path) if args.data_path else Path(args.data_root) / args.dataset / "questions.jsonl"
    if not data_path.is_absolute():
        data_path = project_root / data_path

    toolsets = validate_tool_whitelist(parse_csv(args.tool_whitelist))
    skills = normalize_skill_whitelist(parse_csv(args.skill_whitelist))
    allowed_tools = resolve_toolsets(toolsets)
    skip_statuses = set(parse_csv(args.skip_statuses))

    api_key_envs = parse_csv(args.api_key_envs)
    models = parse_csv(args.models)
    if registry_mode:
        conflicts: list[str] = []
        for option, active in (
            ("--base-url", option_present("--base-url")),
            ("--base-urls", bool(base_urls)),
            ("--key-base-urls", bool(key_base_urls)),
            ("--models", bool(models)),
            ("--api-key-envs", bool(api_key_envs)),
            ("--workers-per-key", args.workers_per_key is not None),
            ("--api-key", args.api_key is not None),
        ):
            if active:
                conflicts.append(option)
        if conflicts:
            print(
                "--endpoint-registry cannot be combined with "
                + ", ".join(conflicts),
                file=sys.stderr,
            )
            return 2
        if not args.model:
            print("--endpoint-registry requires one --model", file=sys.stderr)
            return 2
        if not args.endpoint_watchdog:
            print("--endpoint-registry requires --endpoint-watchdog", file=sys.stderr)
            return 2
        if len(api_modes) > 1:
            print("--endpoint-registry accepts at most one --api-modes value", file=sys.stderr)
            return 2
        if args.workers_per_endpoint is None:
            args.workers_per_endpoint = 10
    if not args.model and not models:
        print("--model or --models is required", file=sys.stderr)
        return 2
    if models and not api_key_envs:
        print("--models requires --api-key-envs", file=sys.stderr)
        return 2
    if args.api_key and api_key_envs:
        print("--api-key cannot be combined with --api-key-envs", file=sys.stderr)
        return 2
    if api_key_envs:
        missing_envs = [name for name in api_key_envs if not os.getenv(name, "")]
        if missing_envs:
            print(f"missing API key environment variable(s): {','.join(missing_envs)}", file=sys.stderr)
            return 2
        api_keys = [os.environ[name] for name in api_key_envs]
        if models and len(models) != len(api_keys):
            print("--models and --api-key-envs must contain the same number of values", file=sys.stderr)
            return 2
        if key_base_urls and len(key_base_urls) != len(api_keys):
            print("--key-base-urls and --api-key-envs must contain the same number of values", file=sys.stderr)
            return 2
        if api_modes and len(api_modes) != len(api_keys):
            print("--api-modes and --api-key-envs must contain the same number of values", file=sys.stderr)
            return 2
        if len(set(api_keys)) != len(api_keys):
            print("--api-key-envs resolved to duplicate API keys", file=sys.stderr)
            return 2
    else:
        api_key = args.api_key or os.getenv(args.api_key_env, "")
        if not api_key:
            print(f"missing API key: set {args.api_key_env} or pass --api-key", file=sys.stderr)
            return 2
        api_keys = [api_key]
    lane_models = models or [args.model] * len(api_keys)
    primary_model = args.model or lane_models[0]

    if args.workers_per_key is not None:
        raw_worker_limits = parse_csv(args.workers_per_key)
        try:
            worker_limits = [int(value) for value in raw_worker_limits]
        except ValueError:
            print("--workers-per-key values must be positive integers", file=sys.stderr)
            return 2
        if not worker_limits or any(limit < 1 for limit in worker_limits):
            print("--workers-per-key values must be positive integers", file=sys.stderr)
            return 2
        if len(worker_limits) == 1:
            worker_limits *= len(api_keys)
        elif len(worker_limits) != len(api_keys):
            print("--workers-per-key must contain one value or one value per API key", file=sys.stderr)
            return 2
    else:
        default_limit = args.workers_per_endpoint or args.num_workers
        worker_limits = [default_limit] * len(api_keys)

    save_name = args.save_name or safe_id(primary_model)
    output_base = Path(args.output_root) / save_name / args.dataset
    if not output_base.is_absolute():
        output_base = project_root / output_base
    resume_from_dir = None
    if args.resume_from:
        resume_from = Path(args.resume_from)
        if not resume_from.is_absolute():
            resume_from = project_root / resume_from
        resume_from_dir = resume_from if resume_from.name == "conv" else resume_from / "conv"
        if not resume_from_dir.is_dir():
            print(f"resume source trajectory directory not found: {resume_from_dir}", file=sys.stderr)
            return 2
        if resume_from_dir.resolve() == (output_base / "conv").resolve():
            print(
                "--resume-from must use a different output run; choose a new --save-name",
                file=sys.stderr,
            )
            return 2
        if args.force:
            print("--resume-from cannot be combined with --force", file=sys.stderr)
            return 2
    rerun_failed = args.rerun_failed or resume_from_dir is not None

    effective_base_urls = (
        [] if registry_mode else key_base_urls or base_urls or [args.base_url]
    )
    if (
        args.endpoint_watchdog
        and not registry_mode
        and any(not base_url for base_url in effective_base_urls)
    ):
        print("--endpoint-watchdog requires an explicit model base URL", file=sys.stderr)
        return 2
    if (
        args.endpoint_watchdog
        and args.endpoint_watchdog_timeout >= args.endpoint_watchdog_interval
    ):
        print(
            "--endpoint-watchdog-timeout must be less than --endpoint-watchdog-interval",
            file=sys.stderr,
        )
        return 2
    pooled = registry_mode or bool(base_urls) or bool(key_base_urls) or len(api_keys) > 1
    if registry_mode:
        num_workers = args.workers_per_endpoint
    else:
        num_workers = (
            sum(worker_limits)
            if key_base_urls
            else len(effective_base_urls) * sum(worker_limits)
            if pooled
            else args.num_workers
        )

    items = load_dataset(data_path)
    if args.limit is not None:
        if args.limit < 1:
            print("--limit must be >= 1", file=sys.stderr)
            return 2
        items = items[: args.limit]
    total_runs = len(items) * args.repeats
    registry_entries_for_display: list[dict[str, object]] = []
    registry_display_error = ""
    if endpoint_registry is not None:
        try:
            registry_entries_for_display = model_entries(
                load_registry(endpoint_registry), primary_model
            )
        except EndpointRegistryError as exc:
            registry_display_error = str(exc)
            if args.dry_run:
                print(f"invalid endpoint registry: {exc}", file=sys.stderr)
                return 2

    print("Browse Comp Eval")
    print(f"  data: {data_path}")
    print(f"  dataset: {args.dataset}")
    print(f"  model: {primary_model}")
    print(f"  provider: {args.provider}")
    print(f"  save_name: {save_name}")
    print(f"  questions: {len(items)}")
    print(f"  limit: {args.limit if args.limit is not None else '(none)'}")
    print(f"  repeats: {args.repeats}")
    print(f"  runs: {total_runs}")
    print(f"  run_schedule: repeat batches ({len(items)} questions per batch)")
    if registry_mode:
        print(f"  endpoint_registry: {endpoint_registry}")
        print(
            "  endpoint_registry_refresh_interval: "
            f"{args.endpoint_registry_refresh_interval:g}s"
        )
        print(f"  workers_per_endpoint: {args.workers_per_endpoint}")
        if registry_display_error:
            print(f"  endpoint_registry_status: unreadable ({registry_display_error})")
        else:
            print(
                f"  endpoint_registry_model_records: {len(registry_entries_for_display)}"
            )
            for index, entry in enumerate(registry_entries_for_display, start=1):
                print(
                    f"    [{index}] base_url={entry['base_url']} "
                    f"status={entry['status']}"
                )
    elif key_base_urls:
        print("  model_key_routes:")
        for index, (model, env_name, base_url, api_mode, worker_limit) in enumerate(
            zip(
                lane_models,
                api_key_envs,
                key_base_urls,
                api_modes or ["chat_completions"] * len(api_keys),
                worker_limits,
            ),
            start=1,
        ):
            print(
                f"    [{index}] model={model} api_key_env={env_name} "
                f"base_url={base_url} api_mode={api_mode} workers={worker_limit}"
            )
    elif base_urls:
        print(f"  model_endpoints: {len(base_urls)}")
        for index, base_url in enumerate(base_urls, start=1):
            print(f"    [{index}] {base_url}")
    else:
        print(f"  model_endpoint: {args.base_url or '(provider default)'}")
    print(f"  api_keys: {len(api_keys)}")
    if models and not key_base_urls:
        print("  model_key_routes:")
        for index, (model, env_name, worker_limit) in enumerate(
            zip(lane_models, api_key_envs, worker_limits), start=1
        ):
            print(
                f"    [{index}] model={model} api_key_env={env_name} "
                f"workers={worker_limit}"
            )
    if pooled:
        if registry_mode:
            print("  endpoint_key_lanes: dynamic")
        else:
            lane_count = len(api_keys) if key_base_urls else len(effective_base_urls) * len(api_keys)
            print(f"  endpoint_key_lanes: {lane_count}")
            print(f"  workers_per_key: {','.join(str(limit) for limit in worker_limits)}")
    print(
        f"  workers: {'dynamic' if registry_mode else num_workers}"
        + (f" ({num_workers} per endpoint)" if registry_mode else "")
    )
    print(f"  retries: {args.max_retries}")
    print(f"  rerun_failed: {rerun_failed}")
    print(f"  numbered_output: {args.numbered_output}")
    print(f"  max_rounds: {args.max_rounds}")
    print(f"  max_tokens: {args.max_tokens if args.max_tokens is not None else '(model default)'}")
    print(f"  temperature: {args.temperature if args.temperature is not None else '(model default)'}")
    print(f"  endpoint_watchdog: {args.endpoint_watchdog}")
    if args.endpoint_watchdog:
        print(
            "  endpoint_watchdog_policy: "
            f"interval={args.endpoint_watchdog_interval:g}s "
            f"timeout={args.endpoint_watchdog_timeout:g}s "
            f"failures={args.endpoint_watchdog_failures} "
            f"recovery_successes={args.endpoint_watchdog_recovery_successes} "
            f"all_down={args.all_endpoints_down_timeout:g}s"
        )
    print(f"  timeout_seconds: {args.timeout_seconds if args.timeout_seconds is not None else '(none)'}")
    print(f"  context_length: {args.context_length if args.context_length else '(auto)'}")
    print(f"  context_compression: {args.context_compression}")
    print(f"  compression_threshold: {args.compression_threshold}")
    print(f"  custom_system_prompt: {args.enable_custom_system_prompt}")
    if args.enable_custom_system_prompt:
        print(f"  custom_system_prompt_file: {args.custom_system_prompt_file}")
    print(
        "  subagent_custom_system_prompt: "
        f"{args.enable_subagent_custom_system_prompt}"
    )
    if args.enable_subagent_custom_system_prompt:
        print(
            "  subagent_custom_system_prompt_file: "
            f"{args.subagent_custom_system_prompt_file}"
        )
    print(f"  workspace_namespace: {workspace_namespace or '(none)'}")
    print(f"  reasoning_effort: {args.reasoning_effort or '(api default)'}")
    print(f"  backend: {args.execution_backend}")
    print(f"  search_mode: {args.search_mode}")
    if args.execution_backend == "ags":
        print(f"  ags_template: {args.ags_template}")
        print(f"  ags_domain: {args.ags_domain or '(unset)'}")
        print(f"  ags_api_key_env: {args.ags_api_key_env}")
        print(f"  ags_api_key_configured: {bool(os.getenv(args.ags_api_key_env, ''))}")
        print(f"  ags_lifetime_timeout: {args.ags_lifetime_timeout}")
        print(f"  ags_command_timeout: {args.ags_command_timeout}")
        print(f"  ags_workspace: {args.ags_workspace}")
        print(f"  ags_upload_bundle: {args.ags_upload_bundle}")
        print(f"  ags_upload_hermes_source: {args.ags_upload_hermes_source}")
        print(
            "  ags_fetch_server: "
            + ("enabled" if args.ags_fetch_server_source_dir is not None else "disabled")
        )
    print(f"  hermes_quiet: {not args.no_quiet}")
    print(f"  tool_whitelist: {','.join(toolsets) if toolsets else '(none)'}")
    print(f"  skill_whitelist: {','.join(skills) if skills else '(none)'}")
    print(f"  question_match_mode: {args.question_match_mode}")
    print(f"  antihack: {'disabled' if args.no_antihack else 'enabled'}")
    print(f"  resume_from: {resume_from_dir or '(none)'}")
    print(f"  output: {output_base}")

    if args.dry_run:
        return 0

    hermes_config = HermesConfig(
        hermes_bin=args.hermes_bin,
        api_key=api_keys[0],
        base_url="" if registry_mode else args.base_url,
        model=primary_model,
        provider=args.provider,
        save_name=save_name,
        dataset=args.dataset,
        max_rounds=args.max_rounds,
        timeout_seconds=args.timeout_seconds,
        toolsets=toolsets,
        skills=skills,
        custom_system_prompt=custom_system_prompt,
        subagent_custom_system_prompt=subagent_custom_system_prompt,
        api_mode=api_modes[0] if len(api_modes) == 1 else None,
        context_length=args.context_length,
        context_compression=args.context_compression == "true",
        compression_threshold=args.compression_threshold,
        reasoning_effort=args.reasoning_effort,
        quiet=not args.no_quiet,
        accept_hooks=args.accept_hooks,
        ignore_rules=args.ignore_rules,
        question_match_mode=args.question_match_mode,
        antihack_enabled=not args.no_antihack,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        search_mode=args.search_mode,
    )
    ags_backend = None
    client_factory = HermesClient
    if args.execution_backend == "ags":
        ags_api_key = os.getenv(args.ags_api_key_env, "")
        if not ags_api_key:
            print(f"missing AGS API key: set {args.ags_api_key_env}", file=sys.stderr)
            return 2
        if not args.ags_domain:
            print("missing AGS domain: pass --ags-domain or set E2B_DOMAIN", file=sys.stderr)
            return 2
        from .ags_worker import AGSWorker, AGSWorkerConfig
        from .e2b_ags_backend import E2BAGSBackend, E2BAGSBackendConfig

        ags_backend = E2BAGSBackend(
            E2BAGSBackendConfig(
                api_key=ags_api_key,
                domain=args.ags_domain,
                create_request_timeout=args.ags_create_request_timeout,
                operation_request_timeout=args.ags_operation_request_timeout,
                kill_request_timeout=args.ags_kill_request_timeout,
            )
        )
        worker_config = AGSWorkerConfig(
            template=args.ags_template,
            lifetime_timeout=args.ags_lifetime_timeout,
            command_timeout=args.ags_command_timeout,
            container_python=args.ags_container_python,
            container_hermes_bin=args.ags_container_hermes,
            upload_worker_bundle=args.ags_upload_bundle,
            upload_hermes_source=args.ags_upload_hermes_source,
            workspace_path=args.ags_workspace,
            fetch_server_source_dir=args.ags_fetch_server_source_dir,
            fetch_server_install_dependencies=args.ags_fetch_server_install_dependencies,
            fetch_server_port=args.ags_fetch_server_port,
            fetch_server_startup_timeout=args.ags_fetch_server_startup_timeout,
            fetch_server_install_timeout=args.ags_fetch_server_install_timeout,
            fetch_server_probe_url=args.ags_fetch_server_probe_url,
        )

        def client_factory(route_config: HermesConfig) -> HermesClient:
            return AGSWorker(route_config, worker_config, ags_backend)  # type: ignore[return-value]

    watchdog_config = EndpointWatchdogConfig(
        enabled=args.endpoint_watchdog,
        interval_seconds=args.endpoint_watchdog_interval,
        request_timeout_seconds=args.endpoint_watchdog_timeout,
        failure_threshold=args.endpoint_watchdog_failures,
        recovery_success_threshold=args.endpoint_watchdog_recovery_successes,
        all_down_timeout_seconds=args.all_endpoints_down_timeout,
    )
    client = (
        MultiEndpointHermesClient(
            hermes_config,
            effective_base_urls,
            worker_limits,
            api_keys=api_keys,
            models=lane_models,
            key_base_urls=key_base_urls or None,
            api_modes=api_modes or None,
            client_factory=client_factory,
            watchdog_config=watchdog_config,
            watchdog_event_path=output_base / "endpoint_health.jsonl",
            endpoint_registry=endpoint_registry,
            endpoint_registry_refresh_interval=args.endpoint_registry_refresh_interval,
        )
        if pooled or args.endpoint_watchdog
        else client_factory(hermes_config)
    )
    runner = BrowseCompRunner(
        client=client,
        config=RunnerConfig(
            output_dir=output_base / "conv",
            workspace_dir=output_base / "workspace",
            summary_path=output_base / "summary.jsonl",
            num_workers=num_workers,
            max_retries=args.max_retries,
            force=args.force,
            skip_statuses=skip_statuses,
            allowed_tools=allowed_tools,
            workspace_namespace=workspace_namespace,
            numbered_output=args.numbered_output,
            rerun_failed=rerun_failed,
            resume_from_dir=resume_from_dir,
        ),
    )
    all_runs = [
        spec
        for repeat_index in range(1, args.repeats + 1)
        for spec in expand_repeat(items, repeat_index)
    ]
    seed_stats: dict[str, int] | None = None
    if resume_from_dir is not None:
        seed_stats = runner.seed_resume_results(all_runs)
        initial_metrics = runner.rewrite_summary(all_runs)
        print("Resume seed")
        print("  " + " ".join(f"{key}={value}" for key, value in sorted(seed_stats.items())))
        print("Overall snapshot before resume")
        print("  " + " ".join(f"{key}={value}" for key, value in sorted(initial_metrics.items())))

    stats: dict[str, int] = {}
    overall_metrics: dict[str, int] | None = None
    interrupted = False
    run_finished = False
    previous_handlers: dict[int, object] = {}
    start_watchdog = getattr(client, "start_watchdog", None)
    stop_watchdog = getattr(client, "stop_watchdog", None)
    ags_close_all = getattr(ags_backend, "close_all", None)
    if callable(ags_close_all):
        previous_handlers = _install_ags_signal_handlers(ags_close_all)
    try:
        if callable(start_watchdog):
            start_watchdog()
        for repeat_index in range(1, args.repeats + 1):
            runs = expand_repeat(items, repeat_index)
            print(f"repeat batch {repeat_index}/{args.repeats}")
            batch_stats = runner.run(runs)
            merge_stats(stats, batch_stats)
            print("  " + " ".join(f"{key}={value}" for key, value in sorted(batch_stats.items())))
            if bool(getattr(client, "watchdog_aborted", False)):
                break
        run_finished = True
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted", file=sys.stderr)
    finally:
        if callable(stop_watchdog):
            stop_watchdog()
        if callable(ags_close_all):
            try:
                cleanup_errors = ags_close_all() or ()
                for error in cleanup_errors:
                    print(f"AGS cleanup error: {error}", file=sys.stderr)
            except Exception:
                print("AGS cleanup failed", file=sys.stderr)
            finally:
                _restore_signal_handlers(previous_handlers)
        if resume_from_dir is not None:
            overall_metrics = runner.rewrite_summary(all_runs)
            overall_complete = (
                overall_metrics["observed"] == overall_metrics["total"]
                and overall_metrics["missing"] == 0
                and overall_metrics["malformed"] == 0
                and overall_metrics["interrupted"] == 0
            )
            metrics_payload = {
                "scope": "overall",
                "state": (
                    "complete"
                    if run_finished and overall_complete
                    else "interrupted"
                    if interrupted
                    else "incomplete"
                ),
                "resume_from": str(resume_from_dir.parent),
                "output": str(output_base),
                "summary_path": str(output_base / "summary.jsonl"),
                "seed": seed_stats,
                "overall": overall_metrics,
            }
            metrics_path = output_base / "resume_metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(metrics_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(metrics_path)
            print("Overall resume results")
            print("  " + " ".join(f"{key}={value}" for key, value in sorted(overall_metrics.items())))
            print(f"  summary: {output_base / 'summary.jsonl'}")
            print(f"  metrics: {metrics_path}")

    watchdog_aborted = bool(getattr(client, "watchdog_aborted", False))
    if watchdog_aborted:
        abort_reason = str(
            getattr(client, "watchdog_abort_reason", "")
            or "all model endpoints unavailable"
        )
        snapshot = getattr(client, "endpoint_health_snapshot", lambda: [])()
        registry_state = getattr(client, "endpoint_registry_state", lambda: {})()
        run_state_path = output_base / "run_state.json"
        run_state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = run_state_path.with_suffix(run_state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "state": "endpoint_unavailable",
                    "exit_code": TEMPORARY_ENDPOINT_FAILURE_EXIT_CODE,
                    "reason": abort_reason,
                    "output": str(output_base),
                    "stats": stats,
                    "endpoints": snapshot,
                    "endpoint_registry": registry_state,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(run_state_path)
        print(abort_reason, file=sys.stderr)
        print(f"run state: {run_state_path}", file=sys.stderr)
        return TEMPORARY_ENDPOINT_FAILURE_EXIT_CODE

    if interrupted:
        return 130

    print("done")
    print("  " + " ".join(f"{key}={value}" for key, value in sorted(stats.items())))
    final_stats = overall_metrics if overall_metrics is not None else stats
    bad = final_stats.get("failed", 0) + final_stats.get("policy_violation", 0)
    return 0 if bad == 0 else 1


def _install_ags_signal_handlers(close_all) -> dict[int, object]:
    previous: dict[int, object] = {}

    def handle_signal(signum, frame) -> None:
        try:
            close_all()
        except Exception:
            pass
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)
    return previous


def _restore_signal_handlers(previous: dict[int, object]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
