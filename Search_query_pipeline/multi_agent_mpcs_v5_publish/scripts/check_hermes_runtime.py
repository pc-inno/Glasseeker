#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from browsecomp_v2.config import load_config
from browsecomp_v2.runner import _hermes_subprocess_env, _resolve_hermes_command


PROBE_CODE = """
import json
import inspect
import os
import shutil
import sys
import yaml
from hermes_cli.config import validate_config_structure
from hermes_constants import get_config_path, get_hermes_home
from run_agent import AIAgent

# Match AgentRunner's subprocess policy after run_agent has loaded
# HERMES_HOME/.env with override=True.
os.environ["BROWSER_CDP_URL"] = os.environ.get(
    "V2_HERMES_BROWSER_CDP_URL", ""
).strip()
os.environ["AGENT_BROWSER_AUTO_CONNECT"] = os.environ.get(
    "V2_HERMES_AGENT_BROWSER_AUTO_CONNECT", ""
).strip()

config_path = get_config_path()
if not config_path.is_file():
    raise FileNotFoundError(f"Hermes config not found: {config_path}")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(config, dict):
    raise TypeError(f"Hermes config must be a YAML mapping: {config_path}")
issues = validate_config_structure(config)
errors = [issue for issue in issues if issue.severity == "error"]
if errors:
    details = "; ".join(f"{issue.path}: {issue.message}" for issue in errors)
    raise ValueError(f"Hermes config validation failed: {details}")
web = config.get("web") if isinstance(config.get("web"), dict) else {}

print(json.dumps({
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "pyyaml": yaml.__file__,
    "aiagent": AIAgent.__module__ + "." + AIAgent.__name__,
    "aiagent_source": inspect.getfile(AIAgent),
    "hermes_home": str(get_hermes_home()),
    "config_path": str(config_path),
    "config_real_path": str(config_path.resolve()),
    "config_is_symlink": config_path.is_symlink(),
    "config_issue_count": len(issues),
    "web_search_backend": web.get("search_backend", ""),
    "web_extract_backend": web.get("extract_backend", ""),
    "browser_cdp_url": os.environ.get("BROWSER_CDP_URL", ""),
    "agent_browser_auto_connect": os.environ.get("AGENT_BROWSER_AUTO_CONNECT", ""),
    "agent_browser_cli": shutil.which("agent-browser") or "",
}))
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check the configured Hermes Python runtime and config without sending an API request."
        )
    )
    parser.add_argument("--env", default=".env_v5", help="workflow environment file")
    args = parser.parse_args()

    config = load_config(args.env)
    python_bin, hermes_root = _resolve_hermes_command(config.hermes_command)
    if not python_bin or not hermes_root:
        print(
            "Hermes runtime could not be resolved; check PYTHON_BIN, HERMES_VENV, "
            "REAL_HERMES_BIN, and V2_HERMES_COMMAND.",
            file=sys.stderr,
        )
        return 2

    print(f"python_bin={python_bin}")
    print(f"hermes_root={hermes_root}")
    proc = subprocess.run(
        [str(python_bin), "-c", PROBE_CODE],
        text=True,
        capture_output=True,
        env=_hermes_subprocess_env(hermes_root),
    )
    if proc.returncode != 0:
        print("Hermes runtime check failed:", file=sys.stderr)
        print(proc.stderr.strip(), file=sys.stderr)
        return proc.returncode or 1

    details = json.loads(proc.stdout)
    print(json.dumps(details, ensure_ascii=False, indent=2))
    print("Hermes runtime check passed; no API request was sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
