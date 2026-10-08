# Evaluation

BrowseComp runner for the local Hermes backend and AGS sandboxes. Only the local launcher, AGS launcher, and judge launcher are included. Input datasets and run results are not distributed.

## Setup

Requires Python 3.11–3.13. Install the sibling [`Evaluation_backend`](../Evaluation_backend/README.md) first, then from this directory:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[ags]'
cp config/credentials.env.example config/credentials.env
# Edit config/credentials.env with your own credentials and service URLs.
set -a; source config/credentials.env; set +a
```

Provide a licensed JSONL dataset outside Git, for example `data/browsecomp/questions.jsonl`. Each row needs `question`, `answer`, and `type`; `id` is optional. The answer is used only by the offline judge.

## Local run

```bash
scripts/run_browsecomp.sh \
  --model YOUR_MODEL --base-url "$HERMES_BASE_URL" \
  --dataset browsecomp --data-path data/browsecomp/questions.jsonl \
  --save-name local_run --workers-per-endpoint 1
```

The launcher uses `../Evaluation_backend/.venv/bin/hermes` and `.venv/bin/python` by default. Set `HERMES_BIN` or `EVAL_PYTHON` if your install differs. Set `HERMES_HOME` to a private directory containing your Hermes `config.yaml`.

## AGS sandbox run

Supply `E2B_DOMAIN`, `E2B_API_KEY`, `AGS_TEMPLATE`, `SEARCH_SERVER_ENDPOINT`, `SEARCH_SERVER_API_KEY`, and `AGS_FETCH_SERVER_SOURCE_DIR` in `config/credentials.env`. The Fetch Server source directory must contain `server.py` and `requirements.lock.txt` or `requirements.txt`. The AGS template and Fetch Server are external prerequisites.

```bash
scripts/run_browsecomp_ags.sh \
  --model YOUR_MODEL --base-url "$HERMES_BASE_URL" \
  --dataset browsecomp --data-path data/browsecomp/questions.jsonl \
  --save-name ags_run --workers-per-endpoint 1
```

## Judge

```bash
scripts/judge_browsecomp.sh \
  --save-name local_run --dataset browsecomp \
  --judge-base-url "$JUDGE_BASE_URL" --judge-model YOUR_JUDGE_MODEL
```

Predictions go to `output/preds/<save-name>/<dataset>/conv/`. Judge results and metrics are written alongside them. Use `--help` on each script for all options. `--dry-run` is available for validation without model or judge API calls. Keep `config/credentials.env`, input data, `output/`, and `logs/` out of Git.
