# Evaluation backend (Hermes)

Hermes source used by the evaluation runner. The code retains its upstream MIT license in [`LICENSE`](LICENSE).

Requires Python 3.11–3.13. From this directory:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
cp config.yaml.example config.yaml
export HERMES_HOME="$PWD/.hermes"
mkdir -p "$HERMES_HOME"
cp config.yaml "$HERMES_HOME/config.yaml"
.venv/bin/hermes --help
```

The sample config keeps evaluation settings from our run and leaves model and search endpoints empty. Fill in your own model name and `base_url`; set credentials through environment variables. The evaluation launcher also supplies model settings for each run. Keep the populated config and `.env` private. AGS sandboxes need a separately prepared template containing Hermes, or source upload as configured by the launcher.
