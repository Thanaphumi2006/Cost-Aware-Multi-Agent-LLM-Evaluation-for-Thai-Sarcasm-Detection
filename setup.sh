#!/usr/bin/env bash
# Stand up the Thai Sarcasm Detector demo reproducibly -- the full cascade
# (cue in the browser -> WangchanBERTa on this machine -> gpt-4.1-mini).
# One command, survives reboots. Run from the repo root:  ./setup.sh
#
#   OPENAI_API_KEY   read from .env (or the environment) -- without it, escalation
#                    stays cue/WCB-only and the server still runs.
#   PORT=8000        override the port.        PYTHON=python3.11  pick the interpreter.
#   ./setup.sh --no-serve   set up everything but don't launch (just prepare).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PY="${PYTHON:-python3}"
VENV="$HERE/.venv"
PORT="${PORT:-8000}"
SERVE=1
[ "${1:-}" = "--no-serve" ] && SERVE=0

echo "==> [1/4] Python virtual environment (.venv)"
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,9) else 1)' 2>/dev/null; then
  echo "    ERROR: need Python >= 3.9 (found: $("$PY" --version 2>&1)). Set PYTHON=... to a newer one." >&2
  exit 1
fi
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip

echo "==> [2/4] Install pinned dependencies (includes torch for the WangchanBERTa tier)"
pip install -q -r requirements.txt

echo "==> [3/4] WangchanBERTa model (the middle tier)"
if [ -f "Gold/wcb_model/config.json" ]; then
  echo "    already built (Gold/wcb_model/) -- skipping"
else
  echo "    training once on the 127-item gold set (~10-15 min on CPU, downloads the base model)..."
  ( cd Gold && python train_final_wcb.py )
fi

if [ "$SERVE" = "0" ]; then
  echo "==> ready. Launch later with:  source .venv/bin/activate && cd Gold && python serve_public.py --port $PORT"
  exit 0
fi

echo "==> [4/4] Launch the demo"
if [ -z "${OPENAI_API_KEY:-}" ] && [ ! -f ".env" ]; then
  echo "    NOTE: no OPENAI_API_KEY and no .env file -> escalation stays cue/WCB-only (server still runs)."
fi
echo "    serving the user page at  http://127.0.0.1:$PORT/app   (Ctrl-C to stop)"
cd Gold && exec python serve_public.py --port "$PORT"
