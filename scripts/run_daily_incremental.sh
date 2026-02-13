#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYBIN="python3"
if [ -x ".venv/bin/python" ]; then
  PYBIN=".venv/bin/python"
elif [ -x "myenv/bin/python" ]; then
  PYBIN="myenv/bin/python"
fi

export PYTHONUNBUFFERED=1
"$PYBIN" -u -m src.collectors.daily_loader
