#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYBIN="python3"
if [ -x ".venv/bin/python" ]; then
  PYBIN=".venv/bin/python"
elif [ -x "myenv/bin/python" ]; then
  PYBIN="myenv/bin/python"
fi

export PYTHONUNBUFFERED=1

"$PYBIN" -u -m src.collectors.refill_loader \
  --universe data/universe_kospi200.csv \
  --universe data/universe_kosdaq150.csv \
  --chunk-days 150 \
  --resume \
  > logs/refill_debug.log 2>&1
