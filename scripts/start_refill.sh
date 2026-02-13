#!/usr/bin/env bash
set -euo pipefail
# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$DIR")"
cd "$PARENT_DIR"

echo "Starting refill at $(date)"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PARENT_DIR"

PYBIN="python3"
if [ -x ".venv/bin/python" ]; then
  PYBIN=".venv/bin/python"
elif [ -x "myenv/bin/python" ]; then
  PYBIN="myenv/bin/python"
fi

"$PYBIN" -u -m src.collectors.refill_loader \
  --universe data/universe_kospi200.csv \
  --universe data/universe_kosdaq150.csv \
  --chunk-days 150 \
  --resume

echo "Refill script exited with $?"
