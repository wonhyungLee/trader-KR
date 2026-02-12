#!/usr/bin/env bash
set -e
# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$DIR")"
cd "$PARENT_DIR"

echo "Starting refill at $(date)"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PARENT_DIR"

./myenv/bin/python -u -m src.collectors.refill_loader 
  --universe data/universe_kospi100.csv 
  --universe data/universe_kosdaq150.csv 
  --chunk-days 150 
  --resume

echo "Refill script exited with $?"
