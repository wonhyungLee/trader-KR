#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/종목선별매매프로그램
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u -m src.collectors.refill_loader \
  --universe data/universe_kospi100.csv \
  --universe data/universe_kosdaq150.csv \
  --chunk-days 150 \
  --resume \
  --notify-every 5
