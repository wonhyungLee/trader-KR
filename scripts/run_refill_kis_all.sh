#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/종목선별매매프로그램
source .venv/bin/activate
export PYTHONUNBUFFERED=1
# Full refill across stock_info (no universe args)
python -u -m src.collectors.refill_loader \
  --source kis \
  --chunk-days 120 \
  --cooldown 0.2 \
  --resume \
  --notify-every 5
