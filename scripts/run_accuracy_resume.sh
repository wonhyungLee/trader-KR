#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/종목선별매매프로그램
source .venv/bin/activate
export PYTHONUNBUFFERED=1
# Resume from accuracy_progress.json automatically
python -u -m src.collectors.accuracy_data_loader \
  --resume \
  --notify-every 5
