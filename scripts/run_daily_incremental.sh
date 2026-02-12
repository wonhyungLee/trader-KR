#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/종목선별매매프로그램
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -u -m src.collectors.daily_loader
