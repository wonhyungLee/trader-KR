#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/bnf-kr-viewer

LOCK_PATH="data/selection_data_loader.lock"
if [[ -f "$LOCK_PATH" ]]; then
  existing_pid=$(cat "$LOCK_PATH" 2>/dev/null || true)
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "selection data loader already running pid=$existing_pid"
    exit 0
  fi
fi

echo $$ > "$LOCK_PATH"
trap 'rm -f "$LOCK_PATH"' EXIT

# Wait for refill loader to finish to avoid API rate-limit overlap
while pgrep -f "src.collectors.refill_loader" >/dev/null 2>&1; do
  sleep 60
 done

# Sector classification (KIS stock info APIs)
/home/ubuntu/bnf-kr-viewer/.venv/bin/python -u -m src.collectors.sector_classifier \
  --refresh-days 0 \
  --sleep 1.0 \
  --notify-every 25

# Selection-related accuracy data (investor flow, program trade, short sale, credit balance, loan, VI)
/home/ubuntu/bnf-kr-viewer/.venv/bin/python -u -m src.collectors.accuracy_data_loader \
  --days 10 \
  --resume \
  --notify-every 10
