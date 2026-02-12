#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/bnf-kr-viewer

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
.venv/bin/python - <<'PY'
from src.utils.config import load_settings
from src.utils.notifier import maybe_notify
s=load_settings()
maybe_notify(s, "[repair] fetch latest daily prices (2 days) for all universe")
PY

# Fetch last 2 days per code (incremental)
.venv/bin/python -u -m src.collectors.daily_loader --chunk-days 2

# Recompute indicators for last 5 days to fix MA25/disparity
.venv/bin/python -u scripts/recompute_daily_indicators.py --start $(date -d "-10 days" +%Y-%m-%d) --notify

.venv/bin/python - <<'PY'
from src.utils.config import load_settings
from src.utils.notifier import maybe_notify
s=load_settings()
maybe_notify(s, "[repair] done: latest daily + ma25/disparity recompute")
PY
