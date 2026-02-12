#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT" ]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "$ROOT"
export PYTHONUNBUFFERED=1
INTERVAL_SEC=${STATUS_NOTIFY_INTERVAL_SEC:-600}
PYBIN="./myenv/bin/python"
if [ ! -x "$PYBIN" ]; then
  PYBIN="python3"
fi

while true; do
  if ! "$PYBIN" - <<'PY'
import sqlite3, json, time
from pathlib import Path
from src.utils.config import load_settings
from src.utils.notifier import maybe_notify
from src.collectors.refill_loader import read_universe

settings = load_settings()
conn = sqlite3.connect('data/market_data.db')
conn.row_factory = sqlite3.Row

# Refill progress
universe_codes = read_universe(['data/universe_kospi100.csv','data/universe_kosdaq150.csv'])
total_universe = len(universe_codes)
row = conn.execute("SELECT COUNT(*) FROM refill_progress WHERE status='DONE'").fetchone()
refill_done = row[0] if row else 0
# Keep remaining based on universe only
refill_remaining = max(total_universe - min(refill_done, total_universe), 0)

# Accuracy progress
prog_path = Path('data/accuracy_progress.json')
prog = {}
if prog_path.exists():
    try:
        prog = json.loads(prog_path.read_text(encoding='utf-8'))
    except Exception:
        prog = {}
acc_last = prog.get('last_index')
acc_total = prog.get('total')

# Accuracy missing counts (summary)
acc_tables = ['investor_flow_daily','program_trade_daily','short_sale_daily','credit_balance_daily','loan_trans_daily','vi_status_daily']
missing = {}
for t in acc_tables:
    row = conn.execute(
        f"SELECT COUNT(*) FROM stock_info s LEFT JOIN (SELECT DISTINCT code FROM {t}) t ON s.code=t.code WHERE t.code IS NULL"
    ).fetchone()
    missing[t] = row[0] if row else None

ts = time.strftime("%Y-%m-%d %H:%M:%S")
msg = (
    f"[DATA STATUS] {ts}\n"
    f"Refill (KOSPI100+KOSDAQ150): done {min(refill_done,total_universe)}/{total_universe}, remaining {refill_remaining}\n"
    f"Accuracy progress: last_index {acc_last}/{acc_total}\n"
    f"Accuracy missing codes: inv {missing['investor_flow_daily']}, prog {missing['program_trade_daily']}, short {missing['short_sale_daily']}, "
    f"credit {missing['credit_balance_daily']}, loan {missing['loan_trans_daily']}, vi {missing['vi_status_daily']}\n"
)

maybe_notify(settings, msg)
print(msg, flush=True)
PY
  then
    echo "[WARN] status_notifier python failed" >&2
  fi
  sleep "$INTERVAL_SEC"
done
