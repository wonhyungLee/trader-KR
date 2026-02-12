#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT" ]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "$ROOT"

PYBIN="./myenv/bin/python"
if [ ! -x "$PYBIN" ]; then
  PYBIN="python3"
fi

# wait for other data loaders to finish
while pgrep -f accuracy_data_loader >/dev/null || pgrep -f universe_loader >/dev/null; do
  sleep 60
done

$PYBIN -m src.collectors.sector_classifier --refresh-days 30 --sleep 0.5 > logs/sector_classifier.log 2>&1

$PYBIN - <<'PY' > logs/sector_verify.log 2>&1
import pandas as pd
import sqlite3
from pathlib import Path

conn = sqlite3.connect('data/market_data.db')
conn.row_factory = sqlite3.Row

uni = pd.read_sql_query('SELECT code, market FROM universe_members', conn)
uni_count = len(uni)

sector = pd.read_sql_query('SELECT code, sector_name FROM sector_map', conn)
known = sector[sector['sector_name'].notna()]['code'].nunique()
unknown = uni_count - known
ratio = known / uni_count if uni_count else 0

print(f'universe_members={uni_count}')
print(f'sector_map known={known} unknown={unknown} ratio={ratio:.2f}')

root = Path('data/universe_sectors')
code_set = set()
files = list(root.rglob('*.csv'))
for p in files:
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    if 'code' not in df.columns:
        continue
    for c in df['code'].astype(str).str.zfill(6).tolist():
        code_set.add(c)

print(f'sector CSV total unique codes={len(code_set)}')
missing = set(uni['code'].astype(str).str.zfill(6).tolist()) - code_set
print(f'sector CSV missing={len(missing)}')
if missing:
    miss_path = root / 'MISSING.csv'
    pd.DataFrame({'code': sorted(missing)}).to_csv(miss_path, index=False)
    print(f'missing list saved: {miss_path}')
PY
