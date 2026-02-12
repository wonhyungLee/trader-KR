#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure repo root is on sys.path when executed from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_settings  # noqa: E402
from src.utils.notifier import maybe_notify  # noqa: E402
from src.utils.project_root import ensure_repo_root  # noqa: E402


def _to_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    text = str(val).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _to_int(val: Optional[str]) -> Optional[int]:
    if val is None:
        return None
    text = str(val).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _norm_code(val: Optional[str]) -> str:
    text = (val or "").strip()
    return text.zfill(6) if text.isdigit() else text


def main():
    ap = argparse.ArgumentParser(description="Import daily_price CSV into SQLite with upsert.")
    ap.add_argument("--csv", default="/home/ubuntu/daily_price.csv", help="CSV file path")
    ap.add_argument("--db", default=None, help="SQLite DB path (default: from settings)")
    ap.add_argument("--chunk-size", type=int, default=20000, help="Rows per batch commit")
    ap.add_argument("--notify-every", type=int, default=0, help="Notify every N rows (0 to disable)")
    args = ap.parse_args()

    ensure_repo_root(Path(__file__).resolve())
    settings = load_settings()
    db_path = args.db or settings.get("database", {}).get("path", "data/market_data.db")
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA busy_timeout=5000;")

    insert_sql = (
        "INSERT OR REPLACE INTO daily_price"
        "(date, code, open, high, low, close, volume, amount, ma25, disparity) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)"
    )

    start_ts = time.time()
    total = 0
    batch = []
    last_notify = 0

    maybe_notify(settings, f"[import] start file={csv_path} chunk={args.chunk_size}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = (
                row.get("date"),
                _norm_code(row.get("code")),
                _to_float(row.get("open")),
                _to_float(row.get("high")),
                _to_float(row.get("low")),
                _to_float(row.get("close")),
                _to_int(row.get("volume")),
                _to_float(row.get("amount")),
                _to_float(row.get("ma25")),
                _to_float(row.get("disparity")),
            )
            batch.append(rec)
            if len(batch) >= args.chunk_size:
                conn.executemany(insert_sql, batch)
                conn.commit()
                total += len(batch)
                batch = []
                if args.notify_every and (total - last_notify) >= args.notify_every:
                    elapsed = int(time.time() - start_ts)
                    maybe_notify(settings, f"[import] rows={total} elapsed={elapsed}s")
                    last_notify = total

    if batch:
        conn.executemany(insert_sql, batch)
        conn.commit()
        total += len(batch)

    elapsed = int(time.time() - start_ts)
    maybe_notify(settings, f"[import] completed rows={total} elapsed={elapsed}s")
    print(f"imported {total} rows in {elapsed}s")


if __name__ == "__main__":
    main()
