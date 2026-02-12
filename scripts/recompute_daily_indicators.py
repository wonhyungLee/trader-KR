#!/usr/bin/env python
"""Recompute MA25 and disparity for daily_price, optionally for a date range.

This fixes missing indicators caused by partial loads.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
import sqlite3
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_settings
from src.utils.notifier import maybe_notify
from src.storage.sqlite_store import SQLiteStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _to_date(value: str | None) -> str | None:
    if not value:
        return None
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--notify", action="store_true", help="send discord notifications")
    args = parser.parse_args()

    settings = load_settings()
    store = SQLiteStore(settings.get("database", {}).get("path", "data/market_data.db"))
    conn = store.conn

    start = _to_date(args.start)
    end = _to_date(args.end)

    if args.notify:
        maybe_notify(settings, f"[repair] recompute ma25/disparity start={start or 'ALL'} end={end or 'ALL'}")

    # Load all prices (we need full series to compute rolling)
    df = pd.read_sql_query(
        "SELECT date, code, open, high, low, close, volume, amount FROM daily_price",
        conn,
    )
    if df.empty:
        logging.warning("daily_price empty")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["code", "date"])
    df["ma25"] = df.groupby("code")["close"].transform(lambda s: s.rolling(25, min_periods=5).mean())
    df["disparity"] = df["close"] / df["ma25"] - 1

    # filter range if provided
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]

    records = df[["date", "code", "open", "high", "low", "close", "volume", "amount", "ma25", "disparity"]].copy()
    records["date"] = records["date"].dt.strftime("%Y-%m-%d")

    rows = [tuple(r) for r in records.to_numpy()]
    conn.executemany(
        """
        INSERT OR REPLACE INTO daily_price
        (date, code, open, high, low, close, volume, amount, ma25, disparity)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.commit()

    if args.notify:
        maybe_notify(settings, f"[repair] recompute done rows={len(rows)}")


if __name__ == "__main__":
    main()
