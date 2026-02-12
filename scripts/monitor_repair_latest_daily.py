#!/usr/bin/env python
"""Monitor repair_latest_daily progress and notify Discord."""
from __future__ import annotations

import argparse
import time
import sqlite3
import subprocess
from datetime import datetime

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_settings
from src.utils.notifier import maybe_notify


def _pgrep(pattern: str) -> bool:
    try:
        res = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False)
        return res.returncode == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=120, help="seconds between updates")
    args = parser.parse_args()

    settings = load_settings()
    db_path = settings.get("database", {}).get("path", "data/market_data.db")

    maybe_notify(settings, "[repair] progress monitor started")

    while True:
        running = _pgrep("repair_latest_daily.py") or _pgrep("src.collectors.daily_loader")
        try:
            conn = sqlite3.connect(db_path)
            total = conn.execute("SELECT COUNT(*) FROM universe_members").fetchone()[0]
            max_date = conn.execute("SELECT MAX(date) FROM daily_price").fetchone()[0]
            updated = 0
            if max_date:
                updated = conn.execute(
                    "SELECT COUNT(DISTINCT code) FROM daily_price WHERE date=?",
                    (max_date,),
                ).fetchone()[0]
            conn.close()
        except Exception:
            total = 0
            max_date = None
            updated = 0

        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        maybe_notify(settings, f"[repair] progress {updated}/{total} latest_date={max_date} ts={ts}")

        if not running:
            break
        time.sleep(max(30, int(args.interval)))

    maybe_notify(settings, "[repair] progress monitor finished")


if __name__ == "__main__":
    main()
