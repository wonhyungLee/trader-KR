#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
import sys
from pathlib import Path

# Ensure repo root on sys.path when executed from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_settings  # noqa: E402
from src.utils.notifier import maybe_notify  # noqa: E402
from src.utils.project_root import ensure_repo_root  # noqa: E402


def _read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_counts(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM universe_members").fetchone()[0] or 0
    done = conn.execute("SELECT COUNT(*) FROM refill_progress WHERE status='DONE'").fetchone()[0] or 0
    running = conn.execute("SELECT COUNT(*) FROM refill_progress WHERE status='RUNNING'").fetchone()[0] or 0
    error = conn.execute("SELECT COUNT(*) FROM refill_progress WHERE status='ERROR'").fetchone()[0] or 0
    last_update = conn.execute("SELECT MAX(updated_at) FROM refill_progress").fetchone()[0]
    return {
        "total": int(total),
        "done": int(done),
        "running": int(running),
        "error": int(error),
        "last_update": last_update,
    }


def main():
    ap = argparse.ArgumentParser(description="Monitor refill_progress and notify via Discord.")
    ap.add_argument("--interval", type=int, default=300, help="poll interval seconds")
    ap.add_argument("--notify-interval", type=int, default=900, help="min seconds between notifications")
    ap.add_argument("--state-file", default="data/refill_monitor_state.json", help="state json path")
    ap.add_argument("--once", action="store_true", help="run once and exit")
    args = ap.parse_args()

    ensure_repo_root(Path(__file__).resolve())
    settings = load_settings()
    state_path = Path(args.state_file)

    while True:
        conn = sqlite3.connect("data/market_data.db")
        conn.row_factory = sqlite3.Row
        counts = _get_counts(conn)
        conn.close()

        state = _read_state(state_path)
        now = int(time.time())
        last_notify_ts = int(state.get("last_notify_ts", 0) or 0)
        last_counts = state.get("last_counts", {})

        changed = counts != last_counts
        due = (now - last_notify_ts) >= int(args.notify_interval)

        if changed or due:
            msg = (
                f"[refill-monitor] done={counts['done']}/{counts['total']} "
                f"running={counts['running']} error={counts['error']} "
                f"last_update={counts['last_update']}"
            )
            maybe_notify(settings, msg)
            state["last_notify_ts"] = now
            state["last_counts"] = counts

        if counts["total"] > 0 and counts["done"] >= counts["total"]:
            if not state.get("completed_notified"):
                maybe_notify(settings, f"[refill-monitor] completed done={counts['done']}/{counts['total']}")
                state["completed_notified"] = True

        _write_state(state_path, state)

        if args.once:
            break
        time.sleep(int(args.interval))


if __name__ == "__main__":
    main()
