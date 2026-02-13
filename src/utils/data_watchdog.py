from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from src.utils.config import load_settings
from src.utils.notifier import maybe_notify
from src.utils.project_root import ensure_repo_root


ACCURACY_TABLES = {
    "investor_flow_daily": "inv",
    "program_trade_daily": "prog",
    "short_sale_daily": "short",
    "credit_balance_daily": "credit",
    "loan_trans_daily": "loan",
    "vi_status_daily": "vi",
}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _read_state(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(path: Path, state: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _lock_active(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
    try:
        pid = int(lock_path.read_text(encoding="utf-8").strip() or "0")
    except Exception:
        pid = 0
    if _pid_alive(pid):
        return True
    try:
        lock_path.unlink()
    except Exception:
        pass
    return False


def _get_last_price_date(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute("SELECT MAX(date) FROM daily_price").fetchone()
    if row and row[0]:
        return str(row[0])
    return None


def _missing_codes_for_date(conn: sqlite3.Connection, table: str, date: str) -> List[str]:
    sql = (
        f"SELECT u.code "
        f"FROM universe_members u "
        f"LEFT JOIN {table} t ON u.code=t.code AND t.date=? "
        f"WHERE t.code IS NULL"
    )
    rows = conn.execute(sql, (date,)).fetchall()
    return [r[0] for r in rows]


def _date_lag_days(last_date: str, tz_name: str = "Asia/Seoul") -> int:
    try:
        ld = datetime.strptime(str(last_date), "%Y-%m-%d").date()
    except Exception:
        return 0
    today = datetime.now(ZoneInfo(tz_name)).date()
    return max(0, (today - ld).days)


def _daily_snapshot(db_path: str = "data/market_data.db") -> Dict[str, Optional[object]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT COUNT(*) FROM daily_price").fetchone()[0]
        codes = conn.execute("SELECT COUNT(DISTINCT code) FROM daily_price").fetchone()[0]
        max_date_row = conn.execute("SELECT MAX(date) FROM daily_price").fetchone()
        max_date = max_date_row[0] if max_date_row else None
        missing = None
        if max_date:
            try:
                missing = len(_missing_codes_for_date(conn, "daily_price", str(max_date)))
            except sqlite3.OperationalError:
                missing = None
        return {
            "rows": int(rows or 0),
            "codes": int(codes or 0),
            "max_date": str(max_date) if max_date else None,
            "missing": int(missing or 0) if missing is not None else None,
        }
    finally:
        conn.close()


def _accuracy_missing_snapshot(date: str, db_path: str = "data/market_data.db") -> tuple[Dict[str, int], int]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        missing_map: Dict[str, int] = {}
        missing_union: set[str] = set()
        for table, label in ACCURACY_TABLES.items():
            try:
                miss = _missing_codes_for_date(conn, table, date)
            except sqlite3.OperationalError:
                miss = []
            missing_map[label] = len(miss)
            missing_union.update(miss)
        return missing_map, len(missing_union)
    finally:
        conn.close()


def _write_codes_csv(path: Path, codes: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("code\n")
        for c in codes:
            f.write(f"{c}\n")


def _run_accuracy_refill(
    date: str,
    codes_file: Path,
    progress_file: Path,
    lock_file: Path,
    notify_every: int,
    item_sleep: float,
    rate_sleep: Optional[float],
    limit: Optional[int],
):
    cmd = [
        sys.executable,
        "-m",
        "src.collectors.accuracy_data_loader",
        "--start-date",
        date,
        "--end-date",
        date,
        "--codes-file",
        str(codes_file),
        "--notify-every",
        str(notify_every),
        "--sleep",
        str(item_sleep),
        "--progress-file",
        str(progress_file),
        "--lock-file",
        str(lock_file),
    ]
    if rate_sleep is not None:
        cmd.extend(["--rate-sleep", str(rate_sleep)])
    if limit is not None and limit > 0:
        cmd.extend(["--limit", str(limit)])

    logging.info("Running accuracy refill: %s", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    return res.returncode


def _run_daily_refill(chunk_days: int):
    cmd = [
        sys.executable,
        "-m",
        "src.collectors.daily_loader",
        "--chunk-days",
        str(chunk_days),
    ]
    logging.info("Running daily refill: %s", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    return res.returncode


def _load_cfg(settings: dict, args) -> Dict[str, object]:
    wd = settings.get("watchdog", {})
    def _get(name: str, default):
        return wd.get(name, default)

    interval = float(args.interval if args.interval is not None else _get("interval_sec", 1800))
    cooldown = float(args.cooldown if args.cooldown is not None else _get("accuracy_cooldown_sec", 21600))
    min_missing = int(args.min_missing if args.min_missing is not None else _get("accuracy_min_missing", 1))
    daily_min_missing = int(args.daily_min_missing if args.daily_min_missing is not None else _get("daily_min_missing", 1))
    daily_chunk_days = int(args.daily_chunk_days if args.daily_chunk_days is not None else _get("daily_chunk_days", 90))
    daily_cooldown = float(args.daily_cooldown if args.daily_cooldown is not None else _get("daily_cooldown_sec", 7200))
    daily_stale_days = int(args.daily_stale_days if args.daily_stale_days is not None else _get("daily_stale_days", 1))
    status_notify_interval = float(
        args.status_notify_interval if args.status_notify_interval is not None else _get("status_notify_interval_sec", 1800)
    )
    if args.daily_enabled is None:
        daily_enabled = bool(_get("daily_enabled", True))
    else:
        daily_enabled = bool(args.daily_enabled)
    notify_every = int(args.notify_every if args.notify_every is not None else _get("accuracy_notify_every", 20))
    item_sleep = float(args.item_sleep if args.item_sleep is not None else _get("accuracy_item_sleep_sec", 0.5))
    rate_sleep = args.rate_sleep if args.rate_sleep is not None else _get("accuracy_rate_sleep_sec", None)
    if rate_sleep is not None:
        rate_sleep = float(rate_sleep)

    progress_file = Path(args.progress_file or _get("accuracy_progress_file", "data/accuracy_progress_watchdog.json"))
    codes_file = Path(args.codes_file or _get("accuracy_codes_file", "data/csv/accuracy_missing_codes.csv"))
    accuracy_lock = Path(args.accuracy_lock_file or _get("accuracy_lock_file", "data/accuracy_loader.lock"))
    daily_lock = Path(args.daily_lock_file or _get("daily_lock_file", "data/daily_loader.lock"))
    lock_file = Path(args.lock_file or _get("lock_file", "data/watchdog.lock"))
    state_file = Path(args.state_file or "data/watchdog_state.json")

    return {
        "interval": interval,
        "cooldown": cooldown,
        "min_missing": min_missing,
        "daily_min_missing": daily_min_missing,
        "daily_chunk_days": daily_chunk_days,
        "daily_cooldown": daily_cooldown,
        "daily_stale_days": daily_stale_days,
        "status_notify_interval": status_notify_interval,
        "daily_enabled": daily_enabled,
        "notify_every": notify_every,
        "item_sleep": item_sleep,
        "rate_sleep": rate_sleep,
        "progress_file": progress_file,
        "codes_file": codes_file,
        "accuracy_lock_file": accuracy_lock,
        "daily_lock_file": daily_lock,
        "lock_file": lock_file,
        "state_file": state_file,
        "limit": args.limit,
        "once": args.once,
        "no_refill": args.no_refill,
    }


def run_once(settings: dict, cfg: Dict[str, object]) -> None:
    state_path: Path = cfg["state_file"]
    state = _read_state(state_path)

    conn = sqlite3.connect("data/market_data.db")
    conn.row_factory = sqlite3.Row
    try:
        last_date = _get_last_price_date(conn)
        if not last_date:
            maybe_notify(settings, "[watchdog] no daily_price date found; skip")
            return

        # missing on last date
        missing_map: Dict[str, int] = {}
        missing_union: List[str] = []
        missing_set = set()

        for table, label in ACCURACY_TABLES.items():
            try:
                miss = _missing_codes_for_date(conn, table, last_date)
            except sqlite3.OperationalError:
                miss = []
            missing_map[label] = len(miss)
            for c in miss:
                if c not in missing_set:
                    missing_set.add(c)
                    missing_union.append(c)

        # daily_price missing for last_date (informational)
        try:
            miss_daily = _missing_codes_for_date(conn, "daily_price", last_date)
            daily_missing_count = len(miss_daily)
        except sqlite3.OperationalError:
            daily_missing_count = 0

        status_msg = (
            f"[watchdog] date={last_date} daily_missing={daily_missing_count} "
            f"inv={missing_map.get('inv', 0)} prog={missing_map.get('prog', 0)} "
            f"short={missing_map.get('short', 0)} credit={missing_map.get('credit', 0)} "
            f"loan={missing_map.get('loan', 0)} vi={missing_map.get('vi', 0)} "
            f"union={len(missing_union)}"
        )
        status_sig = (
            f"{last_date}:{daily_missing_count}:{missing_map.get('inv',0)}:{missing_map.get('prog',0)}:"
            f"{missing_map.get('short',0)}:{missing_map.get('credit',0)}:{missing_map.get('loan',0)}:"
            f"{missing_map.get('vi',0)}:{len(missing_union)}"
        )
        now_ts = time.time()
        last_status_sig = str(state.get("last_status_sig") or "")
        last_status_ts = float(state.get("last_status_notify_ts", 0) or 0)
        status_interval = float(cfg["status_notify_interval"])
        should_notify_status = status_sig != last_status_sig
        if status_interval <= 0 or (now_ts - last_status_ts) >= status_interval:
            should_notify_status = True
        if should_notify_status:
            maybe_notify(settings, status_msg)
            state["last_status_notify_ts"] = now_ts
        else:
            logging.info(status_msg)
        state["last_status_sig"] = status_sig

        if cfg["no_refill"]:
            _write_state(state_path, state)
            return

        date_lag = _date_lag_days(last_date)
        stale_trigger = date_lag >= int(cfg["daily_stale_days"])
        missing_trigger = daily_missing_count >= int(cfg["daily_min_missing"])

        # Daily refill (fill missing daily_price)
        if cfg["daily_enabled"] and (missing_trigger or stale_trigger):
            daily_lock_path: Path = cfg["daily_lock_file"]
            if _lock_active(daily_lock_path):
                maybe_notify(settings, "[watchdog] daily loader already running; skip")
            else:
                last_daily_ts = float(state.get("last_daily_run_ts", 0) or 0)
                cooldown = float(cfg["daily_cooldown"])
                now = time.time()
                if cooldown > 0 and (now - last_daily_ts) < cooldown:
                    remain = int(cooldown - (now - last_daily_ts))
                    maybe_notify(settings, f"[watchdog] daily cooldown active; skip ({remain}s left)")
                else:
                    before = _daily_snapshot()
                    daily_lock_path.parent.mkdir(parents=True, exist_ok=True)
                    daily_lock_path.write_text(str(os.getpid()), encoding="utf-8")
                    try:
                        maybe_notify(
                            settings,
                            f"[watchdog] daily refill start missing={daily_missing_count} lag_days={date_lag} date={last_date}",
                        )
                        rc = _run_daily_refill(int(cfg["daily_chunk_days"]))
                        after = _daily_snapshot()
                        state["last_daily_run_ts"] = time.time()
                        state["last_daily_date"] = last_date
                        state["last_daily_missing"] = daily_missing_count
                        state["last_daily_lag_days"] = date_lag
                        state["last_daily_rc"] = rc
                        _write_state(state_path, state)
                        if rc != 0:
                            maybe_notify(settings, f"[watchdog] daily refill exited rc={rc}")
                        else:
                            row_delta = int(after.get("rows") or 0) - int(before.get("rows") or 0)
                            msg = (
                                f"[watchdog] daily refill done rc={rc} "
                                f"rows={before.get('rows')}->{after.get('rows')} (delta={row_delta}) "
                                f"max_date={before.get('max_date')}->{after.get('max_date')} "
                                f"missing={before.get('missing')}->{after.get('missing')}"
                            )
                            maybe_notify(settings, msg)
                    finally:
                        try:
                            daily_lock_path.unlink()
                        except Exception:
                            pass

        if len(missing_union) < int(cfg["min_missing"]):
            _write_state(state_path, state)
            return

        lock_path: Path = cfg["accuracy_lock_file"]
        if _lock_active(lock_path):
            maybe_notify(settings, "[watchdog] accuracy loader already running; skip")
            _write_state(state_path, state)
            return

        last_run_ts = float(state.get("last_accuracy_run_ts", 0) or 0)
        cooldown = float(cfg["cooldown"])
        now = time.time()
        if cooldown > 0 and (now - last_run_ts) < cooldown:
            remain = int(cooldown - (now - last_run_ts))
            maybe_notify(settings, f"[watchdog] accuracy cooldown active; skip ({remain}s left)")
            _write_state(state_path, state)
            return

        codes_file: Path = cfg["codes_file"]
        _write_codes_csv(codes_file, missing_union)

        maybe_notify(settings, f"[watchdog] accuracy refill start codes={len(missing_union)} date={last_date}")

        rc = _run_accuracy_refill(
            date=last_date,
            codes_file=codes_file,
            progress_file=cfg["progress_file"],
            lock_file=cfg["accuracy_lock_file"],
            notify_every=int(cfg["notify_every"]),
            item_sleep=float(cfg["item_sleep"]),
            rate_sleep=cfg["rate_sleep"],
            limit=cfg["limit"],
        )
        after_map, after_union = _accuracy_missing_snapshot(last_date)
        state["last_accuracy_run_ts"] = time.time()
        state["last_accuracy_date"] = last_date
        state["last_accuracy_missing"] = len(missing_union)
        state["last_accuracy_rc"] = rc
        _write_state(state_path, state)
        if rc != 0:
            maybe_notify(settings, f"[watchdog] accuracy refill exited rc={rc}")
        else:
            maybe_notify(
                settings,
                (
                    f"[watchdog] accuracy refill done rc={rc} date={last_date} "
                    f"union={len(missing_union)}->{after_union} "
                    f"inv={missing_map.get('inv', 0)}->{after_map.get('inv', 0)} "
                    f"prog={missing_map.get('prog', 0)}->{after_map.get('prog', 0)} "
                    f"short={missing_map.get('short', 0)}->{after_map.get('short', 0)} "
                    f"credit={missing_map.get('credit', 0)}->{after_map.get('credit', 0)} "
                    f"loan={missing_map.get('loan', 0)}->{after_map.get('loan', 0)} "
                    f"vi={missing_map.get('vi', 0)}->{after_map.get('vi', 0)}"
                ),
            )
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=None, help="loop interval seconds")
    parser.add_argument("--once", action="store_true", help="run once and exit")
    parser.add_argument("--no-refill", action="store_true", help="only notify, do not trigger refill")
    parser.add_argument("--cooldown", type=float, default=None, help="accuracy refill cooldown seconds")
    parser.add_argument("--daily-cooldown", type=float, default=None, help="daily refill cooldown seconds")
    parser.add_argument("--status-notify-interval", type=float, default=None, help="status notify interval seconds")
    parser.add_argument("--min-missing", type=int, default=None, help="min missing codes to trigger refill")
    parser.add_argument("--daily-min-missing", type=int, default=None, help="min missing daily codes to trigger refill")
    parser.add_argument("--daily-stale-days", type=int, default=None, help="trigger daily refill when last daily date is stale by N days")
    parser.add_argument("--daily-chunk-days", type=int, default=None, help="daily loader chunk days")
    parser.add_argument("--daily-lock-file", type=str, default=None, help="daily loader lock file")
    parser.add_argument("--notify-every", type=int, default=None, help="notify every n codes during refill")
    parser.add_argument("--item-sleep", type=float, default=None, help="sleep seconds per code during refill")
    parser.add_argument("--rate-sleep", type=float, default=None, help="override broker rate sleep for refill")
    parser.add_argument("--progress-file", type=str, default=None, help="progress file for watchdog refill")
    parser.add_argument("--codes-file", type=str, default=None, help="missing codes csv output")
    parser.add_argument("--accuracy-lock-file", type=str, default=None, help="accuracy loader lock file")
    parser.add_argument("--lock-file", type=str, default=None, help="watchdog lock file")
    parser.add_argument("--state-file", type=str, default=None, help="watchdog state json file")
    parser.add_argument("--limit", type=int, default=None, help="limit codes for testing")
    parser.add_argument("--daily-enabled", dest="daily_enabled", action="store_true", help="enable daily refill")
    parser.add_argument("--no-daily", dest="daily_enabled", action="store_false", help="disable daily refill")
    parser.set_defaults(daily_enabled=None)
    args = parser.parse_args()

    ensure_repo_root(Path(__file__).resolve())
    settings = load_settings()
    cfg = _load_cfg(settings, args)

    watchdog_lock = cfg["lock_file"]
    if _lock_active(watchdog_lock):
        maybe_notify(settings, "[watchdog] already running; exit")
        return
    watchdog_lock.parent.mkdir(parents=True, exist_ok=True)
    watchdog_lock.write_text(str(os.getpid()), encoding="utf-8")

    try:
        while True:
            run_once(settings, cfg)
            if cfg["once"]:
                break
            time.sleep(float(cfg["interval"]))
    finally:
        try:
            watchdog_lock.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
