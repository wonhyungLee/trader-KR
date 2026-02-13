from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.brokers.kis_broker import KISBroker
from src.monitor.scanner import Scanner
from src.monitor.signal_engine import SignalEngine
from src.monitor.state_store import StateStore
from src.monitor.subscription_manager import SubscriptionManager
from src.monitor.ws_client import KISWebSocketClient
from src.utils.config import load_settings


def load_universe(settings: dict) -> List[str]:
    monitor = settings.get("monitor", {}) or {}
    paths = monitor.get(
        "universe_paths",
        ["data/universe_kospi100.csv", "data/universe_kosdaq150.csv"],
    )
    codes: List[str] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"universe file missing: {p}")
        df = pd.read_csv(path)
        col = "code" if "code" in df.columns else "Code" if "Code" in df.columns else df.columns[0]
        codes.extend(df[col].astype(str).str.zfill(6).tolist())
    if codes:
        # unique preserve order
        seen = set()
        uniq = []
        for c in codes:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    return []


def load_baseline(settings: dict) -> Dict[str, Dict]:
    db_path = settings.get("database", {}).get("path", "data/market_data.db")
    conn = sqlite3.connect(db_path)
    try:
        query = """
        SELECT dp.code, dp.date, dp.close, dp.ma25, dp.disparity, um.name, um.market
        FROM daily_price dp
        JOIN (
            SELECT code, MAX(date) AS max_date
            FROM daily_price
            GROUP BY code
        ) m ON dp.code = m.code AND dp.date = m.max_date
        LEFT JOIN universe_members um ON dp.code = um.code
        """
        cur = conn.execute(query)
        baseline: Dict[str, Dict] = {}
        for row in cur.fetchall():
            code = str(row[0]).zfill(6)
            baseline[code] = {
                "date": row[1],
                "close": row[2],
                "ma25": row[3],
                "disparity": row[4],
                "name": row[5] or "",
                "market": row[6] or "",
            }
        return baseline
    finally:
        conn.close()


async def monitor_loop():
    settings = load_settings()
    monitor_cfg = settings.get("monitor", {}) or {}
    if not monitor_cfg.get("enabled", False):
        print("monitor disabled (config.settings.yaml monitor.enabled=false)")
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    universe = load_universe(settings)
    baseline = load_baseline(settings)
    state = StateStore(monitor_cfg.get("state_path", "data/monitor_state.json"))
    signal_engine = SignalEngine(settings, baseline, state)
    broker = KISBroker(settings)
    scanner = Scanner(settings, broker)
    sub_manager = SubscriptionManager(settings, baseline, state)
    ws_client = KISWebSocketClient(settings, state, signal_engine.on_tick)

    ws_task = asyncio.create_task(ws_client.run_forever())
    scan_interval = int(monitor_cfg.get("scan_interval_sec", 60))

    try:
        while True:
            snapshot = await asyncio.to_thread(scanner.scan_once, universe)
            if snapshot:
                signal_engine.on_snapshot(snapshot)
                targets = sub_manager.compute_targets(snapshot)
                await ws_client.set_targets(targets)
                state.save()
            await asyncio.sleep(scan_interval)
    finally:
        ws_task.cancel()


def main():
    asyncio.run(monitor_loop())


if __name__ == "__main__":
    main()
