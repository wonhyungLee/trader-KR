from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

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


def _normalize_codes(codes: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for code in codes:
        if not code:
            continue
        norm = str(code).strip().zfill(6)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def fetch_selection_codes(settings: dict) -> List[str]:
    monitor = settings.get("monitor", {}) or {}
    api_url = monitor.get("selection_api") or os.getenv("SELECTION_API_URL")
    if not api_url:
        site_url = settings.get("site_url") or ""
        if site_url:
            api_url = site_url.rstrip("/") + "/selection"
        else:
            api_url = "http://127.0.0.1:5001/selection"
    try:
        resp = requests.get(api_url, timeout=5)
        resp.raise_for_status()
        payload = resp.json() if resp.content else {}
        candidates = payload.get("candidates") or []
        codes = [c.get("code") for c in candidates if isinstance(c, dict)]
        if not codes:
            final_items = (payload.get("stage_items") or {}).get("final") or []
            codes = [c.get("code") for c in final_items if isinstance(c, dict)]
        return _normalize_codes(codes)
    except Exception as exc:
        logging.warning("selection fetch failed (%s): %s", api_url, exc)
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

    baseline = load_baseline(settings)
    state = StateStore(monitor_cfg.get("state_path", "data/monitor_state.json"))
    signal_engine = SignalEngine(settings, baseline, state)
    broker = KISBroker(settings)
    scanner = Scanner(settings, broker)
    sub_manager = SubscriptionManager(settings, baseline, state)
    ws_client = KISWebSocketClient(settings, state, signal_engine.on_tick)

    ws_task = asyncio.create_task(ws_client.run_forever())
    scan_interval = int(monitor_cfg.get("scan_interval_sec", 60))
    last_selected: List[str] = []

    try:
        while True:
            selected = fetch_selection_codes(settings)
            if selected:
                last_selected = selected
            else:
                selected = last_selected

            if not selected:
                logging.info("no selection candidates; skip scan")
                await asyncio.sleep(scan_interval)
                continue

            snapshot = await asyncio.to_thread(scanner.scan_once, selected)
            if snapshot:
                signal_engine.on_snapshot(snapshot)
                targets = sub_manager.targets_from_selection(selected)
                await ws_client.set_targets(targets)
                state.save()
            await asyncio.sleep(scan_interval)
    finally:
        ws_task.cancel()


def main():
    asyncio.run(monitor_loop())


if __name__ == "__main__":
    main()
