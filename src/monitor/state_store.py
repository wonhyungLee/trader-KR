from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Set


class StateStore:
    def __init__(self, path: str = "data/monitor_state.json"):
        self.path = Path(path)
        self.current_subs: Set[str] = set()
        self.last_sub_ts: Dict[str, float] = {}
        self.last_unsub_ts: Dict[str, float] = {}
        self.last_alert_ts: Dict[str, float] = {}
        self.last_prices: Dict[str, float] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.current_subs = set(data.get("current_subs", []))
            self.last_sub_ts = {k: float(v) for k, v in data.get("last_sub_ts", {}).items()}
            self.last_unsub_ts = {k: float(v) for k, v in data.get("last_unsub_ts", {}).items()}
            self.last_alert_ts = {k: float(v) for k, v in data.get("last_alert_ts", {}).items()}
            self.last_prices = {k: float(v) for k, v in data.get("last_prices", {}).items()}
        except Exception:
            return

    def save(self) -> None:
        payload = {
            "current_subs": sorted(self.current_subs),
            "last_sub_ts": self.last_sub_ts,
            "last_unsub_ts": self.last_unsub_ts,
            "last_alert_ts": self.last_alert_ts,
            "last_prices": self.last_prices,
            "updated_at": time.time(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def mark_subscribed(self, code: str) -> None:
        now = time.time()
        self.current_subs.add(code)
        self.last_sub_ts[code] = now

    def mark_unsubscribed(self, code: str) -> None:
        now = time.time()
        self.current_subs.discard(code)
        self.last_unsub_ts[code] = now

    def can_resubscribe(self, code: str, cooldown_sec: int) -> bool:
        last = self.last_unsub_ts.get(code)
        if last is None:
            return True
        return (time.time() - last) >= cooldown_sec

    def should_alert(self, key: str, cooldown_sec: int) -> bool:
        last = self.last_alert_ts.get(key)
        if last is None:
            return True
        return (time.time() - last) >= cooldown_sec

    def mark_alert(self, key: str) -> None:
        self.last_alert_ts[key] = time.time()

    def update_price(self, code: str, price: float) -> None:
        self.last_prices[code] = price
