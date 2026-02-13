from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from src.utils.notifier import maybe_notify
from src.monitor.state_store import StateStore


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return None


class SignalEngine:
    def __init__(self, settings: dict, baseline: Dict[str, Dict], state: StateStore):
        self.settings = settings
        self.baseline = baseline
        self.state = state
        monitor = settings.get("monitor", {}) or {}
        signal = monitor.get("signal", {}) or {}
        self.signal_type = signal.get("type", "disparity_threshold")
        self.threshold = float(signal.get("disparity_threshold", -0.08))
        self.alert_cooldown = int(monitor.get("alert_cooldown_sec", 600))
        self.use_intraday = bool(signal.get("use_intraday", True))

    def _calc_disparity(self, code: str, price: float) -> Optional[float]:
        base = self.baseline.get(code)
        if not base:
            return None
        ma25 = _to_float(base.get("ma25"))
        if not ma25 or ma25 <= 0:
            return None
        return (price / ma25) - 1

    def _maybe_alert(self, code: str, price: float, disparity: float, source: str):
        key = f"{code}:{self.signal_type}"
        if not self.state.should_alert(key, self.alert_cooldown):
            return
        name = self.baseline.get(code, {}).get("name", "")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            f"[monitor] {code} {name} "
            f"price={price:.2f} disparity={disparity:.4f} thr={self.threshold:.4f} "
            f"source={source} at {ts}"
        )
        maybe_notify(self.settings, msg)
        self.state.mark_alert(key)

    def on_tick(self, code: str, price: float, ts: Optional[str] = None, source: str = "ws") -> None:
        if not self.use_intraday:
            return
        if price <= 0:
            return
        self.state.update_price(code, price)
        if self.signal_type != "disparity_threshold":
            return
        disparity = self._calc_disparity(code, price)
        if disparity is None:
            return
        if disparity <= self.threshold:
            self._maybe_alert(code, price, disparity, source)

    def on_snapshot(self, snapshot: Dict[str, Dict]) -> None:
        if not self.use_intraday:
            return
        if self.signal_type != "disparity_threshold":
            return
        for code, data in snapshot.items():
            price = _to_float(data.get("price"))
            if not price or price <= 0:
                continue
            disparity = self._calc_disparity(code, price)
            if disparity is None:
                continue
            if disparity <= self.threshold:
                self._maybe_alert(code, price, disparity, "scan")
