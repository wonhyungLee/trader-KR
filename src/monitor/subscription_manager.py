from __future__ import annotations

from typing import Dict, List, Set, Tuple

from src.monitor.state_store import StateStore


def _to_float(value) -> float:
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return 0.0


class SubscriptionManager:
    def __init__(self, settings: dict, baseline: Dict[str, Dict], state: StateStore):
        self.settings = settings
        self.baseline = baseline
        self.state = state
        monitor = settings.get("monitor", {}) or {}
        signal = monitor.get("signal", {}) or {}
        self.threshold = float(signal.get("disparity_threshold", -0.08))
        self.max_ws_subs = int(monitor.get("max_ws_subs", 20))
        self.subscribe_cooldown = int(monitor.get("subscribe_cooldown_sec", 180))

    def _score(self, code: str, price: float, amount: float) -> float:
        base = self.baseline.get(code) or {}
        ma25 = _to_float(base.get("ma25"))
        # Use previous close as baseline for selection when available.
        base_price = _to_float(base.get("close")) if base.get("close") is not None else 0.0
        if base_price <= 0:
            base_price = price
        if ma25 <= 0 or base_price <= 0:
            return 0.0
        disparity = (base_price / ma25) - 1
        distance = abs(disparity - self.threshold)
        score = 1.0 / (distance + 1e-6)
        # mild liquidity weight
        if amount > 0:
            score *= (1.0 + min(3.0, (amount / 1e11)))
        return score

    def compute_targets(self, snapshot: Dict[str, Dict]) -> Set[str]:
        scored: List[Tuple[str, float]] = []
        for code, data in snapshot.items():
            price = _to_float(data.get("price"))
            amount = _to_float(data.get("amount"))
            if price <= 0:
                continue
            score = self._score(code, price, amount)
            if score <= 0:
                continue
            scored.append((code, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        candidates = [c for c, _ in scored[: self.max_ws_subs]]
        targets: Set[str] = set()
        for code in candidates:
            if code in self.state.current_subs:
                targets.add(code)
                continue
            if self.state.can_resubscribe(code, self.subscribe_cooldown):
                targets.add(code)
        return targets
