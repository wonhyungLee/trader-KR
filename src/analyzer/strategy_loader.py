from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.utils.config import load_yaml


@dataclass
class StrategyParams:
    entry_mode: str
    liquidity_rank: int
    min_amount: float
    rank_mode: str
    buy_kospi: float
    buy_kosdaq: float
    sell_disparity: float
    take_profit_ret: float
    stop_loss: float
    max_holding_days: int
    max_positions: int
    max_per_sector: int
    initial_cash: float
    capital_utilization: float
    trend_ma25_rising: bool
    selection_horizon_days: int


def load_strategy(settings: Dict[str, Any]) -> StrategyParams:
    strat_file = Path("config/strategy.yaml")
    strat = load_yaml(strat_file) if strat_file.exists() else settings.get("strategy", {})

    buy_cfg = strat.get("buy", {}) or {}
    sell_cfg = strat.get("sell", {}) or {}
    pos_cfg = strat.get("position", {}) or {}
    trend_cfg = buy_cfg.get("trend_filter", {}) or {}
    report_cfg = strat.get("report", {}) or {}

    return StrategyParams(
        entry_mode=str(strat.get("entry_mode", "mean_reversion") or "mean_reversion"),
        liquidity_rank=int(strat.get("liquidity_rank", 300)),
        min_amount=float(strat.get("min_amount", 5e10)),
        rank_mode=str(strat.get("rank_mode", "amount") or "amount"),
        buy_kospi=float(buy_cfg.get("kospi_disparity", strat.get("disparity_buy_kospi", -0.05))),
        buy_kosdaq=float(buy_cfg.get("kosdaq_disparity", strat.get("disparity_buy_kosdaq", -0.10))),
        sell_disparity=float(sell_cfg.get("take_profit_disparity", strat.get("disparity_sell", -0.01))),
        take_profit_ret=float(sell_cfg.get("take_profit_ret", strat.get("take_profit_ret", 0.0)) or 0.0),
        stop_loss=float(sell_cfg.get("stop_loss", strat.get("stop_loss", -0.05))),
        max_holding_days=int(sell_cfg.get("max_holding_days", strat.get("max_holding_days", 3))),
        max_positions=int(pos_cfg.get("max_positions", strat.get("max_positions", 10))),
        max_per_sector=int(pos_cfg.get("max_per_sector", strat.get("max_per_sector", 0)) or 0),
        initial_cash=float(pos_cfg.get("initial_cash", strat.get("initial_cash", 10_000_000)) or 10_000_000),
        capital_utilization=float(pos_cfg.get("capital_utilization", strat.get("capital_utilization", 0.0)) or 0.0),
        trend_ma25_rising=bool(trend_cfg.get("ma25_rising", strat.get("trend_ma25_rising", False))),
        selection_horizon_days=int(report_cfg.get("selection_horizon_days", 1)),
    )
