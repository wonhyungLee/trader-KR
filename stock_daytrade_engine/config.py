from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default

def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default

def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v not in (None, "") else default

def _env_bool(key: str, default: bool = True) -> bool:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")

@dataclass(frozen=True)
class EngineConfig:
    # Data
    db_path: str = _env_str("STOCK_DB_PATH", "market_data.db")
    table: str = _env_str("STOCK_DB_TABLE", "daily_price")

    # Strategy (Connors RSI(2) mean-reversion + SMA200 trend filter)
    rsi_period: int = _env_int("DT_RSI_PERIOD", 2)
    rsi_thresh: float = _env_float("DT_RSI_THRESH", 10.0)
    sma_fast: int = _env_int("DT_SMA_FAST", 5)
    sma_trend: int = _env_int("DT_SMA_TREND", 200)
    atr_period: int = _env_int("DT_ATR_PERIOD", 14)

    # Day-trade bracket params (grid search will pick best for win-rate)
    entry_k_grid: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    stop_mult_grid: Tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0)
    target_mult_grid: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

    # Backtest window (bars)
    optimize_lookback_bars: int = _env_int("DT_OPTIMIZE_LOOKBACK_BARS", 2000)
    min_bars_for_indicators: int = _env_int("DT_MIN_BARS_FOR_INDICATORS", 260)
    min_trades_for_score: int = _env_int("DT_MIN_TRADES_FOR_SCORE", 30)

    # Costs
    # fee_bps is applied on entry and exit (each side).
    # We model: entry_cost = entry*(1+fee), exit_proceeds = exit*(1-fee)
    fee_bps: float = _env_float("DT_FEE_BPS", 5.0)

    # Filters
    use_trend_filter: bool = _env_bool("DT_USE_TREND_FILTER", True)

    # Optional constraint to avoid 'win-rate-only' degenerate solutions
    min_fill_rate: Optional[float] = None  # e.g., 0.15

    # Risk helper in plan output
    risk_pct_default: float = _env_float("DT_RISK_PCT_DEFAULT", 1.0)

    # Execution assumption when both stop & target are touched within the same daily bar (unknown order):
    # - "stop_first" is conservative (assume loss)
    both_hit_rule: str = _env_str("DT_BOTH_HIT_RULE", "stop_first")  # stop_first | target_first (NOT recommended)
