from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .indicators import rolling_sma, rsi_sma, atr_sma

@dataclass
class Metrics:
    n_trades: int
    wins: int
    win_rate: float
    total_return: float
    mdd: float
    profit_factor: Optional[float]
    fill_rate: Optional[float]
    signals: int
    fills: int
    gross_profit: float
    gross_loss: float
    avg_return: float
    median_return: float

def _fee_rates(fee_bps: float) -> tuple[float, float]:
    fee = float(fee_bps) / 10000.0
    return (1.0 + fee, 1.0 - fee)

def simulate_daytrade_limit_long(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    *,
    rsi_period: int,
    rsi_thresh: float,
    sma_fast_period: int,
    sma_trend_period: int,
    atr_period: int,
    entry_k: float,
    stop_mult: float,
    target_mult: float,
    fee_bps: float,
    use_trend_filter: bool,
    both_hit_rule: str = "stop_first",
    start: int = 0,
    end: Optional[int] = None,
) -> Metrics:
    """LONG-only, next-day *day trade* (daily bars).

    Signal at day t close (index i):
      - optional trend: close[i] > SMA_trend[i]  (default SMA200)
      - mean reversion: close[i] < SMA_fast[i] AND RSI[i] <= rsi_thresh

    Order plan derived from day t close:
      - entry_limit = close[i] - entry_k * ATR[i]
      - stop        = entry_limit - stop_mult * ATR[i]
      - target      = entry_limit + target_mult * ATR[i]

    Execution on day t+1 (index i+1):
      - Fill if low[i+1] <= entry_limit, assume filled at entry_limit (conservative)
      - If filled, exit within the same day (t+1):
          * stop if low[i+1] <= stop
          * take-profit if high[i+1] >= target
          * else exit at close[i+1] (end-of-day)

    both_hit_rule:
      - "stop_first" (default, conservative): if stop and target touched same day, assume stop triggers first.
      - "target_first": optimistic (not recommended for honest backtests).

    Fees:
      - entry_cost = entry*(1+fee), exit_proceeds = exit*(1-fee)
    """
    o = np.asarray(o, dtype=float)
    h = np.asarray(h, dtype=float)
    l = np.asarray(l, dtype=float)
    c = np.asarray(c, dtype=float)

    n = len(c)
    if end is None:
        end = n
    end = min(int(end), n)
    start = max(0, int(start))
    if end - start < 3:
        return Metrics(0, 0, 0.0, 0.0, 0.0, None, None, 0, 0, 0.0, 0.0, 0.0, 0.0)

    sma_fast = rolling_sma(c, int(sma_fast_period))
    sma_trend = rolling_sma(c, int(sma_trend_period))
    rsi = rsi_sma(c, int(rsi_period))
    atr = atr_sma(h, l, c, int(atr_period))

    idx = np.arange(start, end - 1)  # signal days, needs i+1
    valid = ~np.isnan(sma_fast[idx]) & ~np.isnan(sma_trend[idx]) & ~np.isnan(rsi[idx]) & ~np.isnan(atr[idx])
    if use_trend_filter:
        valid &= c[idx] > sma_trend[idx]

    sig = valid & (c[idx] < sma_fast[idx]) & (rsi[idx] <= float(rsi_thresh))
    signals = int(sig.sum())
    if signals == 0:
        return Metrics(0, 0, 0.0, 0.0, 0.0, None, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    i_idx = idx[sig]
    atr_i = atr[i_idx]
    entry = c[i_idx] - float(entry_k) * atr_i
    stop = entry - float(stop_mult) * atr_i
    target = entry + float(target_mult) * atr_i

    low_n = l[i_idx + 1]
    high_n = h[i_idx + 1]
    close_n = c[i_idx + 1]

    fill = low_n <= entry
    fills = int(fill.sum())
    fill_rate = float(fills / signals) if signals else None
    if fills == 0:
        return Metrics(0, 0, 0.0, 0.0, 0.0, None, fill_rate, signals, 0, 0.0, 0.0, 0.0, 0.0)

    entry_f = entry[fill]
    stop_f = stop[fill]
    target_f = target[fill]
    low_f = low_n[fill]
    high_f = high_n[fill]
    close_f = close_n[fill]

    hit_stop = low_f <= stop_f
    hit_target = high_f >= target_f

    if both_hit_rule == "target_first":
        exit_price = np.where(hit_target, target_f, np.where(hit_stop, stop_f, close_f))
    else:
        # default: stop_first (conservative)
        exit_price = np.where(hit_stop, stop_f, np.where(hit_target, target_f, close_f))

    entry_mult, exit_mult = _fee_rates(float(fee_bps))
    entry_px = entry_f * entry_mult
    exit_px = exit_price * exit_mult
    rets = exit_px / entry_px - 1.0

    trades = int(len(rets))
    wins = int((rets > 0).sum())
    win_rate = float(wins / trades) if trades else 0.0

    eq = np.cumprod(1.0 + rets)
    total_return = float(eq[-1] - 1.0) if trades else 0.0
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    mdd = float(dd.max()) if trades else 0.0

    gross_profit = float(rets[rets > 0].sum())
    gross_loss = float(-rets[rets < 0].sum())
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else None

    avg_ret = float(rets.mean()) if trades else 0.0
    med_ret = float(np.median(rets)) if trades else 0.0

    return Metrics(
        n_trades=trades,
        wins=wins,
        win_rate=win_rate,
        total_return=total_return,
        mdd=mdd,
        profit_factor=pf,
        fill_rate=fill_rate,
        signals=signals,
        fills=fills,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        avg_return=avg_ret,
        median_return=med_ret,
    )

def score_metrics(
    m: Metrics,
    *,
    min_trades: int = 30,
    min_fill_rate: Optional[float] = None,
) -> float:
    """Score for parameter search.

    Goal: maximize win rate (primary).
    We add tiny tie-breakers to avoid pathological 'few-trades-only' solutions.
    """
    if m.n_trades <= 0:
        return -1e9

    score = float(m.win_rate)

    # Penalize too-few trades (soft)
    if m.n_trades < int(min_trades):
        score -= 0.2 * (1.0 - (m.n_trades / max(1.0, float(min_trades))))

    # Optional constraint: require some fill rate so it's tradable
    if min_fill_rate is not None:
        fr = float(m.fill_rate or 0.0)
        if fr < float(min_fill_rate):
            score -= 0.05

    # Light tie-breakers (do NOT dominate win rate)
    pf = float(m.profit_factor or 0.0)
    score += 0.01 * math.tanh(pf - 1.0)
    score += 0.005 * math.tanh((m.n_trades - 100) / 100.0)

    return float(score)

def grid_search_best_params(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    *,
    rsi_period: int,
    rsi_thresh: float,
    sma_fast_period: int,
    sma_trend_period: int,
    atr_period: int,
    entry_k_grid: Iterable[float],
    stop_mult_grid: Iterable[float],
    target_mult_grid: Iterable[float],
    fee_bps: float,
    use_trend_filter: bool,
    both_hit_rule: str,
    min_trades_for_score: int,
    min_fill_rate: Optional[float] = None,
    start: int = 0,
    end: Optional[int] = None,
) -> Dict[str, Any]:
    """Return best params and metrics on the given candle arrays."""
    o = np.asarray(o, dtype=float)
    h = np.asarray(h, dtype=float)
    l = np.asarray(l, dtype=float)
    c = np.asarray(c, dtype=float)

    best: Optional[Dict[str, Any]] = None
    best_score = -1e18

    for k in entry_k_grid:
        for sm in stop_mult_grid:
            for tm in target_mult_grid:
                m = simulate_daytrade_limit_long(
                    o, h, l, c,
                    rsi_period=rsi_period,
                    rsi_thresh=rsi_thresh,
                    sma_fast_period=sma_fast_period,
                    sma_trend_period=sma_trend_period,
                    atr_period=atr_period,
                    entry_k=float(k),
                    stop_mult=float(sm),
                    target_mult=float(tm),
                    fee_bps=float(fee_bps),
                    use_trend_filter=bool(use_trend_filter),
                    both_hit_rule=str(both_hit_rule),
                    start=start,
                    end=end,
                )
                sc = score_metrics(m, min_trades=min_trades_for_score, min_fill_rate=min_fill_rate)
                if sc > best_score:
                    best_score = sc
                    best = {
                        "entry_k": float(k),
                        "stop_mult": float(sm),
                        "target_mult": float(tm),
                        "score": float(sc),
                        "metrics": m,
                    }

    return best or {
        "entry_k": 0.75,
        "stop_mult": 2.0,
        "target_mult": 0.25,
        "score": -1e9,
        "metrics": Metrics(0, 0, 0.0, 0.0, 0.0, None, None, 0, 0, 0.0, 0.0, 0.0, 0.0),
    }
