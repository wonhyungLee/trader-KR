from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import EngineConfig
from .db import fetch_ohlc, list_codes
from .indicators import rolling_sma, rsi_sma, atr_sma
from .backtester import grid_search_best_params, Metrics

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _ease_score_long(close: float, sma_fast: float, sma_trend: float, rsi: float, rsi_thresh: float, use_trend_filter: bool) -> Tuple[float, Dict[str, Any]]:
    """How 'close' is to the ideal trigger (100 means trigger now)."""
    trend_ok = (close > sma_trend) if use_trend_filter else True
    trigger = trend_ok and (close < sma_fast) and (rsi <= rsi_thresh)

    dist_sma_fast = max(0.0, (close - sma_fast) / close) * 100.0
    dist_rsi = max(0.0, (rsi - rsi_thresh) / max(1e-9, rsi_thresh)) * 100.0
    dist_trend = max(0.0, (sma_trend - close) / close) * 100.0 if use_trend_filter else 0.0

    base = 100.0 if trigger else 100.0 - (dist_sma_fast * 200.0 + dist_rsi * 1.0 + dist_trend * 300.0)
    return clamp(base, 0.0, 110.0), {
        "trigger_now": bool(trigger),
        "trend_ok": bool(trend_ok),
        "distance_close_to_sma_fast_pct": round(dist_sma_fast, 4),
        "distance_close_to_sma_trend_pct": round(dist_trend, 4),
        "distance_rsi_to_threshold": round(max(0.0, rsi - rsi_thresh), 4),
    }

def compute_latest_snapshot(code: str, cfg: EngineConfig) -> Optional[Dict[str, Any]]:
    dates, o, h, l, c = fetch_ohlc(cfg.db_path, code, table=cfg.table, limit=cfg.min_bars_for_indicators, desc=True, with_date=True)
    if len(c) < cfg.min_bars_for_indicators:
        return None

    o = np.asarray(o, dtype=float)
    h = np.asarray(h, dtype=float)
    l = np.asarray(l, dtype=float)
    c = np.asarray(c, dtype=float)

    sma_fast = rolling_sma(c, cfg.sma_fast)
    sma_trend = rolling_sma(c, cfg.sma_trend)
    rsi = rsi_sma(c, cfg.rsi_period)
    atr = atr_sma(h, l, c, cfg.atr_period)

    i = len(c) - 1
    if any(math.isnan(float(x)) for x in (sma_fast[i], sma_trend[i], rsi[i], atr[i])):
        return None

    close = float(c[i])
    atr_v = float(atr[i])
    atr_pct = (atr_v / close * 100.0) if close else float("nan")

    ease, detail = _ease_score_long(
        close=close,
        sma_fast=float(sma_fast[i]),
        sma_trend=float(sma_trend[i]),
        rsi=float(rsi[i]),
        rsi_thresh=float(cfg.rsi_thresh),
        use_trend_filter=bool(cfg.use_trend_filter),
    )

    return {
        "code": code,
        "date": dates[-1],
        "open": float(o[i]),
        "high": float(h[i]),
        "low": float(l[i]),
        "close": close,
        "sma_fast": float(sma_fast[i]),
        "sma_trend": float(sma_trend[i]),
        "rsi": float(rsi[i]),
        "atr": atr_v,
        "atr_pct": float(atr_pct),
        "entry_ease_score": round(float(ease), 2),
        **detail,
    }

def build_plan(snapshot: Dict[str, Any], params: Dict[str, Any], cfg: EngineConfig, risk_pct: Optional[float] = None) -> Dict[str, Any]:
    """Build *next-day* day-trade bracket plan derived from today's close."""
    risk_pct = float(cfg.risk_pct_default if risk_pct is None else risk_pct)

    close = float(snapshot["close"])
    atr = float(snapshot["atr"])
    k = float(params["entry_k"])
    sm = float(params["stop_mult"])
    tm = float(params["target_mult"])

    entry = close - k * atr
    stop = entry - sm * atr
    target = entry + tm * atr

    stop_pct = abs(entry - stop) / entry * 100.0 if entry else 0.0
    tp_pct = abs(target - entry) / entry * 100.0 if entry else 0.0
    rr = (target - entry) / (entry - stop) if entry > stop else None

    # Position sizing helper (not mandatory)
    max_pos_pct = (risk_pct / stop_pct) if stop_pct > 0 else None

    return {
        "side": "long",
        "entry_price": round(entry, 4),
        "stop_price": round(stop, 4),
        "target_price": round(target, 4),
        "exit_rule": "익절/손절 미체결 시 당일 종가 청산(EOD)",
        "params": {"entry_k": k, "stop_mult": sm, "target_mult": tm},
        "risk_pct": risk_pct,
        "stop_distance_pct": round(stop_pct, 3),
        "take_profit_distance_pct": round(tp_pct, 3),
        "reward_risk_to_target": round(float(rr), 3) if rr is not None else None,
        "max_position_pct_by_risk": round(max_pos_pct, 3) if max_pos_pct is not None else None,
        "assumptions": {
            "fill": "다음날 저가가 entry_price 이하이면 entry_price에 체결(보수적)",
            "both_hit_rule": cfg.both_hit_rule,
            "fees": f"entry*(1+{cfg.fee_bps}bps), exit*(1-{cfg.fee_bps}bps)",
        },
    }

def recommend_code(
    code: str,
    cfg: EngineConfig,
    *,
    optimize: bool = True,
    optimize_lookback: Optional[int] = None,
    risk_pct: Optional[float] = None,
) -> Dict[str, Any]:
    snap = compute_latest_snapshot(code, cfg)
    if not snap:
        return {"ok": False, "code": code, "error": "not_enough_data_or_indicators"}

    best = None
    if optimize:
        lookback = int(optimize_lookback or cfg.optimize_lookback_bars)
        dates, o, h, l, c = fetch_ohlc(cfg.db_path, code, table=cfg.table, limit=lookback, desc=True, with_date=True)
        if len(c) >= cfg.min_bars_for_indicators:
            o = np.asarray(o, dtype=float)
            h = np.asarray(h, dtype=float)
            l = np.asarray(l, dtype=float)
            c = np.asarray(c, dtype=float)
            best = grid_search_best_params(
                o, h, l, c,
                rsi_period=cfg.rsi_period,
                rsi_thresh=cfg.rsi_thresh,
                sma_fast_period=cfg.sma_fast,
                sma_trend_period=cfg.sma_trend,
                atr_period=cfg.atr_period,
                entry_k_grid=cfg.entry_k_grid,
                stop_mult_grid=cfg.stop_mult_grid,
                target_mult_grid=cfg.target_mult_grid,
                fee_bps=cfg.fee_bps,
                use_trend_filter=cfg.use_trend_filter,
                both_hit_rule=cfg.both_hit_rule,
                min_trades_for_score=cfg.min_trades_for_score,
                min_fill_rate=cfg.min_fill_rate,
                start=0,
                end=len(c),  # full lookback window
            )

    if not best:
        best = {"entry_k": 0.75, "stop_mult": 2.0, "target_mult": 0.25, "score": None, "metrics": None}

    plan = build_plan(snap, best, cfg, risk_pct=risk_pct)

    # Confidence score (0~100): ease + recent backtest win rate
    bt_wr = None
    bt_trades = None
    if best.get("metrics") is not None:
        m: Metrics = best["metrics"]
        bt_wr = float(m.win_rate)
        bt_trades = int(m.n_trades)

    ease = float(snap["entry_ease_score"]) / 110.0  # 0..1
    wr_norm = float(bt_wr) if bt_wr is not None else 0.5
    conf = clamp(ease * 0.55 + wr_norm * 0.45, 0.0, 1.0) * 100.0

    status = "ready" if snap.get("trigger_now") else "wait"

    return {
        "ok": True,
        "code": code,
        "status": status,
        "confidence": round(conf, 1),
        "snapshot": snap,
        "best_params": {
            "entry_k": best.get("entry_k"),
            "stop_mult": best.get("stop_mult"),
            "target_mult": best.get("target_mult"),
            "score": best.get("score"),
            "metrics": best.get("metrics").__dict__ if best.get("metrics") is not None else None,
        },
        "plan": plan,
    }

def scan_ready(
    cfg: EngineConfig,
    *,
    min_rows: int = 3000,
    limit: int = 20,
    optimize: bool = True,
) -> List[Dict[str, Any]]:
    codes = list_codes(cfg.db_path, table=cfg.table, min_rows=min_rows)
    out: List[Dict[str, Any]] = []
    for code, _n in codes:
        snap = compute_latest_snapshot(code, cfg)
        if not snap:
            continue
        if not snap.get("trigger_now"):
            continue
        rec = recommend_code(code, cfg, optimize=optimize)
        if rec.get("ok"):
            out.append(rec)

    out.sort(
        key=lambda x: (x.get("status") == "ready", x.get("confidence", 0.0), (x.get("best_params") or {}).get("score", 0.0)),
        reverse=True,
    )
    return out[: int(limit)]
