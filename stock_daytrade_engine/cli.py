from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np

from .config import EngineConfig
from .db import fetch_ohlc, list_codes
from .recommender import recommend_code, scan_ready
from .backtester import grid_search_best_params, simulate_daytrade_limit_long

def _p(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))

def cmd_recommend(args: argparse.Namespace) -> None:
    cfg = EngineConfig(db_path=args.db, table=args.table)
    out = recommend_code(args.code, cfg, optimize=not args.no_optimize, optimize_lookback=args.lookback, risk_pct=args.risk_pct)
    _p(out)

def cmd_scan(args: argparse.Namespace) -> None:
    cfg = EngineConfig(db_path=args.db, table=args.table)
    out = scan_ready(cfg, min_rows=args.min_rows, limit=args.limit, optimize=not args.no_optimize)
    _p(out)

def cmd_backtest_code(args: argparse.Namespace) -> None:
    cfg = EngineConfig(db_path=args.db, table=args.table)

    dates, o, h, l, c = fetch_ohlc(cfg.db_path, args.code, table=cfg.table, limit=args.lookback, desc=True, with_date=True)
    if len(c) < cfg.min_bars_for_indicators:
        _p({"ok": False, "error": "not_enough_data", "min_required": cfg.min_bars_for_indicators, "n": len(c)})
        return

    o = np.asarray(o, dtype=float)
    h = np.asarray(h, dtype=float)
    l = np.asarray(l, dtype=float)
    c = np.asarray(c, dtype=float)

    n = len(c)
    split = int(n * float(args.split))
    split = max(cfg.min_bars_for_indicators, min(split, n - 2))

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
        end=split,   # train only
    )

    m_test = simulate_daytrade_limit_long(
        o, h, l, c,
        rsi_period=cfg.rsi_period,
        rsi_thresh=cfg.rsi_thresh,
        sma_fast_period=cfg.sma_fast,
        sma_trend_period=cfg.sma_trend,
        atr_period=cfg.atr_period,
        entry_k=float(best["entry_k"]),
        stop_mult=float(best["stop_mult"]),
        target_mult=float(best["target_mult"]),
        fee_bps=cfg.fee_bps,
        use_trend_filter=cfg.use_trend_filter,
        both_hit_rule=cfg.both_hit_rule,
        start=max(0, split - 1),
        end=n,       # test
    )

    out: Dict[str, Any] = {
        "ok": True,
        "code": args.code,
        "lookback": int(args.lookback),
        "split": float(args.split),
        "train_end_date": dates[split - 1] if 0 <= split - 1 < len(dates) else None,
        "test_start_date": dates[split] if 0 <= split < len(dates) else None,
        "best_params": {
            "entry_k": best["entry_k"],
            "stop_mult": best["stop_mult"],
            "target_mult": best["target_mult"],
            "score": best["score"],
        },
        "train_metrics": best["metrics"].__dict__ if best.get("metrics") is not None else None,
        "test_metrics": m_test.__dict__,
    }
    _p(out)

def cmd_backtest_universe(args: argparse.Namespace) -> None:
    cfg = EngineConfig(db_path=args.db, table=args.table)
    codes = list_codes(cfg.db_path, table=cfg.table, min_rows=args.min_rows)

    rows: List[Dict[str, Any]] = []
    total_trades = 0
    total_wins = 0

    for code, _n in codes:
        dates, o, h, l, c = fetch_ohlc(cfg.db_path, code, table=cfg.table, limit=args.lookback, desc=True, with_date=True)
        if len(c) < cfg.min_bars_for_indicators:
            continue

        o = np.asarray(o, dtype=float)
        h = np.asarray(h, dtype=float)
        l = np.asarray(l, dtype=float)
        c = np.asarray(c, dtype=float)

        n = len(c)
        split = int(n * float(args.split))
        split = max(cfg.min_bars_for_indicators, min(split, n - 2))

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
            end=split,
        )

        m_test = simulate_daytrade_limit_long(
            o, h, l, c,
            rsi_period=cfg.rsi_period,
            rsi_thresh=cfg.rsi_thresh,
            sma_fast_period=cfg.sma_fast,
            sma_trend_period=cfg.sma_trend,
            atr_period=cfg.atr_period,
            entry_k=float(best["entry_k"]),
            stop_mult=float(best["stop_mult"]),
            target_mult=float(best["target_mult"]),
            fee_bps=cfg.fee_bps,
            use_trend_filter=cfg.use_trend_filter,
            both_hit_rule=cfg.both_hit_rule,
            start=max(0, split - 1),
            end=n,
        )

        total_trades += int(m_test.n_trades)
        total_wins += int(m_test.wins)

        rows.append({
            "code": code,
            "train_win_rate": float(best["metrics"].win_rate) if best.get("metrics") else None,
            "train_trades": int(best["metrics"].n_trades) if best.get("metrics") else None,
            "test_win_rate": float(m_test.win_rate),
            "test_trades": int(m_test.n_trades),
            "entry_k": best["entry_k"],
            "stop_mult": best["stop_mult"],
            "target_mult": best["target_mult"],
        })

    out = {
        "ok": True,
        "n_codes": len(rows),
        "lookback": int(args.lookback),
        "split": float(args.split),
        "weighted_test_win_rate": (total_wins / total_trades) if total_trades else None,
        "total_test_trades": int(total_trades),
        "rows": rows[: int(args.limit_rows)],
    }
    _p(out)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stock_daytrade_engine", description="LONG-only day-trade recommendation engine (daily bars).")
    p.add_argument("--db", default="market_data.db", help="SQLite DB path (default: market_data.db)")
    p.add_argument("--table", default="daily_price", help="Price table (default: daily_price)")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("recommend", help="Recommend next-day entry/stop/target for one code")
    p_rec.add_argument("--code", required=True)
    p_rec.add_argument("--lookback", type=int, default=None, help="Optimization lookback bars (default: config)")
    p_rec.add_argument("--risk-pct", type=float, default=None, help="Risk percent per trade for sizing helper (default: config)")
    p_rec.add_argument("--no-optimize", action="store_true", help="Disable per-code grid search")
    p_rec.set_defaults(func=cmd_recommend)

    p_scan = sub.add_parser("scan", help="Scan READY signals (trigger_now) on latest date")
    p_scan.add_argument("--min-rows", type=int, default=3000)
    p_scan.add_argument("--limit", type=int, default=20)
    p_scan.add_argument("--no-optimize", action="store_true")
    p_scan.set_defaults(func=cmd_scan)

    p_bt = sub.add_parser("backtest-code", help="Walk-forward backtest for one code (train->test)")
    p_bt.add_argument("--code", required=True)
    p_bt.add_argument("--lookback", type=int, default=3000)
    p_bt.add_argument("--split", type=float, default=0.7, help="Train split ratio (default 0.7)")
    p_bt.set_defaults(func=cmd_backtest_code)

    p_btu = sub.add_parser("backtest-universe", help="Walk-forward backtest for many codes (slow)")
    p_btu.add_argument("--min-rows", type=int, default=3000)
    p_btu.add_argument("--lookback", type=int, default=3000)
    p_btu.add_argument("--split", type=float, default=0.7)
    p_btu.add_argument("--limit-rows", type=int, default=50, help="How many per-code rows to include in output")
    p_btu.set_defaults(func=cmd_backtest_universe)

    return p

def main() -> None:
    p = build_parser()
    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
