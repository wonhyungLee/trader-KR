"""Active-Universe + daytrade backtester.

This runner mirrors the server-side autotrade selection flow:
1) Build daily selection snapshot (same filters/order as `server.py`)
2) Maintain active universe with the rule:
   - remove ONLY when exit reason == "상승추세(MA25) 조건 붕괴"
3) Build trade list: today's candidates first + remaining active universe
4) Execute next-day daytrade model:
   - entry limit (close - 0.75 * ATR)
   - intraday stop/target
   - EOD close fallback

Outputs:
- output_dir/trade_log.csv
- output_dir/equity_curve.csv
- output_dir/annual_metrics.csv
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.analyzer.backtest_runner import StrategyParams, load_strategy
from src.storage.sqlite_store import SQLiteStore
from src.utils.config import load_settings
from src.utils.project_root import ensure_repo_root
from stock_daytrade_engine.config import EngineConfig
from stock_daytrade_engine.indicators import atr_sma


FILTER_TOGGLE_KEYS = ("min_amount", "liquidity", "disparity")
FILTER_TOGGLE_PATH = Path("data/selection_filter_toggles.json")


@dataclass
class ActiveItem:
    code: str
    name: str
    market: str
    last_seen_date: str


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        num = float(value)
        if num != num or num in (float("inf"), float("-inf")):
            return None
        return num
    except Exception:
        return None


def _load_filter_toggles(path: Path = FILTER_TOGGLE_PATH) -> Dict[str, bool]:
    defaults = {k: True for k in FILTER_TOGGLE_KEYS}
    if not path.exists():
        return defaults
    try:
        payload = pd.read_json(path, typ="series").to_dict()
    except Exception:
        return defaults
    if not isinstance(payload, dict):
        return defaults
    out = defaults.copy()
    for key in FILTER_TOGGLE_KEYS:
        if key in payload:
            out[key] = bool(payload.get(key))
    return out


def _krx_tick_size(price: float) -> float:
    p = float(price)
    if p < 1000:
        return 1.0
    if p < 5000:
        return 5.0
    if p < 10000:
        return 10.0
    if p < 50000:
        return 50.0
    if p < 100000:
        return 100.0
    if p < 500000:
        return 500.0
    return 1000.0


def _quantize_to_tick(price: float, tick: float, mode: str = "down") -> float:
    if tick <= 0:
        return float(price)
    x = float(price) / float(tick)
    if mode == "up":
        return math.ceil(x) * tick
    if mode == "nearest":
        return round(x) * tick
    return math.floor(x) * tick


def _round_limit_price(price: float, quote: str = "KRW", side: str = "buy") -> float:
    q = str(quote or "").upper()
    s = str(side or "").lower()
    if q == "KRW":
        tick = _krx_tick_size(price)
        mode = "down" if s in ("buy", "sell") else "down"
        return float(_quantize_to_tick(price, tick, mode=mode))
    return float(round(price, 2))


def _selection_sector_of(row: pd.Series) -> str:
    value = row.get("sector")
    try:
        if value is None or pd.isna(value):
            return "UNKNOWN"
    except Exception:
        return "UNKNOWN"
    text = str(value).strip()
    return text or "UNKNOWN"


def _selection_pass_disparity(
    row: pd.Series,
    entry_mode: str,
    buy_kospi: float,
    buy_kosdaq: float,
) -> bool:
    market = str(row.get("market") or "KOSPI").upper()
    threshold = buy_kospi if "KOSPI" in market else buy_kosdaq
    disp = _safe_float(row.get("disparity"))
    if disp is None:
        return False
    if entry_mode == "trend_follow":
        ret3 = _safe_float(row.get("ret3")) or 0.0
        return disp >= threshold and ret3 >= 0
    return disp <= threshold


def _selection_pass_trend(row: pd.Series) -> bool:
    ma25 = _safe_float(row.get("ma25"))
    ma25_prev = _safe_float(row.get("ma25_prev"))
    if ma25 is None or ma25_prev is None:
        return False
    return ma25 > ma25_prev


def _build_selection_snapshot(
    day_df: pd.DataFrame,
    params: StrategyParams,
    toggles: Dict[str, bool],
    as_of_date: str,
) -> Dict[str, Any]:
    min_amount = float(getattr(params, "min_amount", 0) or 0)
    liquidity_rank = int(getattr(params, "liquidity_rank", 0) or 0)
    buy_kospi = float(getattr(params, "buy_kospi", 0) or 0)
    buy_kosdaq = float(getattr(params, "buy_kosdaq", 0) or 0)
    max_positions = int(getattr(params, "max_positions", 10) or 10)
    # Same as server selection snapshot: one per sector.
    max_per_sector = 1
    rank_mode = str(getattr(params, "rank_mode", "amount") or "amount").lower()
    entry_mode = str(getattr(params, "entry_mode", "mean_reversion") or "mean_reversion").lower()
    trend_filter = bool(getattr(params, "trend_ma25_rising", False))

    min_enabled = toggles.get("min_amount", True) is not False
    liq_enabled = toggles.get("liquidity", True) is not False
    disp_enabled = toggles.get("disparity", True) is not False

    total = int(len(day_df))
    stage_min = day_df.copy()
    if min_enabled and min_amount:
        stage_min = stage_min[stage_min["amount"] >= min_amount]

    stage_liq = stage_min.sort_values("amount", ascending=False)
    if liq_enabled and liquidity_rank:
        stage_liq = stage_liq.head(liquidity_rank)

    if disp_enabled:
        stage_disp = stage_liq[
            stage_liq.apply(
                lambda r: _selection_pass_disparity(r, entry_mode, buy_kospi, buy_kosdaq),
                axis=1,
            )
        ]
        if trend_filter:
            stage_disp = stage_disp[stage_disp["ma25_prev"].notna() & (stage_disp["ma25"] > stage_disp["ma25_prev"])]
    else:
        stage_disp = stage_liq

    ranked = stage_disp.copy()
    if rank_mode == "score":
        if entry_mode == "trend_follow":
            ranked["score"] = (
                ranked["disparity"].fillna(0).astype(float)
                + (0.8 * ranked["ret3"].fillna(0).astype(float))
                + (0.05 * np.log1p(ranked["amount"].fillna(0).astype(float).clip(lower=0)))
            )
        else:
            ranked["score"] = (
                (-ranked["disparity"].fillna(0).astype(float))
                + (0.8 * (-ranked["ret3"].fillna(0).astype(float)))
                + (0.05 * np.log1p(ranked["amount"].fillna(0).astype(float).clip(lower=0)))
            )
        ranked = ranked.sort_values("score", ascending=False)
    else:
        ranked = ranked.sort_values("amount", ascending=False)

    final_rows = []
    sector_counts: Dict[str, int] = {}
    for _, row in ranked.iterrows():
        sec = _selection_sector_of(row)
        if max_per_sector and sector_counts.get(sec, 0) >= max_per_sector:
            continue
        final_rows.append(row)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if len(final_rows) >= max_positions:
            break

    final = pd.DataFrame(final_rows) if final_rows else ranked.head(0).copy()
    if not final.empty:
        final["rank"] = range(1, len(final) + 1)

    candidates: List[Dict[str, Any]] = []
    for _, row in final.iterrows():
        candidates.append(
            {
                "rank": int(row.get("rank") or 0),
                "code": str(row.get("code")),
                "name": str(row.get("name") or ""),
                "market": str(row.get("market") or ""),
                "close": _safe_float(row.get("close")),
                "amount": _safe_float(row.get("amount")),
                "disparity": _safe_float(row.get("disparity")),
                "ma25": _safe_float(row.get("ma25")),
                "ma25_prev": _safe_float(row.get("ma25_prev")),
                "ret3": _safe_float(row.get("ret3")),
                "sector": _selection_sector_of(row),
            }
        )

    # server.py uses liquidity-stage rows for exit-reason diagnostics
    code_rows: Dict[str, pd.Series] = {}
    for _, row in stage_liq.iterrows():
        code = str(row.get("code") or "").strip()
        if code:
            code_rows[code] = row

    return {
        "date": as_of_date,
        "candidates": candidates,
        "summary": {
            "total": total,
            "final": len(candidates),
            "trend_filter": trend_filter,
            "rank_mode": rank_mode,
            "entry_mode": entry_mode,
            "max_positions": max_positions,
            "max_per_sector": max_per_sector,
            "buy_thresholds": {"kospi": buy_kospi, "kosdaq": buy_kosdaq},
        },
        "filter_toggles": toggles,
        "stage_sets": {
            "min_set": set(stage_min["code"].astype(str).tolist()),
            "liq_set": set(stage_liq["code"].astype(str).tolist()),
            "disp_set": set(stage_disp["code"].astype(str).tolist()),
            "code_rows": code_rows,
        },
    }


def _selection_exit_reason(
    code: str,
    prev_snapshot: Dict[str, Any],
    curr_snapshot: Optional[Dict[str, Any]] = None,
) -> str:
    if not code:
        return "이탈"
    code = str(code).strip()
    if not code:
        return "이탈"

    toggles = prev_snapshot.get("filter_toggles") if isinstance(prev_snapshot.get("filter_toggles"), dict) else {}
    min_enabled = toggles.get("min_amount", True) is not False
    liq_enabled = toggles.get("liquidity", True) is not False
    disp_enabled = toggles.get("disparity", True) is not False

    sets = prev_snapshot.get("stage_sets") if isinstance(prev_snapshot.get("stage_sets"), dict) else {}
    min_set = sets.get("min_set") if isinstance(sets.get("min_set"), set) else set()
    liq_set = sets.get("liq_set") if isinstance(sets.get("liq_set"), set) else set()
    disp_set = sets.get("disp_set") if isinstance(sets.get("disp_set"), set) else set()
    code_rows = sets.get("code_rows") if isinstance(sets.get("code_rows"), dict) else {}
    row = code_rows.get(code)

    summary = prev_snapshot.get("summary") if isinstance(prev_snapshot.get("summary"), dict) else {}
    entry_mode = str(summary.get("entry_mode", "mean_reversion")).lower()
    trend_filter = bool(summary.get("trend_filter"))
    thresholds = summary.get("buy_thresholds") if isinstance(summary.get("buy_thresholds"), dict) else {}
    buy_kospi = float(thresholds.get("kospi") or 0)
    buy_kosdaq = float(thresholds.get("kosdaq") or 0)

    if min_enabled and code not in min_set:
        return "거래대금 기준 미달"
    if liq_enabled and code not in liq_set:
        return "거래대금 상위 순위 밖(유동성 필터)"
    if disp_enabled and code not in disp_set:
        if row is not None:
            if not _selection_pass_disparity(row, entry_mode, buy_kospi, buy_kosdaq):
                return "괴리율(및 모멘텀) 조건 미충족"
            if trend_filter and not _selection_pass_trend(row):
                return "상승추세(MA25) 조건 붕괴"
        return "괴리율/추세 조건 미충족"
    if trend_filter and row is not None and not _selection_pass_trend(row):
        return "상승추세(MA25) 조건 붕괴"

    if isinstance(curr_snapshot, dict):
        curr_toggles = curr_snapshot.get("filter_toggles") if isinstance(curr_snapshot.get("filter_toggles"), dict) else {}
        curr_min_enabled = curr_toggles.get("min_amount", min_enabled) is not False
        curr_liq_enabled = curr_toggles.get("liquidity", liq_enabled) is not False
        curr_disp_enabled = curr_toggles.get("disparity", disp_enabled) is not False

        curr_sets = curr_snapshot.get("stage_sets") if isinstance(curr_snapshot.get("stage_sets"), dict) else {}
        curr_min_set = curr_sets.get("min_set") if isinstance(curr_sets.get("min_set"), set) else set()
        curr_liq_set = curr_sets.get("liq_set") if isinstance(curr_sets.get("liq_set"), set) else set()
        curr_disp_set = curr_sets.get("disp_set") if isinstance(curr_sets.get("disp_set"), set) else set()
        curr_code_rows = curr_sets.get("code_rows") if isinstance(curr_sets.get("code_rows"), dict) else {}
        curr_row = curr_code_rows.get(code, row)

        curr_summary = curr_snapshot.get("summary") if isinstance(curr_snapshot.get("summary"), dict) else {}
        curr_entry_mode = str(curr_summary.get("entry_mode", entry_mode)).lower()
        curr_trend_filter = bool(curr_summary.get("trend_filter", trend_filter))
        curr_thresholds = curr_summary.get("buy_thresholds") if isinstance(curr_summary.get("buy_thresholds"), dict) else {}
        curr_buy_kospi = float(curr_thresholds.get("kospi") or buy_kospi)
        curr_buy_kosdaq = float(curr_thresholds.get("kosdaq") or buy_kosdaq)

        if curr_min_enabled and code not in curr_min_set:
            return "거래대금 기준 미달"
        if curr_liq_enabled and code not in curr_liq_set:
            return "거래대금 상위 순위 밖(유동성 필터)"
        if curr_disp_enabled and code not in curr_disp_set:
            if curr_row is not None:
                if not _selection_pass_disparity(curr_row, curr_entry_mode, curr_buy_kospi, curr_buy_kosdaq):
                    return "괴리율(및 모멘텀) 조건 미충족"
                if curr_trend_filter and not _selection_pass_trend(curr_row):
                    return "상승추세(MA25) 조건 붕괴"
            return "괴리율/추세 조건 미충족"
        if curr_trend_filter and curr_row is not None and not _selection_pass_trend(curr_row):
            return "상승추세(MA25) 조건 붕괴"

    return "최종 후보 수/섹터 제한"


def _build_trade_rows(
    active: Dict[str, ActiveItem],
    today_candidates: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[str] = set()

    for row in today_candidates:
        code = str(row.get("code") or "").strip().zfill(6)
        if not code or code in seen:
            continue
        out.append(
            {
                "code": code,
                "name": str(row.get("name") or code).strip() or code,
                "market": str(row.get("market") or "KR").strip() or "KR",
            }
        )
        seen.add(code)
        if limit > 0 and len(out) >= limit:
            return out

    active_rows = sorted(
        active.values(),
        key=lambda r: (-int(r.last_seen_date.replace("-", "")), r.code),
    )
    for row in active_rows:
        code = str(row.code).zfill(6)
        if code in seen:
            continue
        out.append({"code": code, "name": row.name, "market": row.market})
        seen.add(code)
        if limit > 0 and len(out) >= limit:
            return out

    return out


def _load_universe(conn: Any) -> pd.DataFrame:
    universe = pd.read_sql_query("SELECT code, name, market, group_name FROM universe_members", conn)
    universe["code"] = universe["code"].astype(str).str.zfill(6)
    try:
        sector_df = pd.read_sql_query("SELECT code, sector_name, industry_name FROM sector_map", conn)
        if not sector_df.empty:
            sector_df["code"] = sector_df["code"].astype(str).str.zfill(6)
            sector_df["sector"] = sector_df["industry_name"].fillna(sector_df["sector_name"])
            universe = universe.merge(sector_df[["code", "sector"]], on="code", how="left")
        else:
            universe["sector"] = None
    except Exception:
        universe["sector"] = None
    universe["sector"] = universe["sector"].fillna(universe["group_name"]).fillna("UNKNOWN")
    return universe[["code", "name", "market", "sector"]]


def _load_prices(
    conn: Any,
    universe_codes: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    placeholder = ",".join("?" * len(universe_codes))
    sql = f"""
        SELECT date, code, open, high, low, close, amount, ma25, disparity
        FROM daily_price
        WHERE code IN ({placeholder})
          AND date >= ?
          AND date <= ?
        ORDER BY code ASC, date ASC
    """
    params = tuple(universe_codes + [str(start_date), str(end_date)])
    df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return df
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "amount", "ma25", "disparity"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _prepare_features(prices: pd.DataFrame, cfg: EngineConfig) -> pd.DataFrame:
    if prices.empty:
        return prices
    prices = prices.sort_values(["code", "date"]).copy()
    prices["bar_count"] = prices.groupby("code").cumcount() + 1
    prices["ma25_prev"] = prices.groupby("code")["ma25"].shift(1)
    prices["ret3"] = prices.groupby("code")["close"].pct_change(3)
    prices["atr"] = np.nan

    series_list = []
    for _code, group in prices.groupby("code", sort=False):
        arr = atr_sma(
            group["high"].to_numpy(dtype=float),
            group["low"].to_numpy(dtype=float),
            group["close"].to_numpy(dtype=float),
            int(cfg.atr_period),
        )
        series_list.append(pd.Series(arr, index=group.index))
    if series_list:
        prices["atr"] = pd.concat(series_list).sort_index()
    return prices


def _get_row(price_index: pd.DataFrame, date_key: pd.Timestamp, code: str) -> Optional[pd.Series]:
    try:
        row = price_index.loc[(date_key, code)]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        return row.iloc[-1]
    return row


def run_backtest_active_universe(
    store: SQLiteStore,
    params: StrategyParams,
    *,
    start_date: str,
    end_date: str,
    warmup_days: int,
    max_codes: int,
    output_dir: Path,
) -> Dict[str, Any]:
    cfg = EngineConfig(db_path=str(Path("data/market_data.db").resolve()), table="daily_price")
    fee = float(cfg.fee_bps) / 10000.0
    entry_mult = 1.0 + fee
    exit_mult = 1.0 - fee

    conn = store.conn
    universe = _load_universe(conn)
    if universe.empty:
        raise SystemExit("universe_members is empty")

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    warmup_start = (start_ts - timedelta(days=int(warmup_days))).strftime("%Y-%m-%d")

    prices = _load_prices(
        conn,
        universe_codes=universe["code"].tolist(),
        start_date=warmup_start,
        end_date=end_date,
    )
    if prices.empty:
        raise SystemExit("daily_price rows are empty for the requested period")

    prices = _prepare_features(prices, cfg)
    prices = prices.merge(universe, on="code", how="left")
    prices["name"] = prices["name"].fillna(prices["code"])
    prices["market"] = prices["market"].fillna("KR")
    prices["sector"] = prices["sector"].fillna("UNKNOWN")

    all_dates = sorted(prices["date"].drop_duplicates().tolist())
    if len(all_dates) < 2:
        raise SystemExit("not enough dates to backtest")

    rows_by_date = {d: g.copy() for d, g in prices.groupby("date")}
    price_index = prices.set_index(["date", "code"]).sort_index()

    toggles = _load_filter_toggles()
    active: Dict[str, ActiveItem] = {}
    prev_snapshot: Optional[Dict[str, Any]] = None

    cash = float(getattr(params, "initial_cash", 10_000_000) or 10_000_000)
    equity_rows: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    active_events: List[Dict[str, Any]] = []

    for i in range(len(all_dates) - 1):
        d = all_dates[i]
        nd = all_dates[i + 1]
        if d > end_ts:
            break
        day_df = rows_by_date.get(d)
        if day_df is None or day_df.empty:
            continue

        snap = _build_selection_snapshot(day_df, params, toggles, d.strftime("%Y-%m-%d"))
        today_candidates = [c for c in snap["candidates"] if str(c.get("code") or "").strip()]
        today_codes = set(str(c["code"]).zfill(6) for c in today_candidates)

        if prev_snapshot is None:
            for row in today_candidates:
                code = str(row.get("code") or "").zfill(6)
                if not code:
                    continue
                active[code] = ActiveItem(
                    code=code,
                    name=str(row.get("name") or code),
                    market=str(row.get("market") or "KR"),
                    last_seen_date=d.strftime("%Y-%m-%d"),
                )
        else:
            prev_codes = set(str(c["code"]).zfill(6) for c in prev_snapshot["candidates"])
            entered = sorted(today_codes - prev_codes)
            exited = sorted(prev_codes - today_codes)
            by_code = {str(r.get("code")).zfill(6): r for r in today_candidates}

            for code in entered:
                row = by_code.get(code, {})
                active[code] = ActiveItem(
                    code=code,
                    name=str(row.get("name") or code),
                    market=str(row.get("market") or "KR"),
                    last_seen_date=d.strftime("%Y-%m-%d"),
                )
                active_events.append(
                    {"date": d.strftime("%Y-%m-%d"), "code": code, "event": "entered", "reason": ""}
                )

            for code in exited:
                reason = _selection_exit_reason(code, prev_snapshot, snap)
                if reason == "상승추세(MA25) 조건 붕괴":
                    if code in active:
                        active.pop(code, None)
                    active_events.append(
                        {"date": d.strftime("%Y-%m-%d"), "code": code, "event": "removed", "reason": reason}
                    )

        prev_snapshot = snap
        trade_rows = _build_trade_rows(active=active, today_candidates=today_candidates, limit=max_codes)

        # Warm-up phase for active-universe only; trading starts at start_date.
        if d < start_ts:
            continue
        if nd > end_ts:
            break

        day_start_cash = cash
        n_orders = len(trade_rows)
        budget_per_order = (day_start_cash / n_orders) if n_orders > 0 else 0.0
        day_pnl = 0.0
        filled_count = 0

        for row in trade_rows:
            code = str(row.get("code") or "").zfill(6)
            plan_row = _get_row(price_index, d, code)
            exec_row = _get_row(price_index, nd, code)
            if plan_row is None or exec_row is None:
                continue

            close = _safe_float(plan_row.get("close"))
            atr = _safe_float(plan_row.get("atr"))
            bar_count = int(plan_row.get("bar_count") or 0)
            if close is None or atr is None or close <= 0 or atr <= 0:
                continue
            if bar_count < int(cfg.min_bars_for_indicators):
                continue

            entry = _round_limit_price(close - (0.75 * atr), quote="KRW", side="buy")
            stop = _round_limit_price(entry - (2.0 * atr), quote="KRW", side="sell")
            target = _round_limit_price(entry + (0.25 * atr), quote="KRW", side="sell")
            if entry <= 0:
                continue

            low = _safe_float(exec_row.get("low"))
            high = _safe_float(exec_row.get("high"))
            eod_close = _safe_float(exec_row.get("close"))
            if low is None or high is None or eod_close is None:
                continue

            # Trigger: buy limit lte entry, filled at entry.
            if low > entry:
                continue

            entry_cost_per_share = entry * entry_mult
            qty = int(budget_per_order // entry_cost_per_share) if budget_per_order > 0 else 0
            if qty <= 0:
                continue

            hit_stop = low <= stop
            hit_target = high >= target
            if str(cfg.both_hit_rule).lower() == "target_first":
                exit_px = target if hit_target else (stop if hit_stop else eod_close)
            else:
                exit_px = stop if hit_stop else (target if hit_target else eod_close)

            trade_ret = (exit_px * exit_mult) / (entry * entry_mult) - 1.0
            pnl = qty * ((exit_px * exit_mult) - (entry * entry_mult))

            day_pnl += pnl
            filled_count += 1
            trades.append(
                {
                    "plan_date": d.strftime("%Y-%m-%d"),
                    "exec_date": nd.strftime("%Y-%m-%d"),
                    "code": code,
                    "name": str(row.get("name") or code),
                    "entry_price": float(entry),
                    "stop_price": float(stop),
                    "target_price": float(target),
                    "exit_price": float(exit_px),
                    "qty": int(qty),
                    "ret": float(trade_ret),
                    "pnl": float(pnl),
                    "win": int(pnl > 0),
                }
            )

        cash += day_pnl
        equity_rows.append(
            {
                "date": nd.strftime("%Y-%m-%d"),
                "equity": float(cash),
                "day_pnl": float(day_pnl),
                "planned_orders": int(n_orders),
                "filled_trades": int(filled_count),
                "today_candidates": int(len(today_candidates)),
                "active_universe": int(len(active)),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    active_events_df = pd.DataFrame(active_events)

    trades_df.to_csv(output_dir / "trade_log.csv", index=False)
    equity_df.to_csv(output_dir / "equity_curve.csv", index=False)
    active_events_df.to_csv(output_dir / "active_events.csv", index=False)

    annual = compute_annual_metrics(
        equity_df=equity_df,
        trades_df=trades_df,
        initial_cash=float(getattr(params, "initial_cash", 10_000_000) or 10_000_000),
    )
    annual.to_csv(output_dir / "annual_metrics.csv", index=False)

    print(
        f"saved output_dir={output_dir} "
        f"trades={len(trades_df)} equity_rows={len(equity_df)} annual_rows={len(annual)}"
    )
    if not annual.empty:
        show = annual.copy()
        show["win_rate_pct"] = (show["win_rate"] * 100).round(2)
        show["return_pct"] = (show["return"] * 100).round(2)
        show["mdd_pct"] = (show["mdd"] * 100).round(2)
        print(show[["year", "trade_count", "win_rate_pct", "return_pct", "mdd_pct"]].to_string(index=False))

    return {
        "trades": len(trades_df),
        "equity_rows": len(equity_df),
        "annual_rows": len(annual),
        "output_dir": str(output_dir),
    }


def compute_annual_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    if equity_df.empty:
        return pd.DataFrame(columns=["year", "trade_count", "win_rate", "return", "mdd"])

    eq = equity_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date")
    years = sorted(eq["date"].dt.year.unique().tolist())

    rows: List[Dict[str, Any]] = []
    for year in years:
        y_start = pd.Timestamp(year=year, month=1, day=1)
        y_end = pd.Timestamp(year=year, month=12, day=31)
        eq_y = eq[(eq["date"] >= y_start) & (eq["date"] <= y_end)]
        if eq_y.empty:
            continue

        prev = eq[eq["date"] < y_start]
        base_equity = float(prev.iloc[-1]["equity"]) if not prev.empty else float(initial_cash)
        end_equity = float(eq_y.iloc[-1]["equity"])
        year_return = (end_equity / base_equity - 1.0) if base_equity else 0.0

        path = np.array([base_equity] + eq_y["equity"].astype(float).tolist(), dtype=float)
        peaks = np.maximum.accumulate(path)
        dd = (peaks - path) / peaks
        mdd = float(np.max(dd)) if len(dd) else 0.0

        if trades_df.empty:
            trade_count = 0
            win_rate = 0.0
        else:
            t = trades_df[trades_df["exec_date"].astype(str).str.startswith(str(year))]
            trade_count = int(len(t))
            win_rate = float((t["pnl"] > 0).mean()) if trade_count else 0.0

        rows.append(
            {
                "year": int(year),
                "trade_count": int(trade_count),
                "win_rate": float(win_rate),
                "return": float(year_return),
                "mdd": float(mdd),
            }
        )

    return pd.DataFrame(rows)


def _resolve_default_dates(conn: Any) -> tuple[str, str]:
    row = conn.execute("SELECT MIN(date), MAX(date) FROM daily_price").fetchone()
    if not row or not row[0] or not row[1]:
        raise SystemExit("daily_price has no date range")
    return str(row[0]), str(row[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Active-Universe + daytrade annual backtester")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--warmup-days", type=int, default=365, help="active-universe warmup days")
    parser.add_argument(
        "--max-codes",
        type=int,
        default=max(0, int(os.getenv("AUTOTRADE_MAX_SELECTION_CODES", "20") or 20)),
        help="daily max trade list size (default: AUTOTRADE_MAX_SELECTION_CODES or 20)",
    )
    parser.add_argument("--output-dir", default="data/backtest_active_universe_daytrade")
    return parser.parse_args()


def main() -> None:
    ensure_repo_root()
    args = parse_args()
    settings = load_settings()
    params = load_strategy(settings)
    store = SQLiteStore(settings.get("database", {}).get("path", "data/market_data.db"))

    min_date, max_date = _resolve_default_dates(store.conn)
    start_date = args.start_date or min_date
    end_date = args.end_date or max_date

    run_backtest_active_universe(
        store,
        params,
        start_date=str(start_date),
        end_date=str(end_date),
        warmup_days=max(30, int(args.warmup_days)),
        max_codes=max(0, int(args.max_codes)),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
