"""Next-Open 백테스트 러너.

- 전일 종가에서 신호 생성, 익일 시가 체결 가정
- 전략 파라미터는 config/settings.yaml 또는 config/strategy.yaml을 사용
- 결과: data/trade_log.csv, data/equity_curve.csv
"""

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.storage.sqlite_store import SQLiteStore
from src.utils.config import load_settings, load_yaml
from src.utils.project_root import ensure_repo_root


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


def load_strategy(settings: Dict) -> StrategyParams:
    strat_file = Path("config/strategy.yaml")
    strat = load_yaml(strat_file) if strat_file.exists() else settings.get("strategy", {})

    buy_cfg = strat.get("buy", {}) or {}
    sell_cfg = strat.get("sell", {}) or {}
    pos_cfg = strat.get("position", {}) or {}
    trend_cfg = (buy_cfg.get("trend_filter", {}) or {})
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


def select_universe(prices: pd.DataFrame, stock_info: pd.DataFrame, params: StrategyParams) -> List[str]:
    latest = prices.sort_values("date").groupby("code").tail(1)
    merged = latest.merge(stock_info[["code", "market"]], on="code", how="left")
    merged = merged.sort_values("amount", ascending=False)
    merged = merged[merged["amount"] >= params.min_amount]
    return merged.head(params.liquidity_rank)["code"].tolist()


def run_backtest(store: SQLiteStore, params: StrategyParams, output_dir: Path = Path("data"), start_date: str | None = None, end_date: str | None = None, codes: List[str] | None = None):
    """Next-Open 백테스트.

    - 신호 생성: t일 종가 기준(당일 데이터로) 매수 신호 생성
    - 진입 체결: t+1일 시가
    - 청산 체결: t+1일 시가 (t일 종료 시점에 청산 조건 충족 시)
    - 종목 선별(신호) 지표와, 실제 매매(트레이드) 승률을 분리해서 리포트로 저장
      * selection_report.csv: '선별된 종목'의 다음날 성과(기본: 익일 시가->종가)
      * trade_log.csv: 실제 진입/청산이 발생한 트레이드 로그
    """
    prices = store.load_all_prices()
    if prices.empty:
        raise SystemExit("daily_price 가 비어있습니다. 먼저 데이터를 적재하세요.")

    stock_info = pd.read_sql_query("SELECT code, market, group_name FROM universe_members", store.conn)
    if stock_info.empty:
        raise SystemExit("universe_members is empty. Run universe_loader first.")
    market_map = dict(zip(stock_info["code"], stock_info["market"]))
    # sector/industry mapping: prefer sector_map table if available (more granular than universe_members.group_name)
    sector_map = dict(zip(stock_info["code"], stock_info["group_name"]))
    try:
        sm = pd.read_sql_query("SELECT code, sector_name, industry_name FROM sector_map", store.conn)
        if not sm.empty and "sector_name" in sm.columns:
            sm = sm.copy()
            sm["code"] = sm["code"].astype(str).str.zfill(6)
            if "industry_name" in sm.columns:
                sm["sector"] = sm["industry_name"].fillna(sm["sector_name"])
            else:
                sm["sector"] = sm["sector_name"]
            sector_map.update(dict(zip(sm["code"], sm["sector"])))
    except Exception:
        pass
    universe_codes = set(stock_info["code"].tolist())
    if universe_codes:
        prices = prices[prices["code"].isin(universe_codes)]

    # optional code/date filter (for faster focused backtests)
    if codes:
        code_set = set([str(c).zfill(6) for c in codes])
        prices = prices[prices["code"].astype(str).str.zfill(6).isin(code_set)]
    prices["date"] = pd.to_datetime(prices["date"])
    if start_date:
        sd = pd.to_datetime(start_date) - timedelta(days=40)  # warm-up buffer for indicators
        prices = prices[prices["date"] >= sd]
    if end_date:
        ed = pd.to_datetime(end_date)
        prices = prices[prices["date"] <= ed]
    prices = prices.sort_values(["code", "date"]).copy()
    # trend filter용 (ma25 상승)
    prices["ma25_prev"] = prices.groupby("code")["ma25"].shift(1)
    # short-term dump metric (3-day return)
    prices["ret3"] = prices.groupby("code")["close"].pct_change(3)

    # 날짜 리스트
    dates = sorted(prices["date"].unique())
    if len(dates) < 2:
        raise SystemExit("가격 데이터가 부족합니다.")

    cash = float(getattr(params, "initial_cash", 10_000_000) or 10_000_000)
    equity_curve: List[Dict] = []
    trades: List[Dict] = []
    selection_rows: List[Dict] = []
    positions: Dict[str, Dict] = {}

    # 날짜별 슬라이스 접근을 빠르게 하기 위해 groupby 결과를 dict로
    grouped = {code: df.reset_index(drop=True) for code, df in prices.groupby("code")}

    def get_row(code: str, d: pd.Timestamp):
        dfc = grouped.get(code)
        if dfc is None: return None
        row = dfc.loc[dfc["date"] == d]
        if row.empty:
            return None
        return row.iloc[0]

    def buy_threshold(code: str) -> float:
        market = market_map.get(code, "KOSPI") or "KOSPI"
        return params.buy_kospi if "KOSPI" in market else params.buy_kosdaq

    def sector_of(code: str) -> str:
        return sector_map.get(code) or "UNKNOWN"

    # 본 루프: i는 today index, i+1은 next (체결가로 사용)
    entry_mode = str(getattr(params, "entry_mode", "mean_reversion") or "mean_reversion").lower()

    for i in range(len(dates) - 1):
        d = dates[i]
        nd = dates[i + 1]

        # --- 1) 기존 포지션 청산 판단 (t일 기준 -> t+1 시가 청산) ---
        for code in list(positions.keys()):
            today = get_row(code, d)
            next_row = get_row(code, nd)
            if today is None or next_row is None:
                continue

            pos = positions[code]
            pos["hold_days"] += 1
            exit_price = float(next_row["open"])
            if pos["avg_price"] <= 0 or exit_price <= 0:
                continue

            ret = (exit_price / pos["avg_price"]) - 1
            should_sell = False
            if entry_mode == "trend_follow":
                if float(today["disparity"]) <= params.sell_disparity:
                    should_sell = True
            else:
                if float(today["disparity"]) >= params.sell_disparity:
                    should_sell = True
            if params.take_profit_ret and ret >= params.take_profit_ret:
                should_sell = True
            if ret <= params.stop_loss:
                should_sell = True
            if pos["hold_days"] >= params.max_holding_days:
                should_sell = True

            if should_sell:
                proceeds = pos["qty"] * exit_price
                cash += proceeds
                trades.append({
                    "code": code,
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "entry_price": pos["avg_price"],
                    "exit_date": nd.strftime("%Y-%m-%d"),
                    "exit_price": exit_price,
                    "qty": pos["qty"],
                    "pnl": proceeds - (pos["qty"] * pos["avg_price"]),
                    "ret": ret,
                    "hold_days": pos["hold_days"],
                })
                del positions[code]

        # --- 2) 종목 선별(신호) 및 신규 진입 (t일 기준 -> t+1 시가 진입) ---
        day_df = prices.loc[prices["date"] == d].copy()
        next_df = prices.loc[prices["date"] == nd][["code", "open", "close"]].copy()
        if day_df.empty or next_df.empty:
            continue

        # 유동성 필터(당일 기준) + 상위 N개
        day_df = day_df.sort_values("amount", ascending=False)
        day_df = day_df[day_df["amount"] >= params.min_amount].head(params.liquidity_rank)

        # 당일 선별된 종목(실제 generate_signals와 동일하게 amount 순서대로, max_positions 캡 적용)
        signals_final: List[str] = []
        signals_all: List[str] = []
        candidates: List[Tuple[str, str, float]] = []

        # sector counts (existing positions)
        sector_counts: Dict[str, int] = {}
        for code in positions.keys():
            sec = sector_of(code)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

        for _, row in day_df.iterrows():
            code = row["code"]
            th = buy_threshold(code)
            disp = row.get("disparity", 0)
            r3 = row.get("ret3", 0)

            # trend filter: ma25 상승
            if params.trend_ma25_rising:
                ma25 = float(row.get("ma25", 0) or 0)
                ma25_prev = row.get("ma25_prev")
                if ma25_prev is None or pd.isna(ma25_prev):
                    continue
                if ma25 <= float(ma25_prev):
                    continue

            try:
                disp_f = float(disp)
            except Exception:
                disp_f = 0.0
            try:
                r3_f = float(r3)
            except Exception:
                r3_f = 0.0

            if entry_mode == "trend_follow":
                if disp_f >= th and r3_f >= 0:
                    signals_all.append(code)
                    try:
                        amt = float(row.get("amount", 0) or 0)
                    except Exception:
                        amt = 0.0
                    score = (disp_f) + (0.8 * (r3_f)) + (0.05 * math.log1p(max(amt, 0.0)))
                    candidates.append((code, sector_of(code), score))
            else:
                if disp_f <= th:
                    signals_all.append(code)
                    # 후보 스코어(선택): 과매도 깊이 + 단기 투매 강도 + 유동성(거래대금)
                    try:
                        amt = float(row.get("amount", 0) or 0)
                    except Exception:
                        amt = 0.0
                    score = (-disp_f) + (0.8 * (-r3_f)) + (0.05 * math.log1p(max(amt, 0.0)))
                    candidates.append((code, sector_of(code), score))

        # 랭킹 정렬
        if str(params.rank_mode or "amount").lower() == "score":
            candidates.sort(key=lambda x: x[2], reverse=True)

        # sector cap + max_positions
        for code, sec, _score in candidates:
            if params.max_per_sector and params.max_per_sector > 0:
                if sector_counts.get(sec, 0) >= params.max_per_sector:
                    continue
            if len(signals_final) < params.max_positions:
                signals_final.append(code)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
            if len(signals_final) >= params.max_positions:
                break

        # selection report: 익일 시가->종가 기준(기본)
        merged_next = next_df.set_index("code")
        for code in signals_final:
            if code not in merged_next.index:
                continue
            op = float(merged_next.loc[code, "open"])
            cl = float(merged_next.loc[code, "close"])
            if op <= 0:
                continue
            r = (cl / op) - 1
            selection_rows.append({
                "signal_date": d.strftime("%Y-%m-%d"),
                "code": code,
                "next_open": op,
                "next_close": cl,
                "next_intraday_ret": r,
                "win": int(r > 0),
                "signals_all_count": len(signals_all),
                "signals_final_count": len(signals_final),
            })

        # 실제 진입 (max_positions까지, 현금 배분: 동일 로직 유지)
        for code in signals_final:
            if code in positions:
                continue
            if len(positions) >= params.max_positions:
                break

            today = get_row(code, d)
            next_row = get_row(code, nd)
            if today is None or next_row is None:
                continue

            open_price = float(next_row["open"])
            if open_price <= 0:
                continue

            invest_cash = cash
            if getattr(params, "capital_utilization", 0.0) and 0 < float(params.capital_utilization) <= 1.0:
                invest_cash = cash * float(params.capital_utilization)
            remaining = max(1, params.max_positions - len(positions))
            budget = invest_cash / remaining
            qty = int(budget // open_price)
            if qty <= 0:
                continue
            cost = qty * open_price
            cash -= cost
            positions[code] = {
                "qty": qty,
                "avg_price": open_price,
                "entry_date": nd,
                "hold_days": 0,
            }

        # --- 3) 일별 평가액 기록 (t+1일 시가 기준 보유 포지션 평가) ---
        equity = cash
        for code, pos in positions.items():
            nrow = get_row(code, nd)
            if nrow is None:
                continue
            px = float(nrow["open"])
            equity += pos["qty"] * px

        equity_curve.append({"date": nd.strftime("%Y-%m-%d"), "equity": equity, "cash": cash, "positions": len(positions)})

    # 결과 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(equity_curve).to_csv(output_dir / "equity_curve.csv", index=False)
    pd.DataFrame(trades).to_csv(output_dir / "trade_log.csv", index=False)

    # (편의) universe_members 조인해서 trade_log_enriched / daily_actions 생성
    try:
        univ = pd.read_sql_query(
            "SELECT code, name, market, group_name FROM universe_members",
            store.conn,
        )
        if not univ.empty and len(trades):
            tdf = pd.DataFrame(trades).copy()
            tdf["code"] = tdf["code"].astype(str).str.zfill(6)
            univ["code"] = univ["code"].astype(str).str.zfill(6)

            enriched = tdf.merge(univ, on="code", how="left")
            enriched.to_csv(output_dir / "trade_log_enriched.csv", index=False)

            def _join_codes(df):
                return ", ".join((df["code"] + " " + df["name"].fillna("")).tolist())

            buys = enriched.groupby("entry_date").apply(_join_codes).to_dict()
            sells = enriched.groupby("exit_date").apply(_join_codes).to_dict()
            all_dates = sorted(set(buys.keys()) | set(sells.keys()))
            daily_actions = pd.DataFrame(
                [{"date": d, "buys": buys.get(d, ""), "sells": sells.get(d, "")} for d in all_dates]
            )
            daily_actions.to_csv(output_dir / "daily_actions.csv", index=False)
    except Exception:
        pass
    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(output_dir / "selection_report.csv", index=False)

    # 요약 출력
    trade_win_rate = float((pd.DataFrame(trades)["pnl"] > 0).mean()) if len(trades) else 0.0
    sel_win_rate = float(selection_df["win"].mean()) if len(selection_df) else 0.0
    print(f"saved trades={len(trades)}, selection_rows={len(selection_df)}, equity_rows={len(equity_curve)}")
    print(f"trade_win_rate={trade_win_rate:.2%}, selection_win_rate(next_intraday)={sel_win_rate:.2%}")


def main():
    ensure_repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data", help="output directory for reports")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD (inclusive). Warm-up buffer is applied automatically.")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--codes", default=None, help="comma-separated codes to restrict universe (e.g. 005930,000660)")
    args = parser.parse_args()
    settings = load_settings()
    params = load_strategy(settings)
    store = SQLiteStore(settings.get("database", {}).get("path", "data/market_data.db"))
    codes = [c.strip() for c in (args.codes or "").split(",") if c.strip()] or None
    run_backtest(store, params, output_dir=Path(args.output_dir), start_date=args.start_date, end_date=args.end_date, codes=codes)


if __name__ == "__main__":
    main()
