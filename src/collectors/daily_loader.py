"""일일 증분 수집 (KIS)."""

import argparse
import logging
import time
from datetime import datetime, timedelta
import pandas as pd

from src.storage.sqlite_store import SQLiteStore
from src.utils.config import load_settings
from src.utils.db_exporter import maybe_export_db
from src.collectors.kis_price_client import KISPriceClient
from src.collectors.refill_loader import _parse_kis_daily


def fetch_prices_kis(client: KISPriceClient, code: str, start: str, end: str) -> pd.DataFrame:
    res = client.get_daily_prices(code, start.replace("-", ""), end.replace("-", ""))
    return _parse_kis_daily(res)

def _sleep_on_error(exc: Exception, settings: dict, client=None) -> None:
    msg = str(exc)
    low = msg.lower()
    is_rate_limit = any(
        key in low
        for key in (
            "429",
            "rate limit",
            "rate-limited",
            "too many requests",
            "too many request",
            "속도 제한",
            "과도한 요청",
        )
    )
    if "403" in msg:
        sleep_sec = float(settings.get("kis", {}).get("auth_forbidden_cooldown_sec", 600))
    elif is_rate_limit:
        sleep_sec = float(settings.get("kis", {}).get("auth_forbidden_cooldown_sec", 600))
        if client is not None and hasattr(client, "broker"):
            try:
                client.broker.clear_token_cache()
                client.broker.reset_sessions()
            except Exception:
                pass
    elif "500" in msg:
        sleep_sec = float(settings.get("kis", {}).get("consecutive_error_cooldown_sec", 180))
    else:
        sleep_sec = 5.0
    logging.warning("daily_loader error. cooling down %.1fs: %s", sleep_sec, msg)
    time.sleep(max(1.0, sleep_sec))

def _attach_indicators(store: SQLiteStore, code: str, df_new: pd.DataFrame, lookback_rows: int = 120) -> pd.DataFrame:
    """Recompute ma25/disparity for newly fetched rows using DB history.

    The KIS API call in incremental mode may return only 1-2 trading days. If we compute
    rolling indicators only on that small slice, ma25/disparity becomes NaN and the
    selection engine filters out all candidates.

    This function stitches recent close history from DB (lookback_rows trading days)
    with the newly fetched rows and recomputes indicators, then returns only the new
    rows with filled ma25/disparity.
    """
    if df_new is None or df_new.empty:
        return df_new

    df_new = df_new.copy()
    df_new["date"] = pd.to_datetime(df_new["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_new = df_new.dropna(subset=["date"])
    if df_new.empty:
        return df_new

    # Load history up to the day before the first new date.
    first_new = df_new["date"].min()
    try:
        hist_end = (datetime.strptime(first_new, "%Y-%m-%d").date() - timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        hist_end = None

    hist = store.load_recent_closes(code, end_date=hist_end, limit=lookback_rows)

    base = pd.concat([hist, df_new[["date", "close"]]], ignore_index=True) if not hist.empty else df_new[["date", "close"]].copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base = base.dropna(subset=["date"]).sort_values("date")
    base = base.drop_duplicates(subset=["date"], keep="last")

    base["ma25"] = base["close"].rolling(25, min_periods=5).mean()
    base["disparity"] = base["close"] / base["ma25"] - 1

    ind = base[["date", "ma25", "disparity"]].copy()
    ind["date"] = ind["date"].dt.strftime("%Y-%m-%d")

    # Replace the indicators in df_new with recomputed values.
    df_new = df_new.drop(columns=["ma25", "disparity"], errors="ignore")
    df_new = df_new.merge(ind, on="date", how="left")

    cols = ["date", "open", "high", "low", "close", "volume", "amount", "ma25", "disparity"]
    for c in cols:
        if c not in df_new.columns:
            df_new[c] = None
    return df_new[cols]


def main(limit: int | None = None, chunk_days: int = 90, indicator_lookback_rows: int = 120):
    settings = load_settings()
    store = SQLiteStore(settings.get("database", {}).get("path", "data/market_data.db"))
    store.cleanup_stale_running_jobs()
    job_id = store.start_job("daily_loader")
    client = KISPriceClient(settings)
    client.broker.reset_sessions()

    codes = store.list_universe_codes()
    if not codes:
        raise SystemExit("universe_members is empty. Run universe_loader first.")
    if limit:
        codes = codes[:limit]
    today = datetime.today().date()
    errors = 0
    for code in codes:
        try:
            last = store.last_price_date(code)
            if not last:
                # refill이 먼저
                continue
            start_dt = datetime.strptime(last, "%Y-%m-%d").date() + timedelta(days=1)
            if start_dt > today:
                continue

            # forward chunk
            cur_start = start_dt
            while cur_start <= today:
                cur_end = min(cur_start + timedelta(days=chunk_days), today)
                try:
                    df = fetch_prices_kis(client, code, cur_start.strftime("%Y-%m-%d"), cur_end.strftime("%Y-%m-%d"))
                except Exception as exc:
                    errors += 1
                    logging.warning("daily_loader fetch failed %s: %s", code, exc)
                    _sleep_on_error(exc, settings, client)
                    break
                if df.empty:
                    break
                df = _attach_indicators(store, code, df, indicator_lookback_rows)
                store.upsert_daily_prices(code, df)
                max_date = df["date"].max()
                next_start = datetime.strptime(max_date, "%Y-%m-%d").date() + timedelta(days=1)
                if next_start <= cur_start:
                    break
                cur_start = next_start
        except Exception as exc:
            errors += 1
            logging.exception("daily_loader failed for %s", code)
            _sleep_on_error(exc, settings, client)
            continue

    status = "SUCCESS" if errors == 0 else "PARTIAL"
    store.finish_job(job_id, status, f"codes={len(codes)} errors={errors}")

    maybe_export_db(settings, store.db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="처리할 종목 수 제한(테스트 용)")
    parser.add_argument("--chunk-days", type=int, default=90, help="증분 호출 범위(캘린더일)")
    parser.add_argument("--indicator-lookback-rows", type=int, default=120, help="MA25/괴리율 계산용 DB 히스토리 로우 수(트레이딩데이)")
    args = parser.parse_args()
    main(args.limit, args.chunk_days, args.indicator_lookback_rows)
