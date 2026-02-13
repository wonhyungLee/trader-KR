from __future__ import annotations
import argparse
import time
import logging
import traceback
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from src.storage.sqlite_store import SQLiteStore
from src.utils.config import load_settings
from src.collectors.kis_price_client import KISPriceClient
from src.utils.notifier import maybe_notify
from src.utils.db_exporter import maybe_export_db

# Ensure logs are visible
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AuthForbiddenError(Exception):
    pass


def _is_auth_forbidden_error(exc: Exception) -> bool:
    msg = str(exc)
    low = msg.lower()
    if "429" in msg:
        return True
    if "403" in msg and "tokenp" in low:
        return True
    return any(
        key in low
        for key in (
            "rate limit",
            "rate-limited",
            "too many requests",
            "too many request",
            "속도 제한",
            "과도한 요청",
        )
    )

def read_universe(paths: Iterable[str]) -> List[str]:
    codes: List[str] = []
    for p in paths:
        if not Path(p).exists():
            continue
        df = pd.read_csv(p)
        col = "code" if "code" in df.columns else "Code" if "Code" in df.columns else df.columns[0]
        codes.extend(df[col].astype(str).str.zfill(6).tolist())
    seen = set()
    uniq = []
    for c in codes:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _parse_kis_daily(res: dict) -> pd.DataFrame:
    outputs = res.get("output2") or res.get("output") or []
    if not isinstance(outputs, list) or not outputs:
        return pd.DataFrame()
    recs = []
    for o in outputs:
        close = float(o.get("stck_clpr") or 0)
        vol = float(o.get("acml_vol") or 0)
        amount = float(o.get("acml_tr_pbmn") or 0)
        if amount <= 0 and close > 0 and vol > 0:
            amount = close * vol
        recs.append(
            {
                "date": o.get("stck_bsop_date"),
                "open": float(o.get("stck_oprc") or 0),
                "high": float(o.get("stck_hgpr") or 0),
                "low": float(o.get("stck_lwpr") or 0),
                "close": close,
                "volume": vol,
                "amount": amount,
            }
        )
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors='coerce').dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])
    if df.empty:
        return df
    df = df.sort_values("date")
    df["ma25"] = df["close"].rolling(25, min_periods=5).mean()
    df["disparity"] = df["close"] / df["ma25"] - 1
    df["disparity"] = df["disparity"].replace([float("inf"), float("-inf")], pd.NA)
    return df[["date", "open", "high", "low", "close", "volume", "amount", "ma25", "disparity"]]


def fetch_prices_kis(client: KISPriceClient, code: str, start: str, end: str) -> pd.DataFrame:
    res = client.get_daily_prices(code, start.replace("-", ""), end.replace("-", ""))
    return _parse_kis_daily(res)


def backward_refill(
    store: SQLiteStore,
    code: str,
    chunk_days: int,
    sleep: float,
    empty_limit: int = 3,
    kis_client: Optional[KISPriceClient] = None,
    notify_cb=None,
    notify_every: int = 1,
    resume_end: Optional[str] = None,
    auth_cooldown: Optional[float] = None,
):
    today = datetime.today().date()
    current_end = datetime.strptime(resume_end, "%Y-%m-%d").date() if resume_end else today
    
    empty_cnt = 0
    last_min_date: Optional[str] = None
    chunk_idx = 0

    while True:
        start_date = current_end - timedelta(days=chunk_days)
        chunk_idx += 1
        
        print(f"[{code}] Chunk {chunk_idx}: fetching up to {current_end:%Y-%m-%d}...")
        
        try:
            df = fetch_prices_kis(
                kis_client,
                code,
                start_date.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d"),
            )  # type: ignore
        except Exception as e:
            if _is_auth_forbidden_error(e):
                cooldown = float(auth_cooldown or 0)
                print(f"[{code}] 403 tokenP detected. Cooling down {cooldown:.1f}s and clearing cache.")
                if cooldown > 0:
                    time.sleep(cooldown)
                if kis_client is not None:
                    try:
                        kis_client.broker.clear_token_cache()
                        kis_client.broker.reset_sessions()
                    except Exception:
                        pass
                continue
            print(f"[{code}] API Error: {e}")
            empty_cnt += 1
            time.sleep(sleep * 5)
            if empty_cnt >= empty_limit:
                break
            continue

        if df.empty:
            print(f"[{code}] Empty response at {current_end:%Y-%m-%d}")
            empty_cnt += 1
        else:
            min_date_str = df["date"].min()
            min_date = datetime.strptime(min_date_str, "%Y-%m-%d").date()
            
            if last_min_date and min_date_str >= last_min_date:
                print(f"[{code}] Duplicate/No-earlier data at {min_date_str}")
                empty_cnt += 1
            else:
                empty_cnt = 0
                last_min_date = min_date_str
                store.upsert_daily_prices(code, df)
                current_end = min_date - timedelta(days=1)
                print(f"[{code}] Saved {len(df)} rows. Next end: {current_end:%Y-%m-%d}")

        store.upsert_refill_status(
            code=code,
            next_end=current_end.strftime("%Y-%m-%d"),
            last_min=last_min_date,
            status="RUNNING",
            message=f"chunk={chunk_idx} empty={empty_cnt}",
        )

        if empty_cnt >= empty_limit:
            print(f"[{code}] Stopped: empty limit reached.")
            break
        if current_end.year < 1980:
            print(f"[{code}] Stopped: year limit reached.")
            break

        time.sleep(sleep)

    store.upsert_refill_status(
        code=code,
        next_end=current_end.strftime("%Y-%m-%d"),
        last_min=last_min_date,
        status="DONE",
        message=f"chunks={chunk_idx}",
    )


def main():
    print("MAIN START")
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", action="append", help="CSV 파일 경로", default=[])
    parser.add_argument("--code", help="단일 종목 코드", default=None)
    parser.add_argument("--chunk-days", type=int, default=150)
    parser.add_argument("--resume", action="store_true", help="중단 지점부터 재개")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    db_path = settings.get("database", {}).get("path", "data/market_data.db")
    store = SQLiteStore(db_path)
    job_id = store.start_job("refill_loader")
    
    kis_client = KISPriceClient(settings)
    kis_client.broker.reset_sessions()

    sleep = float(settings.get("kis", {}).get("rate_limit_sleep_sec", 0.5))

    if args.code:
        codes = [args.code.zfill(6)]
    elif args.universe:
        codes = read_universe(args.universe)
    else:
        codes = store.list_universe_codes()
    
    if not codes:
        print("Error: No codes to process.")
        return

    if args.limit:
        codes = codes[: args.limit]

    # Calculate global totals
    all_universe_codes = store.list_universe_codes() if not args.universe else read_universe(args.universe)
    total_universe_count = len(all_universe_codes)

    print(f"Processing {len(codes)} codes...")
    maybe_notify(
        settings,
        f"▶️ [refill] start codes={len(codes)} universe={total_universe_count} "
        f"resume={bool(args.resume)} chunk_days={args.chunk_days} rate_sleep={sleep}",
    )
    processed_in_this_run = 0
    try:
        for code in codes:
            status = store.get_refill_status(code)
            if args.resume and status and status["status"] == "DONE":
                continue
            
            resume_end = status["next_end_date"] if status and status["next_end_date"] else None
            
            print(f"=== Starting {code} ({processed_in_this_run+1}/{len(codes)}) ===")
            try:
                backward_refill(
                    store,
                    code,
                    args.chunk_days,
                    sleep,
                    kis_client=kis_client,
                    resume_end=resume_end,
                    auth_cooldown=settings.get("kis", {}).get("auth_forbidden_cooldown_sec", 600),
                )
                processed_in_this_run += 1
                
                # Export DB to CSV after EACH stock
                maybe_export_db(settings, store.db_path)
                
                # Get global done count
                done_count = store.conn.execute("SELECT count(*) FROM refill_progress WHERE status='DONE'").fetchone()[0]
                
                # Notify after each stock completion
                msg = f"✅ [refill] {code} 완료 ({done_count}/{total_universe_count})"
                maybe_notify(settings, msg)
                
                # Prevent Discord rate limit
                time.sleep(0.5)

            except Exception as e:
                print(f"Error processing {code}: {e}")
                traceback.print_exc()
                store.upsert_refill_status(code, resume_end, None, "ERROR", str(e))
                maybe_notify(settings, f"❌ [refill] {code} 오류: {e}")
        
        store.finish_job(job_id, "SUCCESS", f"processed={processed_in_this_run}")
        maybe_notify(
            settings,
            f"✅ [refill] 완료 processed={processed_in_this_run} total={len(codes)}",
        )
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        maybe_notify(settings, f"❌ [refill] 실패: {e}")
        store.finish_job(job_id, "ERROR", str(e))
    finally:
        maybe_export_db(settings, store.db_path)
        maybe_notify(settings, f"🏁 [refill] 전체 작업 종료 (총 {processed_in_this_run}개)")

if __name__ == "__main__":
    main()
