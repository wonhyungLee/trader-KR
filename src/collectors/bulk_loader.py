"""초기 대량 수집 스크립트.

FinanceDataReader를 사용해 최근 2년 일봉을 적재한다. 실전 운영에서 KIS API로 대체 가능.
"""

import argparse
from datetime import datetime, timedelta
import pandas as pd
import FinanceDataReader as fdr  # type: ignore

from src.storage.sqlite_store import SQLiteStore
from src.utils.config import load_settings
from src.utils.db_exporter import maybe_export_db


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma25"] = df["Close"].rolling(25, min_periods=5).mean()
    df["disparity"] = df["Close"] / df["ma25"] - 1
    df["Amount"] = df.get("Amount", df["Close"] * df["Volume"])
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume", "Amount": "amount"})
    df["date"] = df.index.strftime("%Y-%m-%d")
    return df[["date", "open", "high", "low", "close", "volume", "amount", "ma25", "disparity"]]


def load_code(store: SQLiteStore, code: str, start: str, end: str):
    try:
        raw = fdr.DataReader(code, start=start, end=end)
    except Exception as e:
        print(f"skip {code}: {e}")
        return
    if raw.empty:
        print(f"skip {code}: empty")
        return
    df = compute_features(raw)
    store.upsert_daily_prices(code, df)
    print(f"stored {code} {len(df)} rows")


def main(codes: list[str] | None = None, days: int = 500):
    settings = load_settings()
    store = SQLiteStore(settings.get("database", {}).get("path", "data/market_data.db"))
    if codes is None:
        codes = store.list_universe_codes()
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    for code in codes:
        load_code(store, code, start, end)
    maybe_export_db(settings, store.db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=500)
    parser.add_argument("--codes", nargs="*", default=None)
    args = parser.parse_args()
    main(args.codes, args.days)
