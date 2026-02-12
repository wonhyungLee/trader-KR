"""유니버스(350개) 스냅샷 로더.

- 기본: data/universe_kospi200.csv + data/universe_kosdaq150.csv
- 옵션: --record-diff 로 변경 내역 기록
- 옵션: --auto-rank 로 시총 상위 N 자동 재산정(지수 구성과 다를 수 있음)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.storage.sqlite_store import SQLiteStore
from src.utils.config import load_settings
from src.utils.db_exporter import maybe_export_db
from src.utils.project_root import ensure_repo_root


def load_universe_csv(path: str, group_name: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"universe file missing: {path}")
    df = pd.read_csv(p)
    required = {"code", "name", "market"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Missing columns in {path}: {required - set(df.columns)}")
    df = df.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["group_name"] = group_name
    return df[["code", "name", "market", "group_name"]]


def _auto_rank_universe(store: SQLiteStore, top_kospi: int, top_kosdaq: int) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT code, name, market, marcap FROM stock_info WHERE market IS NOT NULL",
        store.conn,
    )
    if df.empty:
        raise RuntimeError("stock_info is empty; auto_rank requires stock_info with marcap")

    kospi = df[df["market"].str.contains("KOSPI", na=False)].sort_values("marcap", ascending=False).head(top_kospi)
    kosdaq = df[df["market"].str.contains("KOSDAQ", na=False)].sort_values("marcap", ascending=False).head(top_kosdaq)
    if len(kospi) < top_kospi or len(kosdaq) < top_kosdaq:
        raise RuntimeError("stock_info does not contain enough KOSPI/KOSDAQ entries for auto_rank")

    kospi = kospi.assign(group_name="KOSPI200")
    kosdaq = kosdaq.assign(group_name="KOSDAQ150")
    return pd.concat([kospi, kosdaq], ignore_index=True)[["code", "name", "market", "group_name"]]


def _record_diff(store: SQLiteStore, df: pd.DataFrame):
    snapshot_date = datetime.utcnow().strftime("%Y-%m-%d")
    prev = pd.read_sql_query("SELECT code, market FROM universe_members", store.conn)
    if prev.empty:
        return

    for market in sorted(df["market"].dropna().unique().tolist()):
        new_codes = set(df[df["market"] == market]["code"].tolist())
        prev_codes = set(prev[prev["market"] == market]["code"].tolist())
        added = sorted(list(new_codes - prev_codes))
        removed = sorted(list(prev_codes - new_codes))
        if added or removed:
            store.insert_universe_change(
                snapshot_date,
                market,
                json.dumps(added, ensure_ascii=False),
                json.dumps(removed, ensure_ascii=False),
            )


def main(args: argparse.Namespace):
    ensure_repo_root(Path(__file__).resolve())
    settings = load_settings()
    store = SQLiteStore(settings.get("database", {}).get("path", "data/market_data.db"))
    job_id = store.start_job("universe_loader")

    try:
        if args.auto_rank:
            df = _auto_rank_universe(store, args.top_kospi, args.top_kosdaq)
            # persist auto_rank snapshot to CSVs
            Path("data").mkdir(parents=True, exist_ok=True)
            df[df["group_name"] == "KOSPI200"].to_csv("data/universe_kospi200.csv", index=False)
            df[df["group_name"] == "KOSDAQ150"].to_csv("data/universe_kosdaq150.csv", index=False)
        else:
            df_kospi = load_universe_csv("data/universe_kospi200.csv", "KOSPI200")
            df_kosdaq = load_universe_csv("data/universe_kosdaq150.csv", "KOSDAQ150")
            df = pd.concat([df_kospi, df_kosdaq], ignore_index=True)

        if len(df) != 350:
            raise RuntimeError(f"universe count must be 350, got {len(df)}")

        if args.record_diff:
            _record_diff(store, df)

        # universe_members 고정
        store.upsert_universe_members(df.to_dict(orient="records"))
        # stock_info는 universe_members 기준 250개만 유지
        if "marcap" in df.columns:
            stock_rows = df.assign(marcap=df["marcap"].fillna(0)).to_dict(orient="records")
        else:
            stock_rows = df.assign(marcap=0).to_dict(orient="records")
        store.replace_stock_info(stock_rows)

        print(f"stored universe {len(df)} symbols at {datetime.now():%Y-%m-%d %H:%M:%S}")
        maybe_export_db(settings, store.db_path)
        store.finish_job(job_id, "SUCCESS", f"stored {len(df)} symbols")
    except Exception as exc:
        store.finish_job(job_id, "ERROR", str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-diff", action="store_true", help="CSV diff를 universe_changes에 기록")
    parser.add_argument("--auto-rank", action="store_true", help="시총 상위 N 자동 재산정")
    parser.add_argument("--top-kospi", type=int, default=200, help="auto_rank KOSPI 상위 N")
    parser.add_argument("--top-kosdaq", type=int, default=150, help="auto_rank KOSDAQ 상위 N")
    args = parser.parse_args()
    main(args)
