from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.brokers.kis_broker import KISBroker
from src.storage.sqlite_store import SQLiteStore
from src.utils.config import load_settings
from src.utils.notifier import maybe_notify
from src.utils.project_root import ensure_repo_root


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


API_URL_STOCK_INFO = "/uapi/domestic-stock/v1/quotations/search-stock-info"
API_URL_SEARCH_INFO = "/uapi/domestic-stock/v1/quotations/search-info"
TR_ID_STOCK_INFO = "CTPF1002R"
TR_ID_SEARCH_INFO = "CTPF1604R"

BAD_SECTOR_TOKENS = ["시가총액", "KOGI", "지배구조", "지수"]


def _parse_output(res: dict) -> Optional[dict]:
    out = res.get("output") or res.get("output1") or res.get("output2")
    if isinstance(out, list):
        return out[0] if out else None
    if isinstance(out, dict):
        return out
    return None


def _pick_sector_fields(payload: dict) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]:
    industry_code = payload.get("std_idst_clsf_cd") or None
    industry_name = payload.get("std_idst_clsf_cd_name") or None

    sector_code = payload.get("idx_bztp_mcls_cd") or payload.get("idx_bztp_scls_cd") or payload.get("idx_bztp_lcls_cd")
    sector_name = payload.get("idx_bztp_mcls_cd_name") or payload.get("idx_bztp_scls_cd_name") or payload.get("idx_bztp_lcls_cd_name")

    def _bad(name: Optional[str]) -> bool:
        if not name:
            return True
        return any(tok in name for tok in BAD_SECTOR_TOKENS)

    if _bad(sector_name):
        sector_name = None
        sector_code = None

    if not sector_name and industry_name:
        sector_name = industry_name
        sector_code = industry_code[:2] if industry_code else None
        source = "search_stock_info:industry_fallback"
    else:
        source = "search_stock_info"

    return sector_code, sector_name, industry_code, industry_name, source


def fetch_sector_info(broker: KISBroker, code: str) -> Dict[str, Optional[str]]:
    params = {
        "PRDT_TYPE_CD": "300",
        "PDNO": code,
    }
    res = broker.request(TR_ID_STOCK_INFO, f"{broker.base_url}{API_URL_STOCK_INFO}", params=params)
    payload = _parse_output(res)
    if payload:
        sector_code, sector_name, industry_code, industry_name, source = _pick_sector_fields(payload)
        return {
            "sector_code": sector_code,
            "sector_name": sector_name,
            "industry_code": industry_code,
            "industry_name": industry_name,
            "source": source,
        }

    # fallback search_info
    res = broker.request(TR_ID_SEARCH_INFO, f"{broker.base_url}{API_URL_SEARCH_INFO}", params=params)
    payload = _parse_output(res)
    sector_code = None
    sector_name = None
    industry_code = None
    industry_name = None
    if payload:
        industry_code = payload.get("prdt_clsf_cd") or None
        industry_name = payload.get("prdt_clsf_name") or None
        sector_name = industry_name
        sector_code = industry_code
    return {
        "sector_code": sector_code,
        "sector_name": sector_name,
        "industry_code": industry_code,
        "industry_name": industry_name,
        "source": "search_info",
    }


def _sanitize_filename(value: str) -> str:
    if not value:
        return "UNKNOWN"
    s = value.strip().replace("/", "-")
    s = s.replace("\\", "-")
    s = s.replace(":", "-")
    return s


def build_sector_csvs(store: SQLiteStore, out_root: Path) -> dict:
    conn = store.conn
    df = pd.read_sql_query(
        """
        SELECT u.code, u.name, u.market, u.group_name, s.sector_code, s.sector_name, s.industry_code, s.industry_name
        FROM universe_members u
        LEFT JOIN sector_map s ON u.code = s.code
        ORDER BY u.code
        """,
        conn,
    )

    total = len(df)
    unique_codes = df["code"].nunique() if not df.empty else 0
    if total == 0:
        return {"total": 0, "unique": 0, "unknown": 0}
    if total != unique_codes:
        logging.warning("duplicate codes detected in universe_members: total=%s unique=%s", total, unique_codes)

    df["sector_name"] = df["sector_name"].fillna("UNKNOWN")
    df["sector_code"] = df["sector_code"].fillna("")
    df["industry_code"] = df["industry_code"].fillna("")
    df["industry_name"] = df["industry_name"].fillna("")

    unknown_df = df[df["sector_name"] == "UNKNOWN"].copy()

    # split by market
    for market in sorted(df["market"].dropna().unique().tolist()):
        mdf = df[df["market"] == market]
        market_dir = out_root / market
        market_dir.mkdir(parents=True, exist_ok=True)
        for sector, sdf in mdf.groupby("sector_name"):
            fname = _sanitize_filename(sector)
            path = market_dir / f"{fname}.csv"
            sdf[["code", "name", "market", "sector_name", "sector_code", "industry_name", "industry_code"]].to_csv(
                path, index=False
            )

    # unknown list
    if not unknown_df.empty:
        out_root.mkdir(parents=True, exist_ok=True)
        unknown_df[["code", "name", "market", "group_name"]].to_csv(out_root / "UNKNOWN.csv", index=False)

    return {"total": total, "unique": unique_codes, "unknown": len(unknown_df)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-days", type=int, default=30, help="n일 이내 갱신 종목은 재호출하지 않음")
    parser.add_argument("--sleep", type=float, default=None, help="종목 간 슬립(초)")
    parser.add_argument("--limit", type=int, default=None, help="처리 종목 수 제한(테스트)")
    parser.add_argument("--notify-every", type=int, default=25, help="n개 종목마다 진행 알림")
    args = parser.parse_args()

    ensure_repo_root(Path(__file__).resolve())

    settings = load_settings()
    store = SQLiteStore(settings.get("database", {}).get("path", "data/market_data.db"))
    broker = KISBroker(settings)

    refresh_days = max(0, int(args.refresh_days))
    all_codes = store.list_universe_codes()
    if len(all_codes) != 350:
        logging.warning("universe_members count expected 350, got %s", len(all_codes))
    targets = store.list_sector_targets(refresh_days)
    if args.limit:
        targets = targets[: args.limit]

    item_sleep = args.sleep
    if item_sleep is None:
        item_sleep = float(settings.get("kis", {}).get("accuracy_item_sleep_sec", 0.1))

    total = len(targets)
    logging.info("sector classifier targets=%s refresh_days=%s", total, refresh_days)
    if args.notify_every > 0:
        maybe_notify(settings, f"[sector] start total={total} refresh_days={refresh_days}")

    updated_rows: List[Dict[str, Optional[str]]] = []
    for idx, code in enumerate(targets, start=1):
        try:
            info = fetch_sector_info(broker, code)
            info.update({"code": code, "updated_at": datetime.utcnow().isoformat()})
            updated_rows.append(info)
        except Exception as exc:
            logging.warning("sector fetch failed %s: %s", code, exc)
            updated_rows.append({
                "code": code,
                "sector_code": None,
                "sector_name": None,
                "industry_code": None,
                "industry_name": None,
                "updated_at": datetime.utcnow().isoformat(),
                "source": "error",
            })

        if updated_rows:
            store.upsert_sector_map(updated_rows)
            updated_rows.clear()

        if idx % 20 == 0 or idx == total:
            logging.info("sector classifier progress %s/%s", idx, total)
            if args.notify_every > 0 and (idx % args.notify_every == 0 or idx == total):
                maybe_notify(settings, f"[sector] {idx}/{total} code={code}")

        if item_sleep and item_sleep > 0:
            time.sleep(item_sleep)

    # build sector CSVs
    out_root = Path("data/universe_sectors")
    summary = build_sector_csvs(store, out_root)

    unknown_codes = store.list_sector_unknowns()
    total_codes = store.list_universe_codes()

    if args.notify_every > 0:
        maybe_notify(
            settings,
            f"[sector] done total={len(total_codes)} unknown={len(unknown_codes)}",
        )
    known_ratio = 0.0
    if total_codes:
        known_ratio = (len(total_codes) - len(unknown_codes)) / len(total_codes)

    logging.info(
        "sector_map coverage: known=%s total=%s ratio=%.2f unknown=%s",
        len(total_codes) - len(unknown_codes),
        len(total_codes),
        known_ratio,
        len(unknown_codes),
    )


if __name__ == "__main__":
    main()
