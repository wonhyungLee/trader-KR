"""KIS Open API 기반 시세 수집 유틸.

REST 호출을 감싸 FinanceDataReader 대신 사용할 수 있도록 한다.
"""
from __future__ import annotations

from datetime import datetime
import pandas as pd
from typing import Optional

from src.brokers.kis_broker import KISBroker


class KISCollector:
    def __init__(self, broker: KISBroker):
        self.broker = broker

    def fetch_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily OHLCV between dates (inclusive) using KIS inquire-daily-price API."""
        # Endpoint spec: /uapi/domestic-stock/v1/quotations/inquire-daily-price
        base_url = self.broker.base_url
        url = f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": start.replace("-", ""),
            "FID_INPUT_DATE_2": end.replace("-", ""),
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "1",
        }
        res = self.broker.request("FHKST01010400", url, params=params)
        output = res.get("output", []) if isinstance(res, dict) else []
        if not output:
            return pd.DataFrame()
        
        records = []
        for row in output:
            vol = float(row.get("acml_vol") or row.get("cntg_vol") or 0)
            close = float(row["stck_clpr"])
            amount = float(row.get("acml_tr_pbmn") or 0)
            if amount <= 0:
                amount = close * vol
                
            records.append(
                {
                    "date": datetime.strptime(row["stck_bsop_date"], "%Y%m%d").strftime("%Y-%m-%d"),
                    "open": float(row["stck_oprc"]),
                    "high": float(row["stck_hgpr"]),
                    "low": float(row["stck_lwpr"]),
                    "close": close,
                    "volume": int(vol),
                    "amount": amount,
                }
            )
        df = pd.DataFrame(records)
        df = df.sort_values("date")
        df["ma25"] = df["close"].rolling(25, min_periods=5).mean()
        df["disparity"] = df["close"] / df["ma25"] - 1
        return df
