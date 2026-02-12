from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.utils.config import load_settings
from src.brokers.kis_broker import KISBroker


class KISPriceClient:
    """간단한 KIS 기간별시세(일봉) 클라이언트.

    - 80초 슬립 기본 (사용자 레이트리밋 설정값 사용)
    - start/end YYYYMMDD 포맷, 최대 100건 제약 대비 chunk 조정은 호출측에서 처리
    """

    def __init__(self, settings: Dict[str, Any] | None = None):
        self.settings = settings or load_settings()
        self.kis = self.settings["kis"]
        self.broker = KISBroker(self.settings)
        self.base_url = self.kis.get(
            "base_url_prod" if self.settings.get("env", "paper") == "prod" else "base_url_paper"
        )
        # self.rate_sleep is no longer needed; KISBroker handles it.

    def _tr_id(self) -> str:
        # 국내주식 기간별 시세(일/주/월/년) TR
        return "FHKST03010100"

    def get_daily_prices(self, code: str, start: str, end: str) -> Dict[str, Any]:
        """start/end: YYYYMMDD"""
        tr_id = self._tr_id()
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": start,
            "FID_INPUT_DATE_2": end,
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "1",
        }
        return self.broker.request(tr_id, url, params=params)
