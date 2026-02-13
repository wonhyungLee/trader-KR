from __future__ import annotations

import logging
import time
from typing import Dict, Iterable, List

from src.brokers.kis_broker import KISBroker


def _to_float(value) -> float:
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return 0.0


def _normalize_code(code: str) -> str:
    return str(code).strip().zfill(6)


class Scanner:
    def __init__(self, settings: dict, broker: KISBroker):
        self.settings = settings
        monitor = settings.get("monitor", {}) or {}
        self.batch_size = min(int(monitor.get("rest_batch_size", 30)), 30)
        self.max_retries = int(monitor.get("rest_max_retries", 5))
        self.broker = broker

    def _parse_record(self, rec: Dict) -> Dict:
        code = rec.get("inter_shrn_iscd") or rec.get("stck_shrn_iscd") or rec.get("iscd") or rec.get("code")
        name = rec.get("inter_kor_isnm") or rec.get("hts_kor_isnm") or rec.get("name", "")
        price = rec.get("inter2_prpr") or rec.get("stck_prpr") or rec.get("prpr") or rec.get("price")
        amount = rec.get("inter2_acml_tr_pbmn") or rec.get("acml_tr_pbmn") or rec.get("amount")
        volume = rec.get("inter2_acml_vol") or rec.get("acml_vol") or rec.get("volume")
        parsed = {
            "code": _normalize_code(code) if code else None,
            "name": name,
            "price": _to_float(price),
            "amount": _to_float(amount),
            "volume": _to_float(volume),
            "raw": rec,
        }
        if parsed["amount"] <= 0 and parsed["price"] > 0 and parsed["volume"] > 0:
            parsed["amount"] = parsed["price"] * parsed["volume"]
        return parsed

    def scan_once(self, universe_codes: Iterable[str]) -> Dict[str, Dict]:
        codes = [_normalize_code(c) for c in universe_codes]
        snapshot: Dict[str, Dict] = {}
        for i in range(0, len(codes), self.batch_size):
            batch = codes[i : i + self.batch_size]
            if not batch:
                continue
            retries = 0
            backoff = 0.5
            while True:
                try:
                    res = self.broker.get_multi_price(batch)
                    outputs = res.get("output") or res.get("output1") or []
                    if isinstance(outputs, list):
                        for rec in outputs:
                            parsed = self._parse_record(rec)
                            code = parsed.get("code")
                            if code:
                                snapshot[code] = parsed
                    break
                except Exception as exc:
                    retries += 1
                    logging.warning("scan batch failed (%s/%s): %s", retries, self.max_retries, exc)
                    if retries >= self.max_retries:
                        break
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 10)
        return snapshot
