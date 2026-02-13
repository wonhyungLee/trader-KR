from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import random
import sqlite3
import logging
import json
import subprocess
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from src.analyzer.backtest_runner import load_strategy
from src.collectors.kis_price_client import KISPriceClient
from src.storage.sqlite_store import SQLiteStore
from src.utils.config import (
    _load_dotenv,
    _load_personal_env,
    load_settings,
    list_kis_key_inventory,
    set_kis_key_enabled,
)
from src.utils.notifier import maybe_notify
from src.utils.db_exporter import maybe_export_db
from src.utils.project_root import ensure_repo_root

# Ensure relative paths resolve from repo root (e.g. data/*, config/*)
ensure_repo_root(Path(__file__).resolve().parent)

# Load .env/개인정보 early so env-gated endpoints (toggle password, etc.) work under systemd.
try:
    _load_dotenv()
    _load_personal_env()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_PATH = Path("data/market_data.db")
FRONTEND_DIST = Path("frontend/dist")
CLIENT_ERROR_LOG = Path("logs/client_error.log")
KIS_TOGGLE_PASSWORD = os.getenv("KIS_TOGGLE_PASSWORD", "")  # set via env; empty disables toggle APIs
FILTER_TOGGLE_PATH = Path("data/selection_filter_toggles.json")
FILTER_TOGGLE_KEYS = ("min_amount", "liquidity", "disparity")
_selection_cache: Dict[str, Any] = {"ts": 0.0, "data": None}
SELECTION_NOTIFY_PATH = Path("data/selection_notify_state.json")
SELECTION_NOTIFY_COOLDOWN_SEC = int(os.getenv("SELECTION_NOTIFY_COOLDOWN_SEC", "300"))
SELECTION_NOTIFY_INTERVAL_SEC = int(os.getenv("SELECTION_NOTIFY_INTERVAL_SEC", "60"))
SELECTION_NOTIFY_REPEAT_SEC = int(os.getenv("SELECTION_NOTIFY_REPEAT_SEC", "3600"))
_selection_notify_health: Dict[str, Any] = {"last_error": None, "last_error_ts": 0.0, "last_ok_ts": 0.0}
DISABLE_TRADING_ENDPOINTS = True
_kis_price_client: Optional[KISPriceClient] = None
_kis_price_client_error: Optional[Exception] = None
_kis_price_client_lock = threading.Lock()

# Candidate-only realtime prices (KIS WebSocket).
# - We open a single WS connection lazily when candidates appear.
# - Frontend polls our in-memory cache; non-candidates don't pay the realtime cost.
_candidate_ws_lock = threading.Lock()
_candidate_ws_targets: set[str] = set()
_candidate_ws_prices: Dict[str, Dict[str, Any]] = {}
_candidate_ws_thread: Optional[threading.Thread] = None
_candidate_ws_health: Dict[str, Any] = {"last_error": None, "last_error_ts": 0.0, "last_ok_ts": 0.0}

# Coupang Partners banner cache (in-memory, best-effort).
_coupang_banner_cache_lock = threading.Lock()
_coupang_banner_cache: Dict[str, Any] = {"key": "", "expires_at": 0.0, "payload": None}

# Create DB and tables if missing
_store = SQLiteStore(str(DB_PATH))
_store.conn.close()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _count(conn: sqlite3.Connection, table_expr: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table_expr}").fetchone()[0]
    except Exception:
        return 0


def _minmax(conn: sqlite3.Connection, table: str) -> dict:
    try:
        row = conn.execute(f"SELECT MIN(date), MAX(date) FROM {table}").fetchone()
        return {"min": row[0], "max": row[1]}
    except Exception:
        return {"min": None, "max": None}


def _distinct_code_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(DISTINCT code) FROM {table}").fetchone()[0]
    except Exception:
        return 0


def _missing_codes(conn: sqlite3.Connection, table: str) -> int:
    try:
        row = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM universe_members u
            LEFT JOIN (SELECT DISTINCT code FROM {table}) t
            ON u.code = t.code
            WHERE t.code IS NULL
            """
        ).fetchone()
        return row[0]
    except Exception:
        return 0


def _pgrep(pattern: str) -> bool:
    try:
        res = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False)
        return res.returncode == 0
    except Exception:
        return False


def _load_filter_toggles(path: Path = FILTER_TOGGLE_PATH) -> Dict[str, bool]:
    defaults = {key: True for key in FILTER_TOGGLE_KEYS}
    if not path.exists():
        return defaults
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return defaults
    if not isinstance(payload, dict):
        return defaults
    out = defaults.copy()
    for key in FILTER_TOGGLE_KEYS:
        if key in payload:
            out[key] = bool(payload.get(key))
    return out


def _save_filter_toggles(toggles: Dict[str, bool], path: Path = FILTER_TOGGLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: bool(toggles.get(key, True)) for key in FILTER_TOGGLE_KEYS}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_selection_notify_state(path: Path = SELECTION_NOTIFY_PATH) -> Dict[str, Any]:
    default = {
        "last_codes": [],
        "last_ts": 0.0,  # last time we sent "new codes" notification (cooldown gate)
        "pending_codes": [],
        "last_sent_codes": [],  # last codes we recommended (new or repeat)
        "last_sent_ts": 0.0,  # last time we sent any recommendation
    }
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    if not isinstance(payload, dict):
        return default
    out = default.copy()
    for key in default:
        if key in payload:
            out[key] = payload.get(key)
    return out


def _save_selection_notify_state(payload: Dict[str, Any], path: Path = SELECTION_NOTIFY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = {
        "last_codes": payload.get("last_codes") or [],
        "last_ts": float(payload.get("last_ts") or 0.0),
        "pending_codes": payload.get("pending_codes") or [],
        "last_sent_codes": payload.get("last_sent_codes") or [],
        "last_sent_ts": float(payload.get("last_sent_ts") or 0.0),
    }
    path.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")


def _async_notify(settings: Dict[str, Any], message: str) -> None:
    if not message:
        return
    thread = threading.Thread(target=maybe_notify, args=(settings, message), daemon=True)
    thread.start()


def _format_candidate_line(code: str, info: Dict[str, Any]) -> str:
    name = info.get("name") or ""
    close = _safe_float(info.get("close"))
    disp = _safe_float(info.get("disparity"))
    parts = [code, name]
    if close is not None:
        parts.append(f"{close:,.0f}")
    if disp is not None:
        parts.append(f"disp={disp * 100:+.2f}%")
    return " ".join([p for p in parts if p])


def _maybe_notify_selection(settings: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
    candidates = snapshot.get("candidates") or []
    now_ts = time.time()
    state_exists = SELECTION_NOTIFY_PATH.exists()
    if not candidates:
        state = _load_selection_notify_state()
        _save_selection_notify_state({
            "last_codes": [],
            "last_ts": float(state.get("last_ts") or 0.0),
            "pending_codes": [],
            "last_sent_codes": state.get("last_sent_codes") or [],
            "last_sent_ts": float(state.get("last_sent_ts") or 0.0),
        })
        return
    codes = [str(c.get("code") or "").zfill(6) for c in candidates if c.get("code")]
    if not codes:
        return

    state = _load_selection_notify_state()
    last_codes = set(state.get("last_codes") or [])
    pending = list(state.get("pending_codes") or [])
    last_sent_codes = list(state.get("last_sent_codes") or [])
    last_sent_ts = float(state.get("last_sent_ts") or 0.0)

    def _system_error_line() -> str:
        err = _selection_notify_health.get("last_error")
        err_ts = float(_selection_notify_health.get("last_error_ts") or 0.0)
        ok_ts = float(_selection_notify_health.get("last_ok_ts") or 0.0)
        if err and err_ts >= ok_ts and err_ts > 0 and (now_ts - err_ts) < 86400:
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(err_ts))
            text = str(err).strip().replace("\n", " ")
            if len(text) > 180:
                text = text[:180] + "..."
            return f"system_error: {text} (at {when})"
        return "system_error: none"

    def _site_line() -> str:
        url = settings.get("site_url") or settings.get("site", {}).get("url") or ""
        url = str(url).strip()
        if not url:
            return ""
        return f"site: {url}"

    info_map = {str(c.get("code") or "").zfill(6): c for c in candidates}

    def _build_message(label: str, rec_codes: list[str]) -> str:
        header = f"[selection] 추천({label}) total={len(codes)}"
        if snapshot.get("date"):
            header += f" date={snapshot.get('date')}"
        lines = [header]
        for idx, code in enumerate(rec_codes[:12], start=1):
            line = _format_candidate_line(code, info_map.get(code, {}))
            if line:
                lines.append(f"{idx}. {line}")
        if len(lines) == 1:
            lines.append("(추천 종목 없음)")
        lines.append(_system_error_line())
        site = _site_line()
        if site:
            lines.append(site)
        return "\n".join(lines)

    # Initialize state without sending notifications on first run
    if not last_codes and not state_exists:
        seed = codes[: min(12, len(codes))]
        _save_selection_notify_state({
            "last_codes": codes,
            "last_ts": float(state.get("last_ts") or 0.0),
            "pending_codes": [],
            "last_sent_codes": seed,
            "last_sent_ts": now_ts,
        })
        return

    new_codes = [c for c in codes if c not in last_codes]
    if new_codes:
        pending = list(dict.fromkeys(pending + new_codes))

    # 1) New codes: notify as soon as cooldown allows.
    if pending:
        cooldown_ok = now_ts - float(state.get("last_ts") or 0.0) >= SELECTION_NOTIFY_COOLDOWN_SEC
        if not cooldown_ok:
            _save_selection_notify_state({
                "last_codes": codes,
                "last_ts": float(state.get("last_ts") or 0.0),
                "pending_codes": pending,
                "last_sent_codes": last_sent_codes,
                "last_sent_ts": last_sent_ts,
            })
            return

        msg = _build_message("신규", pending)
        _async_notify(settings, msg)
        _save_selection_notify_state({
            "last_codes": codes,
            "last_ts": now_ts,
            "pending_codes": [],
            "last_sent_codes": pending[:12],
            "last_sent_ts": now_ts,
        })
        return

    # 2) No new codes: periodically re-recommend the last sent (keeps the channel alive without spamming).
    repeat_sec = SELECTION_NOTIFY_REPEAT_SEC
    repeat_due = repeat_sec > 0 and (now_ts - last_sent_ts) >= repeat_sec
    if repeat_due and last_sent_codes:
        rec = [c for c in last_sent_codes if c in info_map]
        if not rec:
            rec = codes[: min(12, len(codes))]
        msg = _build_message("최근", rec)
        _async_notify(settings, msg)
        _save_selection_notify_state({
            "last_codes": codes,
            "last_ts": float(state.get("last_ts") or 0.0),
            "pending_codes": [],
            "last_sent_codes": rec[:12],
            "last_sent_ts": now_ts,
        })
        return

    state["last_codes"] = codes
    _save_selection_notify_state(state)


def _disabled_response():
    return jsonify({"error": "disabled"}), 404


def _selection_notify_loop() -> None:
    interval = SELECTION_NOTIFY_INTERVAL_SEC
    if interval <= 0:
        return
    while True:
        try:
            settings = load_settings()
            conn = get_conn()
            try:
                _build_selection_snapshot(conn, settings, force=True, notify=True)
                _selection_notify_health["last_ok_ts"] = time.time()
            finally:
                conn.close()
        except Exception as exc:
            _selection_notify_health["last_error"] = str(exc)
            _selection_notify_health["last_error_ts"] = time.time()
            logging.warning("selection notify loop error: %s", exc)
        time.sleep(max(10, interval))


def _verify_toggle_password(payload: Dict[str, Any]) -> bool:
    # Empty password means toggles are disabled (safer default)
    if not KIS_TOGGLE_PASSWORD:
        return False
    pw = str(payload.get("password") or "")
    return pw == KIS_TOGGLE_PASSWORD


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        num = float(value)
        if num != num or num in (float("inf"), float("-inf")):
            return None
        return num
    except Exception:
        return None


def _selection_sector_of(row: pd.Series) -> str:
    for key in ("industry_name", "sector_name"):
        value = row.get(key)
        try:
            if value is None or pd.isna(value):
                continue
        except Exception:
            pass
        value = str(value).strip()
        if value:
            return value
    return "UNKNOWN"


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _get_kis_price_client(settings: Dict[str, Any]) -> Optional[KISPriceClient]:
    global _kis_price_client, _kis_price_client_error
    if _kis_price_client_error:
        return None
    if _kis_price_client is not None:
        return _kis_price_client
    with _kis_price_client_lock:
        if _kis_price_client is not None or _kis_price_client_error:
            return _kis_price_client
        try:
            _kis_price_client = KISPriceClient(settings)
        except Exception as exc:
            _kis_price_client_error = exc
            logging.warning("KISPriceClient init failed; live prices disabled: %s", exc)
            return None
    return _kis_price_client


def _normalize_code(code: Any) -> str:
    return str(code or "").strip().zfill(6)


def _set_candidate_ws_targets(codes: list[str]) -> None:
    codes = [_normalize_code(c) for c in (codes or []) if str(c or "").strip()]
    targets = set(codes)
    with _candidate_ws_lock:
        _candidate_ws_targets.clear()
        _candidate_ws_targets.update(targets)
    if targets:
        _ensure_candidate_ws_thread()


def _ensure_candidate_ws_thread() -> None:
    global _candidate_ws_thread
    with _candidate_ws_lock:
        if _candidate_ws_thread is not None and _candidate_ws_thread.is_alive():
            return

        def _thread_main():
            try:
                asyncio.run(_candidate_ws_loop())
            except Exception as exc:
                logging.exception("candidate ws loop crashed: %s", exc)
                with _candidate_ws_lock:
                    _candidate_ws_health["last_error"] = str(exc)
                    _candidate_ws_health["last_error_ts"] = time.time()

        _candidate_ws_thread = threading.Thread(target=_thread_main, daemon=True)
        _candidate_ws_thread.start()


async def _candidate_ws_loop() -> None:
    from src.monitor.state_store import StateStore
    from src.monitor.ws_client import KISWebSocketClient

    settings = load_settings()
    state = StateStore("data/candidate_ws_state.json")

    def on_tick(code: str, price: float, ts: Optional[str] = None, source: str = "ws") -> None:
        if not code or price is None:
            return
        try:
            price_f = float(price)
        except Exception:
            return
        if price_f <= 0:
            return
        with _candidate_ws_lock:
            _candidate_ws_prices[_normalize_code(code)] = {
                "price": price_f,
                "ts": ts,
                "source": source,
                "updated_at": time.time(),
            }
            _candidate_ws_health["last_ok_ts"] = time.time()

    ws_client = KISWebSocketClient(settings, state, on_tick)

    async def target_loop():
        last_seen: set[str] = set()
        while True:
            with _candidate_ws_lock:
                targets = set(_candidate_ws_targets)
            if targets != last_seen:
                logging.info("candidate ws targets=%s", ",".join(sorted(targets)) if targets else "(none)")
                last_seen = targets
            await ws_client.set_targets(targets)
            await asyncio.sleep(1.0)

    await asyncio.gather(ws_client.run_forever(), target_loop())


def _latest_price_map(conn: sqlite3.Connection, codes: list[str]) -> Dict[str, Dict[str, Any]]:
    if not codes:
        return {}
    placeholder = ",".join("?" * len(codes))
    sql = f"""
        SELECT d.code, d.close, d.date
        FROM daily_price d
        JOIN (
            SELECT code, MAX(date) AS max_date
            FROM daily_price
            WHERE code IN ({placeholder})
            GROUP BY code
        ) m
        ON d.code = m.code AND d.date = m.max_date
    """
    rows = conn.execute(sql, tuple(codes)).fetchall()
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        out[row[0]] = {"close": row[1], "date": row[2]}
    return out


_COUPANG_BANNER_KEYWORDS = [
    # Daily necessities / household staples (생필품 위주).
    "화장지",
    "물티슈",
    "키친타월",
    "세탁세제",
    "섬유유연제",
    "주방세제",
    "락스",
    "쓰레기봉투",
    "고무장갑",
    "랩",
    "위생장갑",
    "치약",
    "칫솔",
    "샴푸",
    "바디워시",
    "핸드워시",
    "생수",
    "라면",
    "즉석밥",
    "커피믹스",
]

_COUPANG_BANNER_CTA_VARIANTS = ["최저가 보기", "배송 일정 확인", "리뷰 보고 선택"]


def _encode_coupang_component(value: Any) -> str:
    # Match JS encodeURIComponent for signature correctness (space -> %20, keep -_.!~*'()).
    return quote(str(value or ""), safe="-_.!~*'()")


def _coupang_signed_date(now_ts: Optional[float] = None) -> str:
    ts = float(now_ts if now_ts is not None else time.time())
    return time.strftime("%y%m%dT%H%M%SZ", time.gmtime(ts))


def _coupang_hmac_sha256_hex(secret_key: str, message: str) -> str:
    return hmac.new(secret_key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def _format_krw(value: Any) -> str:
    num = _safe_float(value)
    if num is None or num <= 0:
        return ""
    return f"{int(round(num)):,}원"


def _coupang_fetch_search_products(
    access_key: str,
    secret_key: str,
    *,
    keyword: str,
    limit: int = 3,
    sub_id: str = "cp-banner",
    timeout_sec: float = 8.0,
) -> list[dict]:
    path = "/v2/providers/affiliate_open_api/apis/openapi/v1/products/search"
    query = (
        f"keyword={_encode_coupang_component(keyword)}"
        f"&limit={_encode_coupang_component(str(limit))}"
        f"&subId={_encode_coupang_component(sub_id)}"
    )
    signed_date = _coupang_signed_date()
    message = f"{signed_date}GET{path}{query}"
    signature = _coupang_hmac_sha256_hex(secret_key, message)
    authorization = (
        "CEA algorithm=HmacSHA256, "
        f"access-key={access_key}, "
        f"signed-date={signed_date}, "
        f"signature={signature}"
    )

    url = f"https://api-gateway.coupang.com{path}?{query}"
    resp = requests.get(url, headers={"Authorization": authorization}, timeout=timeout_sec)
    if not resp.ok:
        raise RuntimeError(f"Coupang API error: {resp.status_code} {resp.text}".strip())

    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError("Coupang API error: invalid JSON") from exc

    if isinstance(data, dict) and data.get("rCode") and str(data.get("rCode")) != "0":
        raise RuntimeError(str(data.get("rMessage") or "Coupang API error"))

    products = data.get("data", {}).get("productData", []) if isinstance(data, dict) else []
    return products if isinstance(products, list) else []


def _coupang_pick_keyword(now_ts: float) -> str:
    pool = _COUPANG_BANNER_KEYWORDS
    if not pool:
        return "생필품"
    # Deterministic pick per 30-minute slot -> reduces API calls under load.
    slot = int(now_ts // 1800)
    return pool[slot % len(pool)] or "생필품"


def _coupang_map_products(products: list[dict], *, badge: str = "생필품") -> list[Dict[str, Any]]:
    items: list[Dict[str, Any]] = []
    for index, product in enumerate(products or []):
        if not isinstance(product, dict):
            continue
        title = str(product.get("productName") or "").strip()
        image = str(product.get("productImage") or "").strip()
        link = str(product.get("productUrl") or "").strip()
        if not title or not image or not link:
            continue

        discount = _safe_float(product.get("productDiscountRate"))
        discount_rate = int(round(discount)) if discount is not None and discount > 0 else None
        rocket = bool(
            product.get("rocketWow")
            or product.get("rocket")
            or product.get("isRocket")
            or product.get("isRocketWow")
            or (str(product.get("rocketDeliveryType") or "").upper() == "ROCKET")
        )
        free_shipping = bool(product.get("isFreeShipping") or product.get("freeShipping"))
        shipping_tag = "로켓배송" if rocket else ("무료배송" if free_shipping else "")

        rating_count_raw = product.get("ratingCount", product.get("reviewCount"))
        rating_count = int(_safe_float(rating_count_raw) or 0)
        rating_raw = product.get("rating", product.get("ratingAverage", product.get("ratingScore")))
        rating = _safe_float(rating_raw)

        meta_parts: list[str] = []
        if rating is not None and rating > 0:
            meta_parts.append(f"★{rating:.1f}")
        if rating_count > 0:
            meta_parts.append(f"리뷰 {rating_count:,}개")
        if shipping_tag:
            meta_parts.append(shipping_tag)
        category_name = str(product.get("categoryName") or "").strip()
        if category_name:
            meta_parts.append(category_name)

        items.append(
            {
                "title": title,
                "image": image,
                "link": link,
                "price": _format_krw(product.get("productPrice")),
                "meta": " · ".join([p for p in meta_parts if p]) if meta_parts else "",
                "badge": badge,
                "discountRate": discount_rate,
                "cta": _COUPANG_BANNER_CTA_VARIANTS[index % len(_COUPANG_BANNER_CTA_VARIANTS)]
                if _COUPANG_BANNER_CTA_VARIANTS
                else "바로 보기",
                "shippingTag": shipping_tag,
                "ratingCount": rating_count if rating_count > 0 else None,
                "rating": rating if rating is not None and rating > 0 else None,
            }
        )
    return items

def _dummy_portfolio_positions(conn: sqlite3.Connection, max_positions: int = 8) -> list[Dict[str, Any]]:
    _ensure_latest_table(conn)
    sql = """
        SELECT l.code, l.date, l.close, l.amount,
               u.name, u.market, s.sector_name, s.industry_name
        FROM daily_price_latest l
        LEFT JOIN universe_members u ON l.code = u.code
        LEFT JOIN sector_map s ON l.code = s.code
        ORDER BY l.amount DESC
        LIMIT ?
    """
    df = pd.read_sql_query(sql, conn, params=(max_positions,))
    if df.empty:
        return []
    budget_per_pos = 10_000_000
    now = pd.Timestamp.utcnow().isoformat()
    records: list[Dict[str, Any]] = []
    for _, row in df.iterrows():
        close = _safe_float(row.get("close")) or 0.0
        qty = max(1, int(budget_per_pos / close)) if close > 0 else 0
        avg_price = close * 0.97 if close > 0 else 0.0
        records.append({
            "code": row.get("code"),
            "name": row.get("name") or row.get("code"),
            "qty": qty,
            "avg_price": avg_price,
            "entry_date": row.get("date"),
            "updated_at": now,
            "market": row.get("market"),
            "sector_name": row.get("sector_name"),
            "industry_name": row.get("industry_name"),
            "last_close": close,
            "last_date": row.get("date"),
        })
    return records


def _build_portfolio(conn: sqlite3.Connection) -> Dict[str, Any]:
    records = _dummy_portfolio_positions(conn, max_positions=10)
    total_value = 0.0
    total_cost = 0.0
    for row in records:
        qty = float(row.get("qty") or 0)
        avg_price = float(row.get("avg_price") or 0)
        last_close = float(row.get("last_close") or 0)
        cost = qty * avg_price if qty and avg_price else 0.0
        market_value = qty * last_close if qty and last_close else 0.0
        pnl = market_value - cost
        pnl_pct = (pnl / cost * 100) if cost else None
        row.update({
            "market_value": market_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })
        total_value += market_value
        total_cost += cost
    totals = {
        "positions_value": total_value,
        "cost": total_cost,
        "pnl": total_value - total_cost if total_cost else None,
        "pnl_pct": ((total_value - total_cost) / total_cost * 100) if total_cost else None,
    }
    return {"positions": records, "totals": totals}


def _build_account_summary(conn: sqlite3.Connection) -> Dict[str, Any]:
    portfolio = _build_portfolio(conn)
    totals = portfolio.get("totals", {})
    cash = 50_000_000.0
    positions_value = float(totals.get("positions_value") or 0.0)
    total_assets = cash + positions_value
    total_pnl = totals.get("pnl")
    total_pnl_pct = totals.get("pnl_pct")
    data_health = {}
    return {
        "connected": False,
        "connected_at": pd.Timestamp.utcnow().isoformat(),
        "data_health": data_health,
        "summary": {
            "total_assets": total_assets,
            "cash": cash,
            "positions_value": positions_value,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
        },
    }


def _build_plans(conn: sqlite3.Connection, selection_data: Dict[str, Any]) -> Dict[str, Any]:
    candidates = selection_data.get("candidates") or []
    exec_date = selection_data.get("date")
    budget_per_pos = 10_000_000
    buys = []
    for idx, row in enumerate(candidates[:12], start=1):
        price = _safe_float(row.get("close")) or 0.0
        qty = max(1, int(budget_per_pos / price)) if price > 0 else 0
        buys.append({
            "id": idx,
            "code": row.get("code"),
            "name": row.get("name"),
            "planned_price": price,
            "qty": qty,
        })
    return {"exec_date": exec_date, "buys": buys, "sells": [], "counts": {"buys": len(buys), "sells": 0}}


def _ensure_latest_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_price_latest (
            code TEXT PRIMARY KEY,
            date TEXT,
            close REAL,
            amount REAL,
            ma25 REAL,
            disparity REAL
        );
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_daily_price_latest
        AFTER INSERT ON daily_price
        BEGIN
            INSERT INTO daily_price_latest(code, date, close, amount, ma25, disparity)
            VALUES (NEW.code, NEW.date, NEW.close, NEW.amount, NEW.ma25, NEW.disparity)
            ON CONFLICT(code) DO UPDATE SET
                date=excluded.date,
                close=excluded.close,
                amount=excluded.amount,
                ma25=excluded.ma25,
                disparity=excluded.disparity
            WHERE excluded.date >= daily_price_latest.date OR daily_price_latest.date IS NULL;
        END;
        """
    )
    row = conn.execute("SELECT COUNT(*) FROM daily_price_latest").fetchone()
    if row and row[0] == 0:
        conn.execute(
            """
            INSERT INTO daily_price_latest(code, date, close, amount, ma25, disparity)
            SELECT d.code, d.date, d.close, d.amount, d.ma25, d.disparity
            FROM daily_price d
            JOIN (
                SELECT code, MAX(date) AS max_date
                FROM daily_price
                GROUP BY code
            ) m
            ON d.code = m.code AND d.date = m.max_date
            """
        )
    conn.commit()


def _build_selection_snapshot(
    conn: sqlite3.Connection,
    settings: Dict[str, Any],
    force: bool = False,
    notify: bool = False,
) -> Dict[str, Any]:
    now_ts = time.time()

    def _cache_set(payload: Dict[str, Any]) -> Dict[str, Any]:
        # Compute time can exceed cache TTL; stamp cache at commit time.
        _selection_cache.update({"ts": time.time(), "data": payload})
        return payload

    cached = _selection_cache.get("data")
    if not force and cached and now_ts - _selection_cache.get("ts", 0) < 30:
        return cached

    params = load_strategy(settings)
    min_amount = float(getattr(params, "min_amount", 0) or 0)
    liquidity_rank = int(getattr(params, "liquidity_rank", 0) or 0)
    buy_kospi = float(getattr(params, "buy_kospi", 0) or 0)
    buy_kosdaq = float(getattr(params, "buy_kosdaq", 0) or 0)
    max_positions = int(getattr(params, "max_positions", 10) or 10)
    # 추천 항목에서는 동일 섹터에서 상위 종목 1개만 허용한다.
    max_per_sector = 1
    rank_mode = str(getattr(params, "rank_mode", "amount") or "amount").lower()
    entry_mode = str(getattr(params, "entry_mode", "mean_reversion") or "mean_reversion").lower()
    take_profit_ret = float(getattr(params, "take_profit_ret", 0) or 0)
    trend_filter = bool(getattr(params, "trend_ma25_rising", False))
    filter_toggles = _load_filter_toggles()
    min_enabled = filter_toggles.get("min_amount", True)
    liq_enabled = filter_toggles.get("liquidity", True)
    disp_enabled = filter_toggles.get("disparity", True)

    universe_df = pd.read_sql_query("SELECT code, name, market, group_name FROM universe_members", conn)
    codes = universe_df["code"].dropna().astype(str).tolist()
    if not codes:
        data = {"date": None, "stages": [], "candidates": [], "summary": {"total": 0}, "filter_toggles": filter_toggles}
        return _cache_set(data)

    placeholder = ",".join("?" * len(codes))
    sql = f"""
        SELECT code, date, close, amount, ma25, disparity
        FROM (
            SELECT code, date, close, amount, ma25, disparity,
                   ROW_NUMBER() OVER (PARTITION BY code ORDER BY date DESC) AS rn
            FROM daily_price
            WHERE code IN ({placeholder})
        )
        WHERE rn <= 4
    """
    df = pd.read_sql_query(sql, conn, params=codes)
    if df.empty:
        data = {"date": None, "stages": [], "candidates": [], "summary": {"total": len(codes)}, "filter_toggles": filter_toggles}
        return _cache_set(data)

    df = df.sort_values(["code", "date"])
    df["ma25_prev"] = df.groupby("code")["ma25"].shift(1)
    df["ret3"] = df.groupby("code")["close"].pct_change(3)
    latest = df.groupby("code").tail(1).copy()
    latest = latest.merge(universe_df, on="code", how="left")
    try:
        sector_df = pd.read_sql_query("SELECT code, sector_name, industry_name FROM sector_map", conn)
        latest = latest.merge(sector_df, on="code", how="left")
    except Exception:
        pass

    total = len(latest)

    # Basic data health diagnostics (helps explain "0 candidates" cases)
    data_health = {
        "latest_rows": int(total),
        "amount_zero": int((latest["amount"].fillna(0) <= 0).sum()) if "amount" in latest.columns else None,
        "amount_nan": int(latest["amount"].isna().sum()) if "amount" in latest.columns else None,
        "ma25_nan": int(latest["ma25"].isna().sum()) if "ma25" in latest.columns else None,
        "disparity_nan": int(latest["disparity"].isna().sum()) if "disparity" in latest.columns else None,
    }
    stage_min = latest[latest["amount"] >= min_amount] if min_amount and min_enabled else latest
    stage_liq = stage_min.sort_values("amount", ascending=False)
    if liquidity_rank and liq_enabled:
        stage_liq = stage_liq.head(liquidity_rank)

    def _pass_disparity(row) -> bool:
        market = row.get("market") or "KOSPI"
        threshold = buy_kospi if "KOSPI" in market else buy_kosdaq
        try:
            disp = float(row.get("disparity") or 0)
            if entry_mode == "trend_follow":
                r3 = float(row.get("ret3") or 0)
                return disp >= threshold and r3 >= 0
            return disp <= threshold
        except Exception:
            return False

    stage_disp = stage_liq[stage_liq.apply(_pass_disparity, axis=1)] if disp_enabled else stage_liq
    if disp_enabled and trend_filter:
        stage_disp = stage_disp[stage_disp["ma25_prev"].notna() & (stage_disp["ma25"] > stage_disp["ma25_prev"])]

    stage_ranked = stage_disp.copy()
    if rank_mode == "score":
        if entry_mode == "trend_follow":
            stage_ranked["score"] = (
                (stage_ranked["disparity"].fillna(0).astype(float))
                + (0.8 * (stage_ranked["ret3"].fillna(0).astype(float)))
                + (0.05 * np.log1p(stage_ranked["amount"].fillna(0).astype(float).clip(lower=0)))
            )
        else:
            stage_ranked["score"] = (
                (-stage_ranked["disparity"].fillna(0).astype(float))
                + (0.8 * (-stage_ranked["ret3"].fillna(0).astype(float)))
                + (0.05 * np.log1p(stage_ranked["amount"].fillna(0).astype(float).clip(lower=0)))
            )
        stage_ranked = stage_ranked.sort_values("score", ascending=False)
    else:
        stage_ranked = stage_ranked.sort_values("amount", ascending=False)

    final_rows = []
    sector_counts: Dict[str, int] = {}
    for _, row in stage_ranked.iterrows():
        sec = _selection_sector_of(row)
        if max_per_sector and max_per_sector > 0 and sector_counts.get(sec, 0) >= max_per_sector:
            continue
        final_rows.append(row)
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if len(final_rows) >= max_positions:
            break
    final = pd.DataFrame(final_rows) if final_rows else stage_ranked.head(0).copy()
    if not final.empty:
        final["rank"] = range(1, len(final) + 1)

    def _pack(df: pd.DataFrame, sort_by: str | None = None, ascending: bool = False) -> list[Dict[str, Any]]:
        if df is None or df.empty:
            return []
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)
        df = df.replace([np.inf, -np.inf], np.nan)
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "code": row.get("code"),
                "name": row.get("name"),
                "market": row.get("market"),
                "amount": _safe_float(row.get("amount")),
                "close": _safe_float(row.get("close")),
                "disparity": _safe_float(row.get("disparity")),
            })
        return rows

    candidates = []
    for _, row in final.iterrows():
        code = row.get("code")
        # Selection/Final 가격은 일봉 종가(DB) 기준으로만 반환한다.
        final_p = _safe_float(row.get("close"))

        candidates.append({
            "rank": int(row.get("rank") or 0),
            "code": code,
            "name": row.get("name"),
            "market": row.get("market"),
            "close": final_p,
            "amount": _safe_float(row.get("amount")),
            "disparity": _safe_float(row.get("disparity")),
            "sector_name": row.get("sector_name"),
            "industry_name": row.get("industry_name"),
        })

    # Update candidate-only realtime WS targets.
    try:
        _set_candidate_ws_targets([c.get("code") for c in candidates if isinstance(c, dict)])
    except Exception:
        pass

    stages = [
        {"key": "universe", "label": "Universe", "count": total, "value": len(codes), "enabled": True},
        {"key": "min_amount", "label": "Amount Filter", "count": len(stage_min), "value": min_amount, "enabled": min_enabled},
        {"key": "liquidity", "label": "Liquidity Rank", "count": len(stage_liq), "value": liquidity_rank, "enabled": liq_enabled},
        {"key": "disparity", "label": "Disparity Threshold", "count": len(stage_disp), "value": {"kospi": buy_kospi, "kosdaq": buy_kosdaq}, "enabled": disp_enabled},
        {"key": "final", "label": "Max Positions", "count": len(final), "value": max_positions, "enabled": True},
    ]

    data = {
        "date": latest["date"].max(),
        "stages": stages,
        "candidates": candidates,
        "stage_items": {
            "min_amount": _pack(stage_min),
            "liquidity": _pack(stage_liq),
            "disparity": _pack(stage_disp),
            "final": _pack(final, sort_by="rank", ascending=True),
        },
        "summary": {
            "total": total,
            "final": len(final),
            "trend_filter": trend_filter,
            "rank_mode": rank_mode,
            "entry_mode": entry_mode,
            "max_positions": max_positions,
            "max_per_sector": max_per_sector,
        },
        "filter_toggles": filter_toggles,
        "pricing": {"sell_rules": {"take_profit_ret": take_profit_ret}},
    }
    # Ensure JSON-safe values (no NaN/Inf)
    data = _json_sanitize(data)
    if notify:
        _maybe_notify_selection(settings, data)
    return _cache_set(data)


app = Flask(__name__, static_folder=str(FRONTEND_DIST), static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})


def _start_background_tasks() -> None:
    if SELECTION_NOTIFY_INTERVAL_SEC <= 0:
        return
    thread = threading.Thread(target=_selection_notify_loop, daemon=True)
    thread.start()


_start_background_tasks()

# ---------- Static ----------
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/bnf")
def serve_index_bnf():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path: str):
    if (FRONTEND_DIST / path).exists():
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


@app.route("/bnf/<path:path>")
def serve_static_bnf(path: str):
    if (FRONTEND_DIST / path).exists():
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


@app.post("/client_error")
def client_error():
    """Client-side error collector (best-effort).
    Safety: cap payload size and rotate log to avoid disk fill.
    """
    payload = request.get_json(silent=True) or {}
    # Hard caps (can override by env)
    max_payload_bytes = int(os.getenv("CLIENT_ERROR_MAX_BYTES", "100000"))  # 100KB
    max_log_bytes = int(float(os.getenv("CLIENT_ERROR_LOG_MAX_MB", "10")) * 1024 * 1024)

    # Reject overly large bodies (still return ok to avoid client retry loops)
    try:
        raw = json.dumps(payload, ensure_ascii=False)
    except Exception:
        raw = "{}"
    if len(raw.encode("utf-8")) > max_payload_bytes:
        return jsonify({"status": "ignored", "reason": "payload_too_large"})

    CLIENT_ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        if CLIENT_ERROR_LOG.exists() and CLIENT_ERROR_LOG.stat().st_size > max_log_bytes:
            rotated = CLIENT_ERROR_LOG.with_suffix(CLIENT_ERROR_LOG.suffix + ".1")
            try:
                if rotated.exists():
                    rotated.unlink()
            except Exception:
                pass
            try:
                CLIENT_ERROR_LOG.rename(rotated)
            except Exception:
                pass
    except Exception:
        pass

    with CLIENT_ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(raw + "\n")
    return jsonify({"status": "ok"})


# ---------- API ----------
@app.get("/universe")
def universe():
    """Universe list (KOSPI200 + KOSDAQ150)."""
    conn = get_conn()
    sector = request.args.get("sector")
    if sector:
        if sector.upper() == "UNKNOWN":
            where = "s.sector_name IS NULL"
            params: Tuple[Any, ...] = ()
        else:
            where = "s.sector_name = ?"
            params = (sector,)
    else:
        where = "1=1"
        params = ()

    try:
        df = pd.read_sql_query(
            f"""
            SELECT u.code, u.name, u.market, u.group_name as 'group',
                   COALESCE(s.sector_name, 'UNKNOWN') AS sector_name,
                   s.industry_name
            FROM universe_members u
            LEFT JOIN sector_map s ON u.code = s.code
            WHERE {where}
            ORDER BY u.code
            """,
            conn,
            params=params,
        )
    except Exception:
        df = pd.read_sql_query(
            "SELECT code, name, market, group_name as 'group' FROM universe_members ORDER BY code",
            conn,
        )
    return jsonify(df.to_dict(orient="records"))


@app.get("/sectors")
def sectors():
    conn = get_conn()
    try:
        df = pd.read_sql_query(
            """
            SELECT u.market,
                   COALESCE(s.sector_name, 'UNKNOWN') AS sector_name,
                   COUNT(*) AS count
            FROM universe_members u
            LEFT JOIN sector_map s ON u.code = s.code
            GROUP BY u.market, COALESCE(s.sector_name, 'UNKNOWN')
            ORDER BY u.market, count DESC, sector_name
            """,
            conn,
        )
    except Exception:
        df = pd.DataFrame([], columns=["market", "sector_name", "count"])
    return jsonify(df.to_dict(orient="records"))


@app.get("/prices")
def prices():
    code = request.args.get("code")
    days = int(request.args.get("days", 180))
    if not code:
        return jsonify([])

    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT date, open, high, low, close, volume, amount, ma25, disparity
        FROM daily_price
        WHERE code=?
        ORDER BY date DESC
        LIMIT ?
        """,
        conn,
        params=(code, days),
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.astype(object).where(pd.notnull(df), None)
    return jsonify(df.to_dict(orient="records"))


@app.get("/prices/realtime")
def prices_realtime():
    code = request.args.get("code")
    if not code:
        return jsonify({"error": "missing_code"}), 400
    code = str(code).strip().zfill(6)
    settings = load_settings()
    price = None
    source = "db"
    date = None

    price_client = _get_kis_price_client(settings)
    if price_client is not None:
        try:
            live = price_client.get_current_price(code)
            live = _safe_float(live)
            if live is not None and live > 0:
                price = live
                source = "kis"
        except Exception as exc:
            logging.warning("realtime price fetch failed; fallback to db: %s", exc)

    if price is None:
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT close, date FROM daily_price WHERE code=? ORDER BY date DESC LIMIT 1",
                (code,),
            ).fetchone()
            if row:
                price = _safe_float(row[0])
                date = row[1]
        finally:
            conn.close()

    return jsonify({"code": code, "price": price, "source": source, "date": date})


@app.get("/selection/realtime_prices")
def selection_realtime_prices():
    codes_param = request.args.get("codes") or ""
    codes: list[str]
    if codes_param.strip():
        parts = [p.strip() for p in str(codes_param).split(",") if p.strip()]
        # basic abuse guard
        parts = parts[:50]
        codes = [_normalize_code(p) for p in parts]
    else:
        with _candidate_ws_lock:
            codes = sorted(_candidate_ws_targets)

    with _candidate_ws_lock:
        prices = {c: _candidate_ws_prices.get(c) for c in codes}
        health = dict(_candidate_ws_health)
        running = bool(_candidate_ws_thread is not None and _candidate_ws_thread.is_alive())
    return jsonify(
        {
            "codes": codes,
            "prices": prices,
            "ws": {"running": running, "health": health},
            "updated_at": time.time(),
        }
    )


@app.get("/api/coupang-banner")
def api_coupang_banner():
    """Coupang Partners banner payload (server-side only; never expose secret key to browser).

    Query params:
      - keyword: force a search keyword (optional)
      - limit: 1~10 (optional, default 3)
    """
    now_ts = time.time()
    keyword_override = str(request.args.get("keyword") or "").strip()
    try:
        limit = int(str(request.args.get("limit") or "3").strip())
    except Exception:
        limit = 3
    limit = max(1, min(10, limit))

    access_key = str(os.getenv("COUPANG_ACCESS_KEY", "") or "").strip()
    secret_key = str(os.getenv("COUPANG_SECRET_KEY", "") or "").strip()
    sub_id = str(os.getenv("COUPANG_SUB_ID", "") or "").strip() or "cp-banner"

    if not access_key or not secret_key:
        return jsonify(
            {
                "error": "missing_config",
                "keyword": keyword_override or "",
                "theme": {
                    "id": "daily-necessities",
                    "title": "생필품 추천",
                    "tagline": "쿠팡 추천 상품을 불러오려면 서버에 COUPANG 키 설정이 필요합니다.",
                    "cta": "바로 보기",
                },
                "items": [],
            }
        )

    if keyword_override:
        keyword = keyword_override
        cache_key = f"cp-banner:kw:{keyword_override}:limit:{limit}"
    else:
        keyword = _coupang_pick_keyword(now_ts)
        slot = int(now_ts // 1800)
        cache_key = f"cp-banner:slot:{slot}:limit:{limit}"

    with _coupang_banner_cache_lock:
        if (
            _coupang_banner_cache.get("key") == cache_key
            and float(_coupang_banner_cache.get("expires_at") or 0.0) > now_ts
            and isinstance(_coupang_banner_cache.get("payload"), dict)
        ):
            return jsonify(_coupang_banner_cache["payload"])

    try:
        products = _coupang_fetch_search_products(
            access_key,
            secret_key,
            keyword=keyword,
            limit=limit,
            sub_id=sub_id,
        )
        items = _coupang_map_products(products, badge="생필품")
    except Exception as exc:
        logging.warning("coupang banner fetch failed: %s", exc)
        items = []

    payload = {
        "keyword": keyword,
        "theme": {
            "id": "daily-necessities",
            "title": "생필품 추천",
            "tagline": "오늘 필요한 생활템을 모아봤습니다.",
            "cta": "쿠팡에서 보기",
        },
        "items": items,
    }

    ttl = 1800 if items else 120
    with _coupang_banner_cache_lock:
        _coupang_banner_cache["key"] = cache_key
        _coupang_banner_cache["expires_at"] = now_ts + ttl
        _coupang_banner_cache["payload"] = payload
    return jsonify(payload)


@app.get("/signals")
def signals():
    if DISABLE_TRADING_ENDPOINTS:
        return _disabled_response()
    conn = get_conn()
    try:
        df = pd.read_sql_query(
            "SELECT signal_date, code, side, qty FROM order_queue ORDER BY created_at DESC LIMIT 30",
            conn,
        )
        return jsonify(df.to_dict(orient="records"))
    except Exception:
        return jsonify([])


@app.get("/orders")
def orders():
    if DISABLE_TRADING_ENDPOINTS:
        return _disabled_response()
    conn = get_conn()
    limit = int(request.args.get("limit", 200))
    try:
        df = pd.read_sql_query(
            """
            SELECT
              signal_date, exec_date, code, side, qty, status, ord_dvsn, ord_unpr, filled_qty, avg_price, created_at, updated_at
            FROM order_queue
            ORDER BY created_at DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
        return jsonify(df.to_dict(orient="records"))
    except Exception:
        return jsonify([])


@app.get("/positions")
def positions():
    if DISABLE_TRADING_ENDPOINTS:
        return _disabled_response()
    conn = get_conn()
    portfolio = _build_portfolio(conn)
    records = portfolio.get("positions") or []
    slim = [
        {
            "code": r.get("code"),
            "name": r.get("name"),
            "qty": r.get("qty"),
            "avg_price": r.get("avg_price"),
            "entry_date": r.get("entry_date"),
            "updated_at": r.get("updated_at"),
        }
        for r in records
    ]
    return jsonify(slim)


@app.get("/portfolio")
def portfolio():
    if DISABLE_TRADING_ENDPOINTS:
        return _disabled_response()
    conn = get_conn()
    return jsonify(_build_portfolio(conn))


@app.get("/plans")
def plans():
    if DISABLE_TRADING_ENDPOINTS:
        return _disabled_response()
    conn = get_conn()
    settings = load_settings()
    selection_data = _build_selection_snapshot(conn, settings)
    return jsonify(_build_plans(conn, selection_data))


@app.get("/account")
def account():
    if DISABLE_TRADING_ENDPOINTS:
        return _disabled_response()
    conn = get_conn()
    return jsonify(_build_account_summary(conn))


@app.get("/engines")
def engines():
    if DISABLE_TRADING_ENDPOINTS:
        return _disabled_response()
    conn = get_conn()
    try:
        last_signal = conn.execute("SELECT MAX(created_at) FROM order_queue").fetchone()[0]
    except Exception:
        last_signal = None
    pending = _count(conn, "order_queue WHERE status='PENDING'")
    sent = _count(conn, "order_queue WHERE status='SENT'")
    done = _count(conn, "order_queue WHERE status='DONE'")
    monitor_running = _pgrep("src.monitor.monitor_main")
    return jsonify({
        "monitor": {"running": monitor_running},
        "trader": {"last_signal": last_signal, "pending": pending, "sent": sent, "done": done},
        "accuracy_loader": {"running": False, "pid": None, "progress": {}},
    })


@app.get("/kis_keys")
def kis_keys():
    if DISABLE_TRADING_ENDPOINTS:
        return _disabled_response()
    rows = list_kis_key_inventory()
    payload = []
    for r in rows:
        payload.append({
            "id": r.get("id"),
            "account": r.get("account_no_masked") or r.get("label"),
            "description": r.get("description"),
            "enabled": r.get("enabled"),
        })
    return jsonify(payload)


@app.post("/kis_keys/toggle")
def kis_keys_toggle():
    if DISABLE_TRADING_ENDPOINTS:
        return _disabled_response()
    payload = request.get_json(silent=True) or {}
    if not KIS_TOGGLE_PASSWORD:
        return jsonify({"error": "toggle_disabled"}), 403
    if not _verify_toggle_password(payload):
        return jsonify({"error": "invalid_password"}), 403
    try:
        idx = int(payload.get("id"))
    except Exception:
        return jsonify({"error": "invalid_id"}), 400
    if idx < 1 or idx > 50:
        return jsonify({"error": "invalid_id"}), 400
    enabled = bool(payload.get("enabled"))
    updated = set_kis_key_enabled(idx, enabled)
    out = []
    for r in updated:
        out.append({
            "id": r.get("id"),
            "account": r.get("account_no_masked") or r.get("label"),
            "description": r.get("description"),
            "enabled": r.get("enabled"),
        })
    return jsonify(out)


@app.get("/selection_filters")
def selection_filters():
    return jsonify(_load_filter_toggles())


@app.post("/selection_filters/toggle")
def selection_filters_toggle():
    payload = request.get_json(silent=True) or {}
    if not KIS_TOGGLE_PASSWORD:
        return jsonify({"error": "toggle_disabled"}), 403
    if not _verify_toggle_password(payload):
        return jsonify({"error": "invalid_password"}), 403
    key = str(payload.get("key") or "")
    if key not in FILTER_TOGGLE_KEYS:
        return jsonify({"error": "invalid_key"}), 400
    enabled = bool(payload.get("enabled"))
    toggles = _load_filter_toggles()
    toggles[key] = enabled
    _save_filter_toggles(toggles)
    _selection_cache["ts"] = 0
    return jsonify(toggles)


def _build_selection_summary(conn: sqlite3.Connection, settings: Dict[str, Any]) -> Dict[str, Any]:
    """매수 후보(Selection)만 계산해서 반환한다. (자동매매/잔고 기능 없음)"""
    params = load_strategy(settings)

    min_amount = float(getattr(params, "min_amount", 0) or 0)
    liquidity_rank = int(getattr(params, "liquidity_rank", 0) or 0)
    buy_kospi = float(getattr(params, "buy_kospi", 0) or 0)
    buy_kosdaq = float(getattr(params, "buy_kosdaq", 0) or 0)
    max_positions = int(getattr(params, "max_positions", 10) or 10)
    # 추천 항목에서는 동일 섹터에서 상위 종목 1개만 허용한다.
    max_per_sector = 1
    rank_mode = str(getattr(params, "rank_mode", "amount") or "amount").lower()
    entry_mode = str(getattr(params, "entry_mode", "mean_reversion") or "mean_reversion").lower()
    trend_filter = bool(getattr(params, "trend_ma25_rising", False))

    universe_df = pd.read_sql_query("SELECT code, name, market, group_name FROM universe_members", conn)
    codes = universe_df["code"].dropna().astype(str).tolist()
    if not codes:
        return {"date": None, "candidates": [], "summary": {"total": 0, "final": 0}}

    placeholder = ",".join("?" * len(codes))
    sql = f"""
        SELECT code, date, close, amount, ma25, disparity
        FROM (
            SELECT code, date, close, amount, ma25, disparity,
                   ROW_NUMBER() OVER (PARTITION BY code ORDER BY date DESC) AS rn
            FROM daily_price
            WHERE code IN ({placeholder})
        )
        WHERE rn <= 4
    """
    df = pd.read_sql_query(sql, conn, params=codes)
    if df.empty:
        return {"date": None, "candidates": [], "summary": {"total": len(codes), "final": 0}}

    df = df.sort_values(["code", "date"])
    df["ma25_prev"] = df.groupby("code")["ma25"].shift(1)
    df["ret3"] = df.groupby("code")["close"].pct_change(3)
    latest = df.groupby("code").tail(1).copy()
    latest = latest.merge(universe_df, on="code", how="left")
    try:
        sector_df = pd.read_sql_query("SELECT code, sector_name, industry_name FROM sector_map", conn)
        latest = latest.merge(sector_df, on="code", how="left")
    except Exception:
        pass

    total = len(latest)

    stage = latest
    if min_amount:
        stage = stage[stage["amount"] >= min_amount]

    stage = stage.sort_values("amount", ascending=False)
    if liquidity_rank:
        stage = stage.head(liquidity_rank)

    def pass_signal(row) -> bool:
        market = str(row.get("market") or "").upper()
        threshold = buy_kospi if "KOSPI" in market else buy_kosdaq
        try:
            disp = float(row.get("disparity") or 0)
            r3 = float(row.get("ret3") or 0)
        except Exception:
            return False

        if entry_mode == "trend_follow":
            return disp >= threshold and r3 >= 0
        return disp <= threshold

    stage = stage[stage.apply(pass_signal, axis=1)]
    if trend_filter:
        stage = stage[stage["ma25_prev"].notna() & (stage["ma25"] > stage["ma25_prev"])]

    ranked = stage.copy()
    if rank_mode == "score":
        if entry_mode == "trend_follow":
            ranked["score"] = (
                (ranked["disparity"].fillna(0).astype(float))
                + (0.8 * (ranked["ret3"].fillna(0).astype(float)))
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

    latest_date = latest["date"].max()
    cols = ["code", "name", "market", "amount", "close", "disparity", "rank", "sector_name", "industry_name"]
    for c in cols:
        if c not in final.columns:
            final[c] = None
    candidates = (
        final[cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna("")
        .to_dict(orient="records")
    )

    return {
        "date": latest_date,
        "candidates": candidates,
        "summary": {
            "total": int(total),
            "final": int(len(candidates)),
            "entry_mode": entry_mode,
            "rank_mode": rank_mode,
            "liquidity_rank": liquidity_rank,
            "min_amount": min_amount,
            "buy_thresholds": {"kospi": buy_kospi, "kosdaq": buy_kosdaq},
            "trend_filter": trend_filter,
            "max_positions": max_positions,
            "max_per_sector": max_per_sector,
        },
    }


@app.get("/selection")
def selection():
    conn = get_conn()
    settings = load_settings()
    return jsonify(_build_selection_snapshot(conn, settings))


@app.get("/status")
def status():
    conn = get_conn()
    out = {
        "universe": {"total": _count(conn, "universe_members")},
        "daily_price": {
            "rows": _count(conn, "daily_price"),
            "codes": _distinct_code_count(conn, "daily_price"),
            "missing_codes": _missing_codes(conn, "daily_price"),
            "date": _minmax(conn, "daily_price"),
        },
        "jobs": {"recent": _count(conn, "job_runs")},
    }
    return jsonify(out)


@app.get("/jobs")
def jobs():
    conn = get_conn()
    limit = int(request.args.get("limit", 20))
    df = pd.read_sql_query("SELECT * FROM job_runs ORDER BY started_at DESC LIMIT ?", conn, params=(limit,))
    return jsonify(df.to_dict(orient="records"))


@app.get("/strategy")
def strategy():
    settings = load_settings()
    params = load_strategy(settings)
    return jsonify(
        {
            "entry_mode": params.entry_mode,
            "liquidity_rank": params.liquidity_rank,
            "min_amount": params.min_amount,
            "rank_mode": params.rank_mode,
            "disparity_buy_kospi": params.buy_kospi,
            "disparity_buy_kosdaq": params.buy_kosdaq,
            "disparity_sell": params.sell_disparity,
            "take_profit_ret": params.take_profit_ret,
            "stop_loss": params.stop_loss,
            "max_holding_days": params.max_holding_days,
            "max_positions": params.max_positions,
            "max_per_sector": params.max_per_sector,
            "trend_ma25_rising": params.trend_ma25_rising,
            "selection_horizon_days": params.selection_horizon_days,
        }
    )


@app.post("/export")
def export_csv():
    settings = load_settings()
    maybe_export_db(settings, str(DB_PATH))
    return jsonify({"status": "success", "message": "CSV export completed"})


if __name__ == "__main__":
    host = os.getenv("BNF_VIEWER_HOST", "0.0.0.0")
    port = int(os.getenv("BNF_VIEWER_PORT", "5001"))
    app.run(host=host, port=port)
