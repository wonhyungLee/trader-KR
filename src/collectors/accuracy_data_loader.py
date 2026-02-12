from __future__ import annotations

import argparse
import csv
import logging
import time
from datetime import datetime, timedelta
import os
from pathlib import Path
import json
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.brokers.kis_broker import KISBroker
from src.storage.sqlite_store import SQLiteStore
from src.utils.config import load_settings
from src.utils.db_exporter import maybe_export_db
from src.utils.notifier import maybe_notify


def _to_float(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return 0.0


def _to_int(value) -> int:
    if value is None:
        return 0
    try:
        return int(float(str(value).replace(",", "")))
    except Exception:
        return 0


def _ymd(d: datetime) -> str:
    return d.strftime("%Y%m%d")


def _normalize_date(value: str) -> Optional[str]:
    if not value:
        return None
    s = str(value).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s
    return None


def _normalize_ymd(value: str) -> Optional[str]:
    if not value:
        return None
    s = str(value).strip().replace("-", "")
    if len(s) == 8 and s.isdigit():
        return s
    return None


def _clamp_ymd(value: str, max_ymd: str) -> str:
    v = _normalize_ymd(value) or value
    if not v:
        return max_ymd
    return min(v, max_ymd)


def _load_progress(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_progress(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _date_in_range(date_str: str, start_ymd: str, end_ymd: str) -> bool:
    if not date_str:
        return False
    s = date_str.replace("-", "")
    return start_ymd <= s <= end_ymd


def _clean_params(params: Dict) -> Dict:
    return {k: v for k, v in params.items() if v is not None and v != ""}


def _request_with_retry(
    broker: KISBroker,
    tr_id: str,
    url: str,
    params: Dict,
    max_retries: Optional[int] = None,
) -> Dict:
    params = _clean_params(params or {})
    return broker.request(tr_id, url, params=params, max_retries=max_retries)


def _append_failed_code(path: Path, code: str, error: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["ts", "code", "error"])
        writer.writerow([datetime.utcnow().isoformat(), code, error[:300]])


class AuthForbiddenError(Exception):
    pass


def _is_auth_forbidden_error(exc: Exception) -> bool:
    msg = str(exc)
    return "403" in msg and "tokenP" in msg


def _safe_fetch(label: str, func):
    try:
        return func(), None
    except Exception as exc:
        if _is_auth_forbidden_error(exc):
            raise AuthForbiddenError(str(exc))
        logging.warning("fetch failed %s: %s", label, exc)
        return [], exc


def load_codes(store: SQLiteStore) -> List[str]:
    codes = store.list_universe_codes()
    if codes:
        return codes
    # fallback to universe CSVs if DB is empty
    paths = ["data/universe_kospi100.csv", "data/universe_kosdaq150.csv"]
    out: List[str] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        col = "code" if "code" in df.columns else df.columns[0]
        out.extend(df[col].astype(str).str.zfill(6).tolist())
    # unique preserve order
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def read_codes_from_paths(paths: Iterable[str]) -> List[str]:
    codes: List[str] = []
    for p in paths:
        if not p:
            continue
        if not Path(p).exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        col = "code" if "code" in df.columns else "Code" if "Code" in df.columns else df.columns[0]
        codes.extend(df[col].astype(str).str.zfill(6).tolist())
    seen = set()
    uniq = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def load_market_map(store: SQLiteStore) -> Dict[str, str]:
    cur = store.conn.execute("SELECT code, market FROM universe_members")
    return {row[0]: (row[1] or "") for row in cur.fetchall()}


def load_last_price_dates(store: SQLiteStore) -> Dict[str, str]:
    cur = store.conn.execute("SELECT code, MAX(date) FROM daily_price GROUP BY code")
    out: Dict[str, str] = {}
    for code, date in cur.fetchall():
        if date:
            out[str(code).zfill(6)] = str(date).replace("-", "")
    return out


def load_global_last_date(store: SQLiteStore) -> Optional[str]:
    cur = store.conn.execute("SELECT MAX(date) FROM daily_price")
    row = cur.fetchone()
    if row and row[0]:
        return str(row[0]).replace("-", "")
    return None


def market_div_code(market: str) -> str:
    if "KOSPI" in market:
        return "1"
    if "KOSDAQ" in market:
        return "2"
    return "3"


def fetch_investor_flow(broker: KISBroker, code: str, end_ymd: str) -> List[Dict[str, object]]:
    url = f"{broker.base_url}/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
    tr_id = "FHPTJ04160001"
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": end_ymd,
        "FID_ORG_ADJ_PRC": "",
        "FID_ETC_CLS_CODE": "",
    }
    res = _request_with_retry(broker, tr_id, url, params)
    output2 = res.get("output2") or res.get("output") or []
    if isinstance(output2, dict):
        output2 = [output2]
    rows = []
    for rec in output2:
        date_raw = rec.get("stck_bsop_date") or rec.get("bsop_date")
        rows.append(
            {
                "date": _normalize_date(date_raw),
                "code": code,
                "foreign_net_value": _to_float(rec.get("frgn_ntby_tr_pbmn") or rec.get("frgn_ntby_qty")),
                "inst_net_value": _to_float(rec.get("orgn_ntby_tr_pbmn") or rec.get("orgn_ntby_qty")),
                "indiv_net_value": _to_float(rec.get("prsn_ntby_tr_pbmn") or rec.get("prsn_ntby_qty")),
            }
        )
    return rows


def fetch_program_trade(broker: KISBroker, code: str, end_ymd: str) -> List[Dict[str, object]]:
    url = f"{broker.base_url}/uapi/domestic-stock/v1/quotations/program-trade-by-stock-daily"
    tr_id = "FHPPG04650201"
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": end_ymd,
    }
    res = _request_with_retry(broker, tr_id, url, params)
    output = res.get("output") or []
    if isinstance(output, dict):
        output = [output]
    rows = []
    for rec in output:
        date_raw = rec.get("stck_bsop_date") or rec.get("bsop_date")
        rows.append(
            {
                "date": _normalize_date(date_raw),
                "code": code,
                "program_net_value": _to_float(rec.get("whol_smtn_ntby_tr_pbmn") or rec.get("whol_smtn_ntby_qty")),
            }
        )
    return rows


def fetch_short_sale(broker: KISBroker, code: str, start_ymd: str, end_ymd: str) -> List[Dict[str, object]]:
    url = f"{broker.base_url}/uapi/domestic-stock/v1/quotations/daily-short-sale"
    tr_id = "FHPST04830000"
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": start_ymd,
        "FID_INPUT_DATE_2": end_ymd,
    }
    res = _request_with_retry(broker, tr_id, url, params)
    output2 = res.get("output2") or []
    if isinstance(output2, dict):
        output2 = [output2]
    rows = []
    for rec in output2:
        date_raw = rec.get("stck_bsop_date") or rec.get("bsop_date")
        rows.append(
            {
                "date": _normalize_date(date_raw),
                "code": code,
                "short_volume": _to_float(rec.get("ssts_cntg_qty")),
                "short_value": _to_float(rec.get("ssts_tr_pbmn")),
                "short_ratio": _to_float(rec.get("ssts_vol_rlim") or rec.get("ssts_tr_pbmn_rlim")),
            }
        )
    return rows


def fetch_credit_balance(broker: KISBroker, code: str, end_ymd: str) -> List[Dict[str, object]]:
    url = f"{broker.base_url}/uapi/domestic-stock/v1/quotations/daily-credit-balance"
    tr_id = "FHPST04760000"
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_COND_SCR_DIV_CODE": "20476",
        "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": end_ymd,
    }
    res = _request_with_retry(broker, tr_id, url, params)
    output = res.get("output") or []
    if isinstance(output, dict):
        output = [output]
    rows = []
    for rec in output:
        date_raw = rec.get("stlm_date") or rec.get("deal_date") or rec.get("stck_bsop_date")
        rows.append(
            {
                "date": _normalize_date(date_raw),
                "code": code,
                "credit_qty": _to_float(rec.get("whol_loan_rmnd_stcn")),
                "credit_value": _to_float(rec.get("whol_loan_rmnd_amt")),
            }
        )
    return rows


def fetch_loan_trans(broker: KISBroker, code: str, start_ymd: str, end_ymd: str, mrkt_div: str) -> List[Dict[str, object]]:
    url = f"{broker.base_url}/uapi/domestic-stock/v1/quotations/daily-loan-trans"
    tr_id = "HHPST074500C0"
    params = {
        "MRKT_DIV_CLS_CODE": mrkt_div,
        "MKSC_SHRN_ISCD": code,
        "START_DATE": start_ymd,
        "END_DATE": end_ymd,
        "CTS": "",
    }
    res = _request_with_retry(broker, tr_id, url, params)
    output = res.get("output1") or res.get("output") or []
    if isinstance(output, dict):
        output = [output]
    rows = []
    for rec in output:
        date_raw = rec.get("bsop_date") or rec.get("stck_bsop_date")
        rows.append(
            {
                "date": _normalize_date(date_raw),
                "code": code,
                "loan_qty": _to_float(rec.get("rmnd_stcn")),
                "loan_value": _to_float(rec.get("rmnd_amt")),
            }
        )
    return rows


def fetch_vi_status(broker: KISBroker, code: str, end_ymd: str) -> List[Dict[str, object]]:
    url = f"{broker.base_url}/uapi/domestic-stock/v1/quotations/inquire-vi-status"
    tr_id = "FHPST01390000"
    params = {
        "FID_DIV_CLS_CODE": "0",
        "FID_COND_SCR_DIV_CODE": "20139",
        "FID_MRKT_CLS_CODE": "0",
        "FID_INPUT_ISCD": code,
        "FID_RANK_SORT_CLS_CODE": "0",
        "FID_INPUT_DATE_1": end_ymd,
        "FID_TRGT_CLS_CODE": "",
        "FID_TRGT_EXLS_CLS_CODE": "",
    }
    res = _request_with_retry(broker, tr_id, url, params)
    output = res.get("output") or []
    if isinstance(output, dict):
        output = [output]
    count = len(output) if isinstance(output, list) else 0
    return [{"date": _normalize_date(end_ymd), "code": code, "vi_count": count}]


def filter_rows(rows: Iterable[Dict[str, object]], start_ymd: str, end_ymd: str) -> List[Dict[str, object]]:
    out = []
    for r in rows:
        date = r.get("date")
        if not date:
            continue
        if _date_in_range(date, start_ymd, end_ymd):
            out.append(r)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3, help="최근 n일 데이터 수집")
    parser.add_argument("--start-date", type=str, default=None, help="시작일자(YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="종료일자(YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("--from-2024", action="store_true", help="2024-01-01부터 수집")
    parser.add_argument("--start-index", type=int, default=0, help="이미 처리된 종목 수(0-based skip count)")
    parser.add_argument("--resume", action="store_true", help="progress 파일 기준으로 이어서 수행")
    parser.add_argument("--notify-every", type=int, default=1, help="n개 종목마다 진행 알림")
    parser.add_argument("--sleep", type=float, default=None, help="종목 처리 간 슬립(초)")
    parser.add_argument("--rate-sleep", type=float, default=None, help="요청 간 슬립(초). 설정 시 KIS rate_limit_sleep_sec를 덮어씀")
    parser.add_argument("--limit", type=int, default=None, help="처리할 종목 수 제한(테스트)")
    parser.add_argument("--codes-file", action="append", default=[], help="CSV 경로(코드 컬럼) 지정 시 해당 코드만 처리")
    parser.add_argument("--codes", type=str, default=None, help="쉼표로 구분된 코드 리스트(예: 005930,000660)")
    parser.add_argument("--progress-file", type=str, default="data/accuracy_progress.json", help="progress 파일 경로")
    parser.add_argument("--lock-file", type=str, default="data/accuracy_loader.lock", help="lock 파일 경로")
    args = parser.parse_args()

    settings = load_settings()
    store = SQLiteStore(settings.get("database", {}).get("path", "data/market_data.db"))
    broker = KISBroker(settings)
    if args.rate_sleep is not None:
        broker.rate_limit_sleep = float(args.rate_sleep)
    else:
        broker.rate_limit_sleep = float(
            settings.get("kis", {}).get("accuracy_rate_limit_sleep_sec", broker.rate_limit_sleep)
        )
    all_codes = load_codes(store)
    override_codes: List[str] = []
    if args.codes_file:
        override_codes.extend(read_codes_from_paths(args.codes_file))
    if args.codes:
        override_codes.extend([c.strip().zfill(6) for c in args.codes.split(",") if c.strip()])
    if override_codes:
        seen = set()
        uniq = []
        for c in override_codes:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        all_codes = uniq

    progress_path = Path(args.progress_file)
    if args.resume:
        prog = _load_progress(progress_path) or {}
        last_index = int(prog.get("last_index", -1))
        if last_index >= 0:
            args.start_index = max(args.start_index, last_index + 1)

    total_global = len(all_codes)
    if args.start_index < 0:
        args.start_index = 0
    if args.start_index > total_global:
        args.start_index = total_global
    codes = all_codes[args.start_index:]
    if args.limit:
        codes = codes[: args.limit]
    market_map = load_market_map(store)
    last_date_map = load_last_price_dates(store)
    global_last = load_global_last_date(store)

    end_dt = datetime.today()
    today_ymd = _ymd(end_dt)
    end_ymd = today_ymd
    if args.from_2024:
        start_ymd = "20240101"
    elif args.start_date:
        start_ymd = _normalize_ymd(args.start_date) or _ymd(end_dt - timedelta(days=max(1, args.days) - 1))
    else:
        start_ymd = _ymd(end_dt - timedelta(days=max(1, args.days) - 1))
    if args.end_date:
        end_ymd = _normalize_ymd(args.end_date) or end_ymd
    if end_ymd > today_ymd:
        end_ymd = today_ymd
    if global_last and global_last < end_ymd:
        end_ymd = global_last
    if start_ymd > end_ymd:
        start_ymd = end_ymd

    lock_path = Path(args.lock_file)
    if lock_path.exists():
        try:
            existing_pid = int(lock_path.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            existing_pid = 0
        if existing_pid > 0:
            try:
                os.kill(existing_pid, 0)
                maybe_notify(settings, f"[accuracy] already running pid={existing_pid}")
                return
            except Exception:
                # stale lock
                pass
        try:
            lock_path.unlink()
        except Exception:
            pass
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(str(os.getpid()), encoding="utf-8")

    total = len(codes)
    done = 0
    errors = 0
    failed_codes: List[str] = []
    failed_path = Path("data/csv/failed_codes_accuracy.csv")
    maybe_notify(
        settings,
        f"[accuracy] start {args.start_index}/{total_global} range={start_ymd}-{end_ymd} rate_sleep={broker.rate_limit_sleep}",
    )

    item_sleep = args.sleep
    if item_sleep is None:
        item_sleep = float(settings.get("kis", {}).get("accuracy_item_sleep_sec", 0.1))

    auth_cooldown = float(settings.get("kis", {}).get("auth_forbidden_cooldown_sec", 600))

    try:
        idx = 0
        while idx < len(codes):
            code = codes[idx]
            done += 1
            done_global = args.start_index + done
            try:
                code_end = _clamp_ymd(last_date_map.get(code, end_ymd), end_ymd)
                code_start = start_ymd if start_ymd <= code_end else code_end
                inv_rows, inv_err = _safe_fetch(
                    "investor_flow",
                    lambda: filter_rows(fetch_investor_flow(broker, code, code_end), code_start, code_end),
                )
                prog_rows, prog_err = _safe_fetch(
                    "program_trade",
                    lambda: filter_rows(fetch_program_trade(broker, code, code_end), code_start, code_end),
                )
                short_rows, short_err = _safe_fetch(
                    "short_sale",
                    lambda: filter_rows(fetch_short_sale(broker, code, code_start, code_end), code_start, code_end),
                )
                cred_rows, cred_err = _safe_fetch(
                    "credit_balance",
                    lambda: filter_rows(fetch_credit_balance(broker, code, code_end), code_start, code_end),
                )
                mrkt_div = market_div_code(market_map.get(code, ""))
                loan_rows, loan_err = _safe_fetch(
                    "loan_trans",
                    lambda: filter_rows(
                        fetch_loan_trans(broker, code, code_start, code_end, mrkt_div), code_start, code_end
                    ),
                )
                vi_rows, vi_err = _safe_fetch("vi_status", lambda: fetch_vi_status(broker, code, code_end))

                if inv_rows:
                    store.upsert_investor_flow(inv_rows)
                if prog_rows:
                    store.upsert_program_trade(prog_rows)
                if short_rows:
                    store.upsert_short_sale(short_rows)
                if cred_rows:
                    store.upsert_credit_balance(cred_rows)
                if loan_rows:
                    store.upsert_loan_trans(loan_rows)
                if vi_rows:
                    store.upsert_vi_status(vi_rows)

                had_error = any([inv_err, prog_err, short_err, cred_err, loan_err, vi_err])
                if had_error:
                    errors += 1
                    failed_codes.append(code)
                    error_summary = "; ".join(
                        [
                            f"inv={inv_err}",
                            f"prog={prog_err}",
                            f"short={short_err}",
                            f"credit={cred_err}",
                            f"loan={loan_err}",
                            f"vi={vi_err}",
                        ]
                    )
                    _append_failed_code(failed_path, code, error_summary)

                if args.notify_every > 0 and (done % args.notify_every == 0 or done == total):
                    msg = (
                        f"[accuracy] {done_global}/{total_global} {code} "
                        f"inv={len(inv_rows)} prog={len(prog_rows)} short={len(short_rows)} "
                        f"credit={len(cred_rows)} loan={len(loan_rows)} vi={len(vi_rows)} err={1 if had_error else 0}"
                    )
                    maybe_notify(settings, msg)
            except AuthForbiddenError as exc:
                # Pause and reset cache, then retry the same code
                logging.warning("auth forbidden detected. cooling down %.1fs", auth_cooldown)
                if auth_cooldown > 0:
                    time.sleep(auth_cooldown)
                try:
                    broker.clear_token_cache()
                    broker.reset_sessions()
                except Exception:
                    pass
                done -= 1
                continue
            except Exception as exc:
                errors += 1
                failed_codes.append(code)
                logging.exception("accuracy loader failed for %s", code)
                if args.notify_every > 0:
                    maybe_notify(settings, f"[accuracy] ERROR {done_global}/{total_global} {code} {exc}")
            _save_progress(
                progress_path,
                {
                    "last_index": done_global - 1,
                    "last_code": code,
                    "total": total_global,
                    "updated_at": datetime.utcnow().isoformat(),
                },
            )
            if item_sleep and item_sleep > 0:
                time.sleep(item_sleep)
            idx += 1
    finally:
        try:
            lock_path.unlink()
        except Exception:
            pass

    maybe_export_db(settings, store.db_path)
    if failed_codes:
        maybe_notify(settings, f"[accuracy] done {args.start_index + done}/{total_global} errors={errors} failed={len(failed_codes)}")
    else:
        maybe_notify(settings, f"[accuracy] done {args.start_index + done}/{total_global} errors={errors}")


if __name__ == "__main__":
    main()
