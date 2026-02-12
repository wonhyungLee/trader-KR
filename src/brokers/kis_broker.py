from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

import requests
from requests import HTTPError, RequestException

from src.utils.config import load_settings, load_kis_keys, has_kis_toggle_file, has_personal_kis_records
from src.utils.http_retry import is_retryable_status, sleep_backoff
from src.utils.rate_limiter import RateLimiter


TOKEN_CACHE_DEFAULT = ".cache/kis_token.json"


class KISKeySession:
    """Manages token and session for a single KIS key set."""
    def __init__(self, key_config: Dict[str, str], base_url: str, token_cache_path: str, use_hashkey: bool = False, hashkey_cache_ttl: float = 30.0):
        self.app_key = key_config["app_key"]
        self.app_secret = key_config["app_secret"]
        self.account_no = key_config.get("account_no")
        self.account_product = key_config.get("account_product", "01")
        self.base_url = base_url
        self.token_cache_path = token_cache_path.replace(".json", f"_{self.app_key[:8]}.json")
        self.session = requests.Session()
        self._token: Optional[str] = None
        self._token_expire: Optional[datetime] = None
        
        # Hashkey support
        self.use_hashkey = use_hashkey
        self.hashkey_cache_ttl = hashkey_cache_ttl
        self._hashkey_cache: Dict[str, Any] = {"key": None, "value": None, "ts": 0.0}

    def get_hashkey(self, body: Dict[str, Any]) -> Optional[str]:
        if not self.use_hashkey:
            return None
        try:
            body_key = json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        except Exception:
            body_key = str(body)

        now = time.time()
        ck = self._hashkey_cache
        if ck.get("key") == body_key and (now - float(ck.get("ts", 0.0))) <= self.hashkey_cache_ttl:
            return ck.get("value")

        url = f"{self.base_url}/uapi/hashkey"
        headers = {
            "content-type": "application/json; charset=utf-8",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        try:
            resp = self.session.post(url, headers=headers, json=body, timeout=10)
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
            hashv = data.get("HASH") or data.get("hash") or (data.get("output") or {}).get("HASH")
            if hashv:
                self._hashkey_cache = {"key": body_key, "value": hashv, "ts": now}
            return hashv
        except Exception as e:
            logging.warning("hashkey fetch failed for %s: %s", self.app_key[:8], e)
            return None

    def _load_token_cache(self):
        if not os.path.exists(self.token_cache_path):
            return
        try:
            with open(self.token_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._token = data.get("access_token")
            exp = data.get("expires_at")
            if exp:
                self._token_expire = datetime.fromisoformat(exp)
        except Exception:
            return

    def _save_token_cache(self, token: str, expires_at: datetime):
        self._token = token
        self._token_expire = expires_at
        payload = {"access_token": token, "expires_at": expires_at.isoformat()}
        os.makedirs(os.path.dirname(self.token_cache_path), exist_ok=True)
        with open(self.token_cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def ensure_token(self) -> str:
        if not self._token:
            self._load_token_cache()
        now = datetime.now(timezone.utc)
        if self._token and self._token_expire:
            if self._token_expire.tzinfo is None:
                self._token_expire = self._token_expire.replace(tzinfo=timezone.utc)
            if self._token_expire > now + timedelta(minutes=5):
                return self._token
        token, exp = self.issue_token()
        self._save_token_cache(token, exp)
        return token

    def issue_token(self):
        url = f"{self.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        resp = self.session.post(url, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token") or data.get("access_token_token") or data.get("approval_key")
        exp_sec = int(data.get("expires_in", 3600))
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=exp_sec)
        return token, expires_at

    def issue_ws_approval(self) -> str:
        url = f"{self.base_url}/oauth2/Approval"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "secretkey": self.app_secret,
        }
        resp = self.session.post(url, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        key = data.get("approval_key")
        if not key:
            raise RuntimeError(f"ws approval_key missing: {data}")
        return key


class KISBroker:
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = settings or load_settings()
        self.env = self.settings.get("kis", {}).get("env", self.settings.get("env", "paper"))
        if self.env != "prod":
            logging.warning("KIS env=paper로 설정되어 있습니다. (조회/수집만 사용)");
            # viewer 버전은 조회/수집만 수행하며, 주문/잔고 기능이 제거되었습니다.
        self.custtype = self.settings["kis"].get("custtype", "P")
        self.rate_limit_sleep = float(self.settings["kis"].get("rate_limit_sleep_sec", 0.5))
        self.timeout_connect = float(self.settings["kis"].get("timeout_connect_sec", 5))
        self.timeout_read = float(self.settings["kis"].get("timeout_read_sec", 20))
        self.max_retries = int(self.settings["kis"].get("max_retries", 8))
        self.backoff_base = float(self.settings["kis"].get("backoff_base_sec", 2))
        self.backoff_cap = float(self.settings["kis"].get("backoff_cap_sec", 60))
        self.backoff_jitter = float(self.settings["kis"].get("backoff_jitter_sec", 0.5))
        self.consecutive_error_cooldown_after = int(
            self.settings["kis"].get("consecutive_error_cooldown_after", 10)
        )
        self.consecutive_error_cooldown_sec = float(
            self.settings["kis"].get("consecutive_error_cooldown_sec", 180)
        )
        self.auth_forbidden_cooldown_sec = float(
            self.settings["kis"].get("auth_forbidden_cooldown_sec", 600)
        )
        self.session_reset_every = int(self.settings["kis"].get("session_reset_every", 3))
        self.base_url = self.settings["kis"].get(
            "base_url_prod" if self.env == "prod" else "base_url_paper",
            "https://openapivts.koreainvestment.com:29443",
        )
        self.ws_url = self.settings["kis"].get(
            "ws_url_prod" if self.env == "prod" else "ws_url_paper",
            "ws://ops.koreainvestment.com:21000" if self.env == "prod" else "ws://ops.koreainvestment.com:31000",
        )
        self.token_cache_path = self.settings["kis"].get("token_cache_path", TOKEN_CACHE_DEFAULT)
        
        # Load hashkey settings
        self.use_hashkey = bool(self.settings["kis"].get("use_hashkey", False))
        self.hashkey_cache_ttl = float(self.settings["kis"].get("hashkey_cache_ttl_sec", 30))

        # Load multiple keys for rotation (enabled only)
        key_configs = load_kis_keys()
        if not key_configs:
            if has_kis_toggle_file() and has_personal_kis_records():
                raise RuntimeError("No enabled KIS keys. Enable at least one in the dashboard.")
            # Fallback to single key from settings/env if no keys found in 개인정보
            key_configs = [{
                "app_key": self.settings["kis"].get("app_key"),
                "app_secret": self.settings["kis"].get("app_secret"),
                "account_no": self.settings["kis"].get("account_no"),
                "account_product": self.settings["kis"].get("acnt_prdt_cd", "01")
            }]
        
        self.sessions = [KISKeySession(cfg, self.base_url, self.token_cache_path, self.use_hashkey, self.hashkey_cache_ttl) for cfg in key_configs]
        self._current_session_idx = 0
        self._consecutive_errors = 0
        self._auth_forbidden_last_ts = 0.0

        # Rate Limiter setup (Total capacity = num_keys * single_key_limit)
        safe_tps = 1.0 / max(0.01, self.rate_limit_sleep)
        total_tps = safe_tps * len(self.sessions)
        self.rate_limiter = RateLimiter(
            max_tokens=max(10.0, total_tps * 2),
            refill_rate=total_tps,
            trading_reserve=max(5.0, total_tps * 0.2) 
        )

    @property
    def current_session(self) -> KISKeySession:
        return self.sessions[self._current_session_idx]

    def rotate_session(self):
        self._current_session_idx = (self._current_session_idx + 1) % len(self.sessions)

    def reset_sessions(self):
        for s in self.sessions:
            try:
                s.session.close()
            except Exception:
                pass
            s.session = requests.Session()
        self._current_session_idx = 0

    def clear_token_cache(self):
        for s in self.sessions:
            s._token = None
            s._token_expire = None
            s._hashkey_cache = {"key": None, "value": None, "ts": 0.0}

        try:
            base = Path(self.token_cache_path)
            if base.parent.exists():
                pattern = f"{base.stem}*.json"
                for p in base.parent.glob(pattern):
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception as exc:
            logging.warning("Failed to clear token cache files: %s", exc)

        try:
            state_path = Path(self.rate_limiter.state_file)
            if state_path.exists():
                state_path.unlink()
        except Exception:
            pass

    def _cooldown_on_auth_forbidden(self, reason: str):
        cooldown = self.auth_forbidden_cooldown_sec
        if cooldown <= 0:
            return
        now = time.time()
        sleep_sec = cooldown
        if self._auth_forbidden_last_ts > 0:
            elapsed = now - self._auth_forbidden_last_ts
            if elapsed < cooldown:
                sleep_sec = cooldown - elapsed
        self._auth_forbidden_last_ts = time.time()
        logging.warning("KIS 403 (%s). Cooling down %.1fs and clearing token cache.", reason, sleep_sec)
        time.sleep(sleep_sec)
        self.clear_token_cache()
        self.reset_sessions()

    # Proxy properties for backward compatibility
    @property
    def app_key(self): return self.current_session.app_key
    @property
    def app_secret(self): return self.current_session.app_secret
    @property
    def account_no(self): return self.current_session.account_no
    @property
    def account_product(self): return self.current_session.account_product

    # ---------------- Token -----------------
    def ensure_token(self) -> str:
        return self.current_session.ensure_token()

    def issue_token(self):
        return self.current_session.issue_token()

    def issue_ws_approval(self) -> str:
        return self.current_session.issue_ws_approval()

    # --------------- Base request ---------------
    def request(
        self,
        tr_id: str,
        url: str,
        method: str = "GET",
        params=None,
        data=None,
        json_body=None,
        max_retries: Optional[int] = None,
        priority: str = "LOW",
    ) -> Dict[str, Any]:
        method = method.upper()
        last_exc: Optional[Exception] = None
        retries = self.max_retries if max_retries is None else max(1, int(max_retries))

        for attempt in range(1, max(1, retries) + 1):
            # Rotate session on each attempt or periodically to distribute load
            if attempt > 1:
                self.rotate_session()

            sess = self.current_session
            
            if attempt > 1 and self.session_reset_every > 0 and (attempt - 1) % self.session_reset_every == 0:
                sess.session = requests.Session()

            # Wait for rate limit token
            self.rate_limiter.wait(priority=priority)

            try:
                try:
                    token = sess.ensure_token()
                except HTTPError as exc:
                    status = exc.response.status_code if getattr(exc, "response", None) else None
                    if status == 403:
                        self._cooldown_on_auth_forbidden("token")
                        token = sess.ensure_token()
                    else:
                        raise

                headers = {
                    "content-type": "application/json; charset=utf-8",
                    "authorization": f"Bearer {token}",
                    "appkey": sess.app_key,
                    "appsecret": sess.app_secret,
                    "tr_id": tr_id,
                }
                if self.custtype:
                    headers["custtype"] = self.custtype

                # Optional hashkey header for POST requests
                if method != "GET" and json_body is not None:
                    hk = sess.get_hashkey(json_body)
                    if hk:
                        headers["hashkey"] = hk

                timeout = (self.timeout_connect, self.timeout_read)
                if method == "GET":
                    resp = sess.session.get(url, headers=headers, params=params, timeout=timeout)
                else:
                    resp = sess.session.post(url, headers=headers, params=params, data=data, json=json_body, timeout=timeout)

                # KIS quirk: sometimes returns 500 for expired token
                is_token_expired = False
                if resp.status_code == 500:
                    try:
                        err_data = resp.json()
                        if err_data.get("msg_cd") == "EGW00123": # 기간이 만료된 token 입니다.
                            is_token_expired = True
                    except Exception:
                        pass

                if (resp.status_code in (401, 403) or is_token_expired):
                    if resp.status_code == 403:
                        self._cooldown_on_auth_forbidden("api")
                    logging.info("KIS token expired (%s) for key %s, refreshing...", resp.status_code, sess.app_key[:8])
                    token, exp = sess.issue_token()
                    sess._save_token_cache(token, exp)
                    # Retry with the same key but refreshed token
                    continue

                if is_retryable_status(resp.status_code):
                    logging.error("KIS API Error %s: %s", resp.status_code, resp.text)
                    raise HTTPError(f"{resp.status_code} retryable", response=resp)

                resp.raise_for_status()
                self._consecutive_errors = 0
                try:
                    return resp.json()
                except Exception:
                    return {"text": resp.text}
            except HTTPError as exc:
                status = exc.response.status_code if getattr(exc, "response", None) else None
                if status == 403:
                    self._cooldown_on_auth_forbidden("api")
                    last_exc = exc
                    continue
                if not is_retryable_status(status):
                    raise
                last_exc = exc
            except RequestException as exc:
                last_exc = exc

            self._consecutive_errors += 1
            if (
                self.consecutive_error_cooldown_after > 0
                and self._consecutive_errors >= self.consecutive_error_cooldown_after
            ):
                logging.warning(
                    "KIS consecutive errors=%s, cooldown %.1fs",
                    self._consecutive_errors,
                    self.consecutive_error_cooldown_sec,
                )
                time.sleep(self.consecutive_error_cooldown_sec)
                self._consecutive_errors = 0

            if attempt < retries:
                delay = sleep_backoff(attempt, self.backoff_base, self.backoff_cap, self.backoff_jitter)
                logging.warning("KIS retry %s/%s in %.1fs (%s) using key %s", attempt, retries, delay, tr_id, sess.app_key[:8])

        if last_exc:
            raise last_exc
        raise RuntimeError("request failed")

    # --------------- Quotes ---------------
    def get_current_price(self, code: str) -> Dict[str, Any]:
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        tr_id = "FHKST01010100"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
        }
        return self.request(tr_id, url, params=params)

    def get_multi_price(self, codes: list[str]) -> Dict[str, Any]:
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/intstock-multprice"
        tr_id = "FHKST11300006"
        params: Dict[str, Any] = {}
        for idx, code in enumerate(codes[:30], start=1):
            params[f"FID_COND_MRKT_DIV_CODE_{idx}"] = "J"
            params[f"FID_INPUT_ISCD_{idx}"] = code
        return self.request(tr_id, url, params=params)


if __name__ == "__main__":
    broker = KISBroker()
    print(broker.issue_token())
