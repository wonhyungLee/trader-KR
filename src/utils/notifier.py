import logging
import time

import requests
from typing import Optional


def _append_site(settings: dict, message: str) -> str:
    url = settings.get("site_url") or settings.get("site", {}).get("url")
    if not url or url in message:
        return message
    return f"{message} | site: {url}"


def send_telegram(bot_token: str, chat_id: str, text: str, parse_mode: str = "Markdown") -> bool:
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": parse_mode}, timeout=5)
        if not resp.ok:
            logging.warning("telegram send failed: %s", resp.text)
            return False
        return True
    except Exception as e:
        logging.warning("telegram send error: %s", e)
        return False


def maybe_notify(settings: dict, message: str):
    message = _append_site(settings, message)
    # Discord 시도
    dc = settings.get("discord", {})
    dc_success = False
    if dc and dc.get("enabled") and dc.get("webhook"):
        try:
            resp = requests.post(dc["webhook"], json={"content": message}, timeout=5)
            if resp.status_code == 429:
                # Rate limited - wait and retry once
                retry_after = resp.json().get("retry_after", 1)
                logging.warning("Discord rate limited. Retrying after %s seconds...", retry_after)
                time.sleep(retry_after + 0.1)
                resp = requests.post(dc["webhook"], json={"content": message}, timeout=5)
            
            if resp.ok:
                dc_success = True
            else:
                logging.warning("discord send failed: %s", resp.text)
        except Exception:
            logging.exception("discord send failed")
    
    # Discord 실패하거나 비활성화된 경우 Telegram 시도 (설정된 경우)
    if not dc_success:
        tg = settings.get("telegram", {})
        if tg and tg.get("enabled"):
            send_telegram(tg.get("token"), tg.get("chat_id"), message)
