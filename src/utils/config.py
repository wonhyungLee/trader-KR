import os
import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml


_env_pattern = re.compile(r"\${([^}]+)}")
_kis_key_pattern = re.compile(r"^KIS(\d+)_(KEY|SECRET|ACCOUNT_NUMBER|ACCOUNT_CODE)\s*=\s*(.*)$")
_kis_desc_keywords = ("계좌", "한투", "ISA", "연금")
_kis_toggle_path = os.path.join("data", "kis_key_toggles.json")


def _mask_account_no(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= 4:
        return "*" * len(text)
    return ("*" * (len(text) - 4)) + text[-4:]


def _parse_personal_kis_records(path: str = "개인정보") -> Dict[int, Dict[str, str]]:
    if not os.path.exists(path):
        return {}

    records: Dict[int, Dict[str, str]] = {}
    last_idx: Optional[int] = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    comment = line.lstrip("#").strip()
                    if last_idx is not None:
                        rec = records.get(last_idx, {})
                        if comment and not rec.get("description"):
                            if any(k in comment for k in _kis_desc_keywords):
                                rec["description"] = comment
                                records[last_idx] = rec
                    continue

                m = _kis_key_pattern.match(line)
                if not m:
                    last_idx = None
                    continue
                idx = int(m.group(1))
                field = m.group(2)
                value = m.group(3).strip().strip('"').strip("'")
                rec = records.setdefault(idx, {})
                if field == "KEY":
                    rec["app_key"] = value
                elif field == "SECRET":
                    rec["app_secret"] = value
                elif field == "ACCOUNT_NUMBER":
                    rec["account_no"] = value
                elif field == "ACCOUNT_CODE":
                    rec["account_product"] = value
                records[idx] = rec
                last_idx = idx
    except Exception:
        return records
    return records


def _load_kis_toggle_state(path: str = _kis_toggle_path) -> Dict[str, bool]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "keys" in data and isinstance(data["keys"], dict):
                return {str(k): bool(v) for k, v in data["keys"].items()}
            return {str(k): bool(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


def _save_kis_toggle_state(keys: Dict[str, bool], path: str = _kis_toggle_path) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "updated_at": datetime.utcnow().isoformat(),
        "keys": {str(k): bool(v) for k, v in keys.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def has_kis_toggle_file(path: str = _kis_toggle_path) -> bool:
    return os.path.exists(path)


def has_personal_kis_records(path: str = "개인정보") -> bool:
    return bool(_parse_personal_kis_records(path))


def list_kis_key_inventory(max_index: int = 8) -> List[Dict[str, Any]]:
    records = _parse_personal_kis_records()
    toggles = _load_kis_toggle_state()
    inventory: List[Dict[str, Any]] = []
    for idx in range(1, max_index + 1):
        rec = records.get(idx)
        if not rec:
            continue
        inventory.append({
            "id": idx,
            "label": f"KIS{idx}",
            "description": rec.get("description", ""),
            "account_no_masked": _mask_account_no(rec.get("account_no")),
            "account_code": rec.get("account_product"),
            "enabled": bool(toggles.get(str(idx), True)),
        })
    return inventory


def set_kis_key_enabled(idx: int, enabled: bool) -> List[Dict[str, Any]]:
    toggles = _load_kis_toggle_state()
    toggles[str(idx)] = bool(enabled)
    _save_kis_toggle_state(toggles)
    return list_kis_key_inventory()


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no extra dependency).

    Existing environment variables are preserved; .env only fills missing keys.
    Lines starting with `#` are ignored and values may be quoted.
    """
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # keep already-exported values
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # .env loading failures should not break the app
        return


def _load_personal_env(path: str = "개인정보") -> None:
    """Load keys from the local 개인정보 file if present.

    The file is cloud-config style but contains KEY=\"...\" pairs; this loader
    fills missing env vars only and never prints secrets.
    """
    if not os.path.exists(path):
        return

    kv: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    kv[key] = value
                    if key not in os.environ:
                        os.environ[key] = value
    except Exception:
        return

    # Optional override: select a specific KIS key index (1..50).
    # This is useful to rotate appkeys without editing code.
    key_index = os.environ.get("KIS_KEY_INDEX")
    if key_index:
        try:
            idx = int(key_index)
        except Exception:
            idx = 0
        if 1 <= idx <= 50:
            key = kv.get(f"KIS{idx}_KEY")
            sec = kv.get(f"KIS{idx}_SECRET")
            num = kv.get(f"KIS{idx}_ACCOUNT_NUMBER")
            code = kv.get(f"KIS{idx}_ACCOUNT_CODE")
            if key:
                os.environ["KIS_APP_KEY"] = key
            if sec:
                os.environ["KIS_APP_SECRET"] = sec
            if num:
                os.environ["KIS_ACCOUNT_NO"] = num
            if code:
                os.environ["KIS_ACNT_PRDT_CD"] = code

    if "KIS_APP_KEY" not in os.environ:
        for i in range(1, 51):
            key = kv.get(f"KIS{i}_KEY")
            if key:
                os.environ["KIS_APP_KEY"] = key
                break

    if "KIS_APP_SECRET" not in os.environ:
        for i in range(1, 51):
            sec = kv.get(f"KIS{i}_SECRET")
            if sec:
                os.environ["KIS_APP_SECRET"] = sec
                break

    if "KIS_ACCOUNT_NO" not in os.environ:
        for i in range(1, 51):
            num = kv.get(f"KIS{i}_ACCOUNT_NUMBER")
            code = kv.get(f"KIS{i}_ACCOUNT_CODE")
            if num and code:
                os.environ["KIS_ACCOUNT_NO"] = num
                if "KIS_ACNT_PRDT_CD" not in os.environ:
                    os.environ["KIS_ACNT_PRDT_CD"] = code
                break


def _sub_env(value: str) -> str:
    """Replace ${VAR} with environment variable if present."""
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        return os.environ.get(key, match.group(0))

    return _env_pattern.sub(repl, value)


def load_kis_keys() -> List[Dict[str, str]]:
    """Extract enabled KIS{n} key sets from the 개인정보 file."""
    records = _parse_personal_kis_records()
    if not records:
        return []
    toggles = _load_kis_toggle_state()
    keys: List[Dict[str, str]] = []
    for i in sorted(records.keys()):
        rec = records[i]
        if toggles and toggles.get(str(i), True) is False:
            continue
        app_key = rec.get("app_key")
        app_secret = rec.get("app_secret")
        if app_key and app_secret:
            keys.append({
                "app_key": app_key,
                "app_secret": app_secret,
                "account_no": rec.get("account_no"),
                "account_product": rec.get("account_product") or "01",
            })
    return keys


def load_yaml(path: str) -> Dict[str, Any]:
    # Populate os.environ from .env and 개인정보 before substitution
    _load_dotenv()
    _load_personal_env()
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # env substitution for string values
    substituted = _env_pattern.sub(lambda m: os.environ.get(m.group(1), m.group(0)), raw)
    data = yaml.safe_load(substituted) or {}
    return data


def load_settings(path: str = "config/settings.yaml") -> Dict[str, Any]:
    return load_yaml(path)
