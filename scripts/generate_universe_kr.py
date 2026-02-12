"""
KOSPI200 / KOSDAQ150 유니버스 CSV 생성 스크립트.

- 가능한 경우 FinanceDataReader(=finance-datareader)로 인덱스/시장 목록을 가져옵니다.
- 환경/버전에 따라 지원되는 심볼이 다를 수 있어 여러 경로로 시도합니다.
- 결과:
  - data/universe_kospi200.csv
  - data/universe_kosdaq150.csv

사용:
  python scripts/generate_universe_kr.py
  python scripts/generate_universe_kr.py --out-dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

try:
    import FinanceDataReader as fdr  # type: ignore
except Exception as exc:
    raise SystemExit(
        "FinanceDataReader가 필요합니다. 먼저 설치하세요:\n"
        "  pip install finance-datareader\n"
        f"(import error: {exc})"
    )


def _norm_krx_code(val: str) -> str:
    return str(val).strip().zfill(6)


def _pick_cols(df: pd.DataFrame) -> tuple[str, str, str]:
    """Try to find code/name/market columns."""
    cols = {c.lower(): c for c in df.columns}
    code = cols.get("code") or cols.get("symbol") or cols.get("종목코드") or df.columns[0]
    name = cols.get("name") or cols.get("company") or cols.get("종목명") or df.columns[1]
    market = cols.get("market") or cols.get("market")
    return code, name, market or ""


def _sort_by_marcap(df: pd.DataFrame) -> pd.DataFrame:
    # common column names in FinanceDataReader listings
    for key in ("Marcap", "MarketCap", "marcap", "marketcap", "시가총액"):
        if key in df.columns:
            return df.sort_values(key, ascending=False)
    # fallback: keep order
    return df


def _try_stocklisting(symbols: list[str]) -> pd.DataFrame | None:
    for sym in symbols:
        try:
            df = fdr.StockListing(sym)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df["__source__"] = sym
                return df
        except Exception:
            continue
    return None


def build_kospi200() -> pd.DataFrame:
    # Try direct index listing first (if supported)
    df = _try_stocklisting(["KOSPI200", "KRX-KOSPI200", "KOSPI 200"])
    if df is not None:
        pass
    else:
        # fallback: take top 200 by marcap from KOSPI listing
        df = _try_stocklisting(["KOSPI", "KRX", "KRX-DESC", "KRX-ALL"])
        if df is None:
            raise RuntimeError("KOSPI 목록을 불러올 수 없습니다.")
        df = df[df.get("Market", "").astype(str).str.contains("KOSPI", na=False)] if "Market" in df.columns else df
        df = _sort_by_marcap(df).head(200)

    code_col, name_col, _ = _pick_cols(df)
    out = pd.DataFrame({
        "code": df[code_col].astype(str).map(_norm_krx_code),
        "name": df[name_col].astype(str),
        "market": "KOSPI",
    })
    out = out.drop_duplicates(subset=["code"]).head(200)
    return out


def build_kosdaq150() -> pd.DataFrame:
    # Try direct index listing first (if supported)
    df = _try_stocklisting(["KOSDAQ150", "KRX-KOSDAQ150", "KOSDAQ 150"])
    if df is not None:
        pass
    else:
        # fallback: take top 150 by marcap from KOSDAQ listing
        df = _try_stocklisting(["KOSDAQ", "KRX", "KRX-ALL"])
        if df is None:
            raise RuntimeError("KOSDAQ 목록을 불러올 수 없습니다.")
        df = df[df.get("Market", "").astype(str).str.contains("KOSDAQ", na=False)] if "Market" in df.columns else df
        df = _sort_by_marcap(df).head(150)

    code_col, name_col, market_col = _pick_cols(df)
    market = df[market_col].astype(str) if market_col and market_col in df.columns else "KOSDAQ"
    out = pd.DataFrame({
        "code": df[code_col].astype(str).map(_norm_krx_code),
        "name": df[name_col].astype(str),
        "market": market,
    })
    out = out.drop_duplicates(subset=["code"]).head(150)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data", help="CSV 출력 폴더")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kospi200 = build_kospi200()
    kosdaq150 = build_kosdaq150()

    (out_dir / "universe_kospi200.csv").write_text(kospi200.to_csv(index=False), encoding="utf-8")
    (out_dir / "universe_kosdaq150.csv").write_text(kosdaq150.to_csv(index=False), encoding="utf-8")

    print("✅ generated:")
    print(f" - {(out_dir / 'universe_kospi200.csv').as_posix()} ({len(kospi200)} rows)")
    print(f" - {(out_dir / 'universe_kosdaq150.csv').as_posix()} ({len(kosdaq150)} rows)")


if __name__ == "__main__":
    main()
