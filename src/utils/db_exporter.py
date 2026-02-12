from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def _normalize_tables(value) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
        return items or None
    if isinstance(value, Iterable):
        items = [str(v).strip() for v in value if str(v).strip()]
        return items or None
    return None


def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [row[0] for row in cur.fetchall()]


def _order_by_for_table(table: str) -> Optional[str]:
    if table == "daily_price":
        return "date, code"
    if table == "stock_info":
        return "code"
    if table == "universe_members":
        return "code"
    if table == "order_queue":
        return "id"
    if table == "position_state":
        return "code"
    if table == "refill_progress":
        return "code"
    if table == "job_runs":
        return "id"
    return None


def _select_query(table: str, order_by: Optional[str]) -> str:
    if order_by:
        return f'SELECT * FROM "{table}" ORDER BY {order_by}'
    return f'SELECT * FROM "{table}"'


def _max_date_from_csv(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["date"])
    if df.empty:
        return None
    max_date = df["date"].dropna().max()
    return str(max_date) if max_date else None


def export_db(
    db_path: str,
    out_dir: str,
    tables: Optional[List[str]] = None,
    mode: str = "overwrite",
    timestamp: Optional[str] = None,
) -> dict[str, int]:
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        table_list = tables or list_tables(conn)
        results: dict[str, int] = {}
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        for table in table_list:
            if mode == "timestamped":
                dest_dir = out_base / ts
            else:
                dest_dir = out_base
            dest_dir.mkdir(parents=True, exist_ok=True)
            out_path = dest_dir / f"{table}.csv"
            order_by = _order_by_for_table(table)

            if mode == "append" and table == "daily_price":
                max_date = _max_date_from_csv(out_path)
                if not max_date:
                    df = pd.read_sql_query(_select_query(table, order_by), conn)
                    df.to_csv(out_path, index=False)
                    results[table] = len(df)
                else:
                    df = pd.read_sql_query(
                        f'SELECT * FROM "{table}" WHERE date > ? ORDER BY {order_by}',
                        conn,
                        params=(max_date,),
                    )
                    if not df.empty:
                        df.to_csv(out_path, index=False, header=False, mode="a")
                    results[table] = len(df)
            else:
                df = pd.read_sql_query(_select_query(table, order_by), conn)
                df.to_csv(out_path, index=False)
                results[table] = len(df)
        return results
    finally:
        conn.close()


def maybe_export_db(settings: dict, db_path: str) -> Optional[dict[str, int]]:
    cfg = settings.get("export_csv", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return None
    tables = _normalize_tables(cfg.get("tables"))
    out_dir = str(cfg.get("out_dir", "data/csv"))
    mode = str(cfg.get("mode", "overwrite"))
    if mode not in {"overwrite", "timestamped", "append"}:
        mode = "overwrite"
    try:
        return export_db(db_path=db_path, out_dir=out_dir, tables=tables, mode=mode)
    except Exception as exc:
        print(f"[export_csv] failed: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Export SQLite DB tables to CSV files.")
    parser.add_argument("--db", default="data/market_data.db", help="SQLite DB path")
    parser.add_argument("--out-dir", default="data/csv", help="Output directory for CSV files")
    parser.add_argument("--tables", nargs="*", default=None, help="Table names to export (default: all)")
    parser.add_argument("--mode", choices=["overwrite", "timestamped", "append"], default="overwrite")
    parser.add_argument("--timestamp", default=None, help="Timestamp label for timestamped mode")
    args = parser.parse_args()

    results = export_db(
        db_path=args.db,
        out_dir=args.out_dir,
        tables=_normalize_tables(args.tables),
        mode=args.mode,
        timestamp=args.timestamp,
    )
    total = sum(results.values())
    print(f"exported {len(results)} tables, {total} rows -> {args.out_dir}")


if __name__ == "__main__":
    main()
