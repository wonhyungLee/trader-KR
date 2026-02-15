from __future__ import annotations

import sqlite3
from typing import List, Tuple, Optional

def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def list_codes(db_path: str, table: str = "daily_price", min_rows: int = 1) -> List[Tuple[str, int]]:
    """Return [(code, n_rows), ...]"""
    conn = connect(db_path)
    try:
        cur = conn.execute(
            f"SELECT code, COUNT(*) as n FROM {table} GROUP BY code HAVING n >= ? ORDER BY code",
            (int(min_rows),),
        )
        return [(str(r[0]), int(r[1])) for r in cur.fetchall()]
    finally:
        conn.close()

def fetch_ohlc(
    db_path: str,
    code: str,
    table: str = "daily_price",
    limit: Optional[int] = None,
    desc: bool = False,
    with_date: bool = True,
):
    """Fetch OHLC rows for a code.

    Returns:
      - if with_date=True: (dates, o, h, l, c)
      - else: (o, h, l, c)

    Note:
      - If desc=True, it fetches in DESC order, then reverses to ASC for indicator alignment.
    """
    conn = connect(db_path)
    try:
        order = "DESC" if desc else "ASC"
        lim_sql = f" LIMIT {int(limit)}" if limit is not None else ""
        cols = "date, open, high, low, close" if with_date else "open, high, low, close"
        cur = conn.execute(
            f"SELECT {cols} FROM {table} WHERE code=? ORDER BY date {order}{lim_sql}",
            (code,),
        )
        rows = cur.fetchall()
        if not rows:
            if with_date:
                return [], [], [], [], []
            return [], [], [], []

        if desc:
            rows = list(reversed(rows))

        if with_date:
            dates = [str(r[0]) for r in rows]
            o = [float(r[1]) for r in rows]
            h = [float(r[2]) for r in rows]
            l = [float(r[3]) for r in rows]
            c = [float(r[4]) for r in rows]
            return dates, o, h, l, c
        else:
            o = [float(r[0]) for r in rows]
            h = [float(r[1]) for r in rows]
            l = [float(r[2]) for r in rows]
            c = [float(r[3]) for r in rows]
            return o, h, l, c
    finally:
        conn.close()
