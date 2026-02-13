import sqlite3
import os
import logging
import re
import pandas as pd
from typing import List, Dict, Any, Optional, Iterable
from datetime import datetime, timedelta


# --- code normalization (KR numeric codes / US tickers) ---

def normalize_code(value) -> str:
    """Normalize codes across markets.

    - KR: numeric 6-digit code (zero-padded)
    - US: ticker symbol (uppercase, no padding)
    """
    if value is None:
        return ''
    s = str(value).strip()
    if not s:
        return ''
    s = s.upper()
    return s.zfill(6) if s.isdigit() else s

SCHEMA = {
    "universe_members": """
        CREATE TABLE IF NOT EXISTS universe_members (
            code TEXT PRIMARY KEY,
            market TEXT,
            name TEXT,
            group_name TEXT,
            updated_at TEXT
        );
    """,
    "stock_info": """
        CREATE TABLE IF NOT EXISTS stock_info (
            code TEXT PRIMARY KEY,
            name TEXT,
            market TEXT,
            marcap REAL,
            updated_at TEXT
        );
    """,
    "daily_price": """
        CREATE TABLE IF NOT EXISTS daily_price (
            date TEXT,
            code TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            amount REAL,
            ma25 REAL,
            disparity REAL,
            PRIMARY KEY (date, code)
        );
    """,
    "order_queue": """
        CREATE TABLE IF NOT EXISTS order_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_date TEXT,
            exec_date TEXT,
            code TEXT,
            side TEXT,
            qty INTEGER,
            rank INTEGER,
            status TEXT,
            ord_dvsn TEXT,
            ord_unpr REAL,
            odno TEXT,
            ord_orgno TEXT,
            filled_qty INTEGER,
            avg_price REAL,
            api_resp TEXT,
            cancel_resp TEXT,
            created_at TEXT,
            sent_at TEXT,
            updated_at TEXT
        );
    """,
    "position_state": """
        CREATE TABLE IF NOT EXISTS position_state (
            code TEXT PRIMARY KEY,
            name TEXT,
            qty INTEGER,
            avg_price REAL,
            entry_date TEXT,
            updated_at TEXT
        );
    """,
    "refill_progress": """
        CREATE TABLE IF NOT EXISTS refill_progress (
            code TEXT PRIMARY KEY,
            next_end_date TEXT,
            last_min_date TEXT,
            status TEXT,
            updated_at TEXT,
            message TEXT
        );
    """,
    "job_runs": """
        CREATE TABLE IF NOT EXISTS job_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_name TEXT,
            started_at TEXT,
            finished_at TEXT,
            status TEXT,
            message TEXT
        );
    """,
    "investor_flow_daily": """
        CREATE TABLE IF NOT EXISTS investor_flow_daily (
            date TEXT,
            code TEXT,
            foreign_net_value REAL,
            inst_net_value REAL,
            indiv_net_value REAL,
            updated_at TEXT,
            PRIMARY KEY (date, code)
        );
    """,
    "program_trade_daily": """
        CREATE TABLE IF NOT EXISTS program_trade_daily (
            date TEXT,
            code TEXT,
            program_net_value REAL,
            updated_at TEXT,
            PRIMARY KEY (date, code)
        );
    """,
    "short_sale_daily": """
        CREATE TABLE IF NOT EXISTS short_sale_daily (
            date TEXT,
            code TEXT,
            short_volume REAL,
            short_value REAL,
            short_ratio REAL,
            updated_at TEXT,
            PRIMARY KEY (date, code)
        );
    """,
    "credit_balance_daily": """
        CREATE TABLE IF NOT EXISTS credit_balance_daily (
            date TEXT,
            code TEXT,
            credit_qty REAL,
            credit_value REAL,
            updated_at TEXT,
            PRIMARY KEY (date, code)
        );
    """,
    "loan_trans_daily": """
        CREATE TABLE IF NOT EXISTS loan_trans_daily (
            date TEXT,
            code TEXT,
            loan_qty REAL,
            loan_value REAL,
            updated_at TEXT,
            PRIMARY KEY (date, code)
        );
    """,
    "vi_status_daily": """
        CREATE TABLE IF NOT EXISTS vi_status_daily (
            date TEXT,
            code TEXT,
            vi_count INTEGER,
            updated_at TEXT,
            PRIMARY KEY (date, code)
        );
    """,
    "sector_map": """
        CREATE TABLE IF NOT EXISTS sector_map (
            code TEXT PRIMARY KEY,
            sector_code TEXT,
            sector_name TEXT,
            industry_code TEXT,
            industry_name TEXT,
            updated_at TEXT,
            source TEXT
        );
    """,
    "universe_changes": """
        CREATE TABLE IF NOT EXISTS universe_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT,
            market TEXT,
            added_codes_json TEXT,
            removed_codes_json TEXT
        );
    """,
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_daily_price_code_date ON daily_price(code, date);",
    "CREATE INDEX IF NOT EXISTS idx_order_queue_exec_status ON order_queue(exec_date, status);",
    "CREATE INDEX IF NOT EXISTS idx_order_queue_status ON order_queue(status);",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_order_queue_exec_code_side_pending ON order_queue(exec_date, code, side) WHERE status='PENDING';",
    "CREATE INDEX IF NOT EXISTS idx_refill_progress_status ON refill_progress(status);",
    "CREATE INDEX IF NOT EXISTS idx_job_runs_job_name ON job_runs(job_name);",
    "CREATE INDEX IF NOT EXISTS idx_investor_flow_daily_code_date ON investor_flow_daily(code, date);",
    "CREATE INDEX IF NOT EXISTS idx_program_trade_daily_code_date ON program_trade_daily(code, date);",
    "CREATE INDEX IF NOT EXISTS idx_short_sale_daily_code_date ON short_sale_daily(code, date);",
    "CREATE INDEX IF NOT EXISTS idx_credit_balance_daily_code_date ON credit_balance_daily(code, date);",
    "CREATE INDEX IF NOT EXISTS idx_loan_trans_daily_code_date ON loan_trans_daily(code, date);",
    "CREATE INDEX IF NOT EXISTS idx_vi_status_daily_code_date ON vi_status_daily(code, date);",
    "CREATE INDEX IF NOT EXISTS idx_sector_map_sector ON sector_map(sector_name);",
    "CREATE INDEX IF NOT EXISTS idx_sector_map_updated ON sector_map(updated_at);",
    "CREATE INDEX IF NOT EXISTS idx_universe_changes_date ON universe_changes(snapshot_date);",
]


class SQLiteStore:
    def __init__(self, db_path: str = "data/market_data.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA busy_timeout=5000;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.row_factory = sqlite3.Row
        self._refill_cols: Optional[set[str]] = None
        self.ensure_schema()

    def ensure_schema(self):
        cur = self.conn.cursor()
        for ddl in SCHEMA.values():
            cur.execute(ddl)
        self._ensure_refill_progress_columns()
        for idx in INDEXES:
            try:
                cur.execute(idx)
            except (sqlite3.IntegrityError, sqlite3.OperationalError) as exc:
                logging.warning('failed to create index: %s (%s)', idx, exc)
        self.conn.commit()

    def _ensure_refill_progress_columns(self):
        try:
            cur = self.conn.execute("PRAGMA table_info(refill_progress)")
            cols = {row[1] for row in cur.fetchall()}
            if "next_end_date" not in cols:
                self.conn.execute("ALTER TABLE refill_progress ADD COLUMN next_end_date TEXT")
            if "last_min_date" not in cols:
                self.conn.execute("ALTER TABLE refill_progress ADD COLUMN last_min_date TEXT")
            if "message" not in cols:
                self.conn.execute("ALTER TABLE refill_progress ADD COLUMN message TEXT")
            self._refill_cols = cols | {"next_end_date", "last_min_date", "message"}
        except Exception:
            pass

    # ---------- universe_members ----------
    def upsert_universe_members(self, rows: Iterable[Dict[str, Any]]):
        now = datetime.utcnow().isoformat()
        data = []
        codes = []
        for r in rows:
            code = normalize_code(r.get("code"))
            if not code:
                continue
            codes.append(code)
            data.append(
                (
                    code,
                    r.get("market"),
                    r.get("name"),
                    r.get("group_name"),
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT INTO universe_members(code, market, name, group_name, updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(code) DO UPDATE SET
                market=excluded.market,
                name=excluded.name,
                group_name=excluded.group_name,
                updated_at=excluded.updated_at;
            """,
            data,
        )
        if codes:
            placeholder = ",".join("?" * len(codes))
            self.conn.execute(f"DELETE FROM universe_members WHERE code NOT IN ({placeholder})", tuple(codes))
        self.conn.commit()

    def list_universe_codes(self) -> List[str]:
        cur = self.conn.execute("SELECT code FROM universe_members ORDER BY code")
        rows = [row[0] for row in cur.fetchall()]
        return rows

    def load_universe_df(self) -> pd.DataFrame:
        cur = self.conn.execute("SELECT code, name, market, group_name FROM universe_members ORDER BY code")
        return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])

    # ---------- stock_info ----------
    def upsert_stock_info(self, rows: Iterable[Dict[str, Any]]):
        now = datetime.utcnow().isoformat()
        data = []
        for r in rows:
            data.append(
                (
                    r.get("code"),
                    r.get("name"),
                    r.get("market"),
                    float(r.get("marcap") or 0),
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT INTO stock_info(code, name, market, marcap, updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(code) DO UPDATE SET
                name=excluded.name,
                market=excluded.market,
                marcap=excluded.marcap,
                updated_at=excluded.updated_at;
            """,
            data,
        )
        self.conn.commit()

    def list_stock_codes(self) -> List[str]:
        return self.list_universe_codes()

    # ---------- sector_map ----------
    def upsert_sector_map(self, rows: Iterable[Dict[str, Any]]):
        now = datetime.utcnow().isoformat()
        data = []
        for r in rows:
            code = normalize_code(r.get("code"))
            if not code:
                continue
            data.append(
                (
                    code,
                    r.get("sector_code"),
                    r.get("sector_name"),
                    r.get("industry_code"),
                    r.get("industry_name"),
                    r.get("updated_at") or now,
                    r.get("source"),
                )
            )
        if not data:
            return
        self.conn.executemany(
            """
            INSERT INTO sector_map(code, sector_code, sector_name, industry_code, industry_name, updated_at, source)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(code) DO UPDATE SET
                sector_code=excluded.sector_code,
                sector_name=excluded.sector_name,
                industry_code=excluded.industry_code,
                industry_name=excluded.industry_name,
                updated_at=excluded.updated_at,
                source=excluded.source;
            """,
            data,
        )
        self.conn.commit()

    def list_sector_targets(self, refresh_days: int) -> List[str]:
        if refresh_days <= 0:
            cur = self.conn.execute("SELECT code FROM universe_members ORDER BY code")
            return [row[0] for row in cur.fetchall()]
        threshold = (datetime.utcnow() - timedelta(days=refresh_days)).isoformat()
        cur = self.conn.execute(
            """
            SELECT u.code
            FROM universe_members u
            LEFT JOIN sector_map s ON u.code = s.code
            WHERE s.updated_at IS NULL OR s.updated_at < ?
            ORDER BY u.code
            """,
            (threshold,),
        )
        return [row[0] for row in cur.fetchall()]

    def list_sector_unknowns(self) -> List[str]:
        cur = self.conn.execute(
            "SELECT u.code FROM universe_members u LEFT JOIN sector_map s ON u.code=s.code WHERE s.sector_name IS NULL"
        )
        return [row[0] for row in cur.fetchall()]

    # ---------- universe_changes ----------
    def insert_universe_change(self, snapshot_date: str, market: str, added_json: str, removed_json: str):
        self.conn.execute(
            """
            INSERT INTO universe_changes(snapshot_date, market, added_codes_json, removed_codes_json)
            VALUES(?,?,?,?)
            """,
            (snapshot_date, market, added_json, removed_json),
        )
        self.conn.commit()

    def get_stock(self, code: str) -> Optional[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM stock_info WHERE code=?", (code,))
        return cur.fetchone()

    def replace_stock_info(self, rows: Iterable[Dict[str, Any]]):
        now = datetime.utcnow().isoformat()
        data = []
        codes = []
        for r in rows:
            code = normalize_code(r.get("code"))
            if not code:
                continue
            codes.append(code)
            data.append(
                (
                    code,
                    r.get("name"),
                    r.get("market"),
                    float(r.get("marcap") or 0),
                    now,
                )
            )
        self.conn.execute("DELETE FROM stock_info")
        if data:
            self.conn.executemany(
                """
                INSERT INTO stock_info(code, name, market, marcap, updated_at)
                VALUES(?,?,?,?,?)
                """,
                data,
            )
        self.conn.commit()

    # ---------- job_runs ----------
    def _extract_job_pid(self, message: str) -> Optional[int]:
        if not message:
            return None
        match = re.search(r"\bpid=(\d+)\b", str(message))
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
        return None

    def _is_process_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    def cleanup_stale_running_jobs(self, stale_sec: int = 24 * 60 * 60) -> int:
        """Mark old RUNNING rows as ERROR when process has died or job duration exceeded.

        Returns the number of cleaned rows.
        """
        limit_dt = (datetime.utcnow() - timedelta(seconds=stale_sec)).isoformat()
        rows = self.conn.execute(
            "SELECT id, message, started_at FROM job_runs WHERE status='RUNNING' AND started_at < ?",
            (limit_dt,),
        ).fetchall()
        updated = 0
        for row in rows:
            pid = self._extract_job_pid(row["message"])
            if pid and self._is_process_alive(pid):
                continue
            self.conn.execute(
                "UPDATE job_runs SET finished_at=?, status='ERROR', message=? WHERE id=?",
                (
                    datetime.utcnow().isoformat(),
                    f"{row['message']} | stale_running_cleanup",
                    row[0],
                ),
            )
            updated += 1
        if updated:
            self.conn.commit()
        return updated

    def start_job(self, job_name: str, message: str = "") -> int:
        now = datetime.utcnow().isoformat()
        pid = os.getpid()
        msg = str(message or "").strip()
        if msg:
            if "pid=" not in msg:
                msg = f"{msg} pid={pid}"
        else:
            msg = f"pid={pid}"
        cur = self.conn.execute(
            "INSERT INTO job_runs(job_name, started_at, status, message) VALUES(?,?,?,?)",
            (job_name, now, "RUNNING", msg),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_job(self, job_id: int, status: str = "SUCCESS", message: str = ""):
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            "UPDATE job_runs SET finished_at=?, status=?, message=? WHERE id=?",
            (now, status, message or "", job_id),
        )
        self.conn.commit()

    # ---------- daily_price ----------
    def upsert_daily_prices(self, code: str, df: pd.DataFrame):
        """Insert/replace daily price rows.

        Note: pandas may carry NaN values for indicators. We normalize NaN -> NULL to avoid
        propagating IEEE NaN into SQLite (which can break comparisons and filters downstream).
        """
        if df.empty:
            return
        cols = ["date", "code", "open", "high", "low", "close", "volume", "amount", "ma25", "disparity"]
        df = df.copy()
        df["code"] = code
        df = df[cols]
        # IMPORTANT: cast to object before null-normalization.
        # Otherwise float columns keep NaN dtype and None turns back into NaN.
        df = df.astype(object).where(pd.notna(df), None)
        records = list(df.itertuples(index=False, name=None))
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO daily_price(date, code, open, high, low, close, volume, amount, ma25, disparity)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            records,
        )
        self.conn.commit()

    def last_price_date(self, code: str) -> Optional[str]:
        cur = self.conn.execute("SELECT max(date) FROM daily_price WHERE code=?", (code,))
        row = cur.fetchone()
        return row[0] if row and row[0] else None

    def load_recent_closes(self, code: str, end_date: Optional[str] = None, limit: int = 120) -> pd.DataFrame:
        """Load recent (date, close) rows for indicator recomputation.

        Args:
            code: stock code
            end_date: inclusive end date (YYYY-MM-DD). If None, use latest.
            limit: maximum number of rows (trading days) to return.

        Returns:
            DataFrame sorted by date ascending with columns [date, close].
        """
        code = normalize_code(code)
        limit = int(limit) if limit else 120
        if end_date:
            cur = self.conn.execute(
                "SELECT date, close FROM daily_price WHERE code=? AND date<=? ORDER BY date DESC LIMIT ?",
                (code, end_date, limit),
            )
        else:
            cur = self.conn.execute(
                "SELECT date, close FROM daily_price WHERE code=? ORDER BY date DESC LIMIT ?",
                (code, limit),
            )
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=["date", "close"])
        df = pd.DataFrame(rows, columns=[c[0] for c in cur.description])
        return df.sort_values("date")

    def load_prices(self, codes: List[str]) -> pd.DataFrame:
        placeholder = ",".join("?" * len(codes))
        cur = self.conn.execute(
            f"SELECT * FROM daily_price WHERE code IN ({placeholder})", tuple(codes)
        )
        return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])

    def load_all_prices(self) -> pd.DataFrame:
        cur = self.conn.execute("SELECT * FROM daily_price")
        return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])

    # ---------- order_queue ----------
    def add_pending_orders(self, orders: List[Dict[str, Any]], exec_date: str):
        now = datetime.utcnow().isoformat()
        # Idempotency: 같은 exec_date에 대해 PENDING을 중복 생성하지 않도록 기존 PENDING을 제거 후 재생성한다.
        self.conn.execute("DELETE FROM order_queue WHERE exec_date=? AND status='PENDING'", (exec_date,))
        rows = []
        for o in orders:
            rows.append(
                (
                    o.get("signal_date"),
                    exec_date,
                    o["code"],
                    o["side"],
                    int(o["qty"]),
                    int(o.get("rank", 0)),
                    "PENDING",
                    o.get("ord_dvsn", "01"),
                    float(o.get("ord_unpr") or 0),
                    None,
                    None,
                    0,
                    0.0,
                    None,
                    None,
                    now,
                    None,
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT INTO order_queue(
                signal_date, exec_date, code, side, qty, rank, status, ord_dvsn, ord_unpr,
                odno, ord_orgno, filled_qty, avg_price, api_resp, cancel_resp, created_at, sent_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        self.conn.commit()

    def list_orders(self, status: Optional[List[str]] = None, exec_date: Optional[str] = None) -> List[sqlite3.Row]:
        query = "SELECT * FROM order_queue WHERE 1=1"
        params: List[Any] = []
        if status:
            placeholder = ",".join("?" * len(status))
            query += f" AND status IN ({placeholder})"
            params.extend(status)
        if exec_date:
            query += " AND exec_date=?"
            params.append(exec_date)
        query += " ORDER BY rank ASC, id ASC"
        cur = self.conn.execute(query, tuple(params))
        return cur.fetchall()

    def update_order_status(self, order_id: int, status: str, **kwargs):
        fields = ["status=?", "updated_at=?"]
        params: List[Any] = [status, datetime.utcnow().isoformat()]
        if "odno" in kwargs:
            fields.append("odno=?")
            params.append(kwargs["odno"])
        if "ord_orgno" in kwargs:
            fields.append("ord_orgno=?")
            params.append(kwargs["ord_orgno"])
        if "api_resp" in kwargs:
            fields.append("api_resp=?")
            params.append(kwargs["api_resp"])
        if "cancel_resp" in kwargs:
            fields.append("cancel_resp=?")
            params.append(kwargs["cancel_resp"])
        if "sent_at" in kwargs:
            fields.append("sent_at=?")
            params.append(kwargs["sent_at"])
        if "filled_qty" in kwargs:
            fields.append("filled_qty=?")
            params.append(kwargs["filled_qty"])
        if "avg_price" in kwargs:
            fields.append("avg_price=?")
            params.append(kwargs["avg_price"])
        sql = f"UPDATE order_queue SET {', '.join(fields)} WHERE id=?"
        params.append(order_id)
        self.conn.execute(sql, tuple(params))
        self.conn.commit()

    # ---------- position_state ----------
    def upsert_position(self, code: str, name: str, qty: int, avg_price: float, entry_date: str):
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO position_state(code, name, qty, avg_price, entry_date, updated_at)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(code) DO UPDATE SET
                qty=excluded.qty,
                avg_price=excluded.avg_price,
                entry_date=excluded.entry_date,
                updated_at=excluded.updated_at;
            """,
            (code, name, qty, avg_price, entry_date, now),
        )
        self.conn.commit()

    def list_positions(self) -> List[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM position_state")
        return cur.fetchall()

    def replace_positions(self, positions: List[Dict[str, Any]], entry_date: str):
        """잔고 조회 결과를 최종 진실로 보고 position_state를 통째로 재구성한다.

        - qty>0 필터는 호출 측에서 수행하는 것을 권장
        - 운영 중 유령 포지션(이미 청산된 종목)이 남지 않게 한다
        """
        now = datetime.utcnow().isoformat()
        cur = self.conn.cursor()
        cur.execute("DELETE FROM position_state")
        rows = []
        for p in positions:
            code = p.get('code')
            if not code:
                continue
            rows.append((code, p.get('name',''), int(p.get('qty') or 0), float(p.get('avg_price') or 0), entry_date, now))
        cur.executemany(
            """
            INSERT INTO position_state(code, name, qty, avg_price, entry_date, updated_at)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(code) DO UPDATE SET
                name=excluded.name,
                qty=excluded.qty,
                avg_price=excluded.avg_price,
                entry_date=excluded.entry_date,
                updated_at=excluded.updated_at;
            """,
            rows,
        )
        self.conn.commit()

    # ---------- refill_progress ----------
    def get_refill_status(self, code: str) -> Optional[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM refill_progress WHERE code=?", (code,))
        return cur.fetchone()

    def upsert_refill_status(
        self,
        code: str,
        next_end: Optional[str],
        last_min: Optional[str],
        status: str,
        message: str = "",
    ):
        now = datetime.utcnow().isoformat()
        cols = self._refill_cols
        if cols is None:
            cur = self.conn.execute("PRAGMA table_info(refill_progress)")
            cols = {row[1] for row in cur.fetchall()}
            self._refill_cols = cols

        payload = {
            "code": code,
            "next_end_date": next_end,
            "last_min_date": last_min,
            "status": status,
            "updated_at": now,
            "message": message or "",
        }
        # Backward compatibility columns if present
        if "last_fetched_end_date" in cols:
            payload["last_fetched_end_date"] = next_end
        if "min_date_in_db" in cols:
            payload["min_date_in_db"] = last_min

        cols_list = [c for c in payload.keys() if c in cols or c in {"code", "next_end_date", "last_min_date", "status", "updated_at", "message", "last_fetched_end_date", "min_date_in_db"}]
        cols_list = list(dict.fromkeys(cols_list))
        placeholders = ",".join(["?"] * len(cols_list))
        updates = ",".join([f"{c}=excluded.{c}" for c in cols_list if c != "code"])
        values = [payload[c] for c in cols_list]

        self.conn.execute(
            f"INSERT INTO refill_progress({','.join(cols_list)}) VALUES({placeholders}) "
            f"ON CONFLICT(code) DO UPDATE SET {updates}",
            values,
        )
        self.conn.commit()

    # ---------- accuracy data ----------
    def upsert_investor_flow(self, rows: Iterable[Dict[str, Any]]):
        now = datetime.utcnow().isoformat()
        data = []
        for r in rows:
            data.append(
                (
                    r.get("date"),
                    r.get("code"),
                    float(r.get("foreign_net_value") or 0),
                    float(r.get("inst_net_value") or 0),
                    float(r.get("indiv_net_value") or 0),
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT INTO investor_flow_daily(date, code, foreign_net_value, inst_net_value, indiv_net_value, updated_at)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(date, code) DO UPDATE SET
                foreign_net_value=excluded.foreign_net_value,
                inst_net_value=excluded.inst_net_value,
                indiv_net_value=excluded.indiv_net_value,
                updated_at=excluded.updated_at;
            """,
            data,
        )
        self.conn.commit()

    def upsert_program_trade(self, rows: Iterable[Dict[str, Any]]):
        now = datetime.utcnow().isoformat()
        data = []
        for r in rows:
            data.append(
                (
                    r.get("date"),
                    r.get("code"),
                    float(r.get("program_net_value") or 0),
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT INTO program_trade_daily(date, code, program_net_value, updated_at)
            VALUES(?,?,?,?)
            ON CONFLICT(date, code) DO UPDATE SET
                program_net_value=excluded.program_net_value,
                updated_at=excluded.updated_at;
            """,
            data,
        )
        self.conn.commit()

    def upsert_short_sale(self, rows: Iterable[Dict[str, Any]]):
        now = datetime.utcnow().isoformat()
        data = []
        for r in rows:
            data.append(
                (
                    r.get("date"),
                    r.get("code"),
                    float(r.get("short_volume") or 0),
                    float(r.get("short_value") or 0),
                    float(r.get("short_ratio") or 0),
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT INTO short_sale_daily(date, code, short_volume, short_value, short_ratio, updated_at)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(date, code) DO UPDATE SET
                short_volume=excluded.short_volume,
                short_value=excluded.short_value,
                short_ratio=excluded.short_ratio,
                updated_at=excluded.updated_at;
            """,
            data,
        )
        self.conn.commit()

    def upsert_credit_balance(self, rows: Iterable[Dict[str, Any]]):
        now = datetime.utcnow().isoformat()
        data = []
        for r in rows:
            data.append(
                (
                    r.get("date"),
                    r.get("code"),
                    float(r.get("credit_qty") or 0),
                    float(r.get("credit_value") or 0),
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT INTO credit_balance_daily(date, code, credit_qty, credit_value, updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(date, code) DO UPDATE SET
                credit_qty=excluded.credit_qty,
                credit_value=excluded.credit_value,
                updated_at=excluded.updated_at;
            """,
            data,
        )
        self.conn.commit()

    def upsert_loan_trans(self, rows: Iterable[Dict[str, Any]]):
        now = datetime.utcnow().isoformat()
        data = []
        for r in rows:
            data.append(
                (
                    r.get("date"),
                    r.get("code"),
                    float(r.get("loan_qty") or 0),
                    float(r.get("loan_value") or 0),
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT INTO loan_trans_daily(date, code, loan_qty, loan_value, updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(date, code) DO UPDATE SET
                loan_qty=excluded.loan_qty,
                loan_value=excluded.loan_value,
                updated_at=excluded.updated_at;
            """,
            data,
        )
        self.conn.commit()

    def upsert_vi_status(self, rows: Iterable[Dict[str, Any]]):
        now = datetime.utcnow().isoformat()
        data = []
        for r in rows:
            data.append(
                (
                    r.get("date"),
                    r.get("code"),
                    int(r.get("vi_count") or 0),
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT INTO vi_status_daily(date, code, vi_count, updated_at)
            VALUES(?,?,?,?)
            ON CONFLICT(date, code) DO UPDATE SET
                vi_count=excluded.vi_count,
                updated_at=excluded.updated_at;
            """,
            data,
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
