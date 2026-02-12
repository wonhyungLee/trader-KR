import sqlite3
import pandas as pd
import os

db_path = "data/market_data.db"

if not os.path.exists(db_path):
    print("Database not found.")
    exit()

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=== DB Inspection Report ===")

# 1. Table Counts
tables = ["stock_info", "daily_price", "refill_progress", "investor_flow_daily", "program_trade_daily", "short_sale_daily"]
print("\n[Table Row Counts]")
for t in tables:
    try:
        cur.execute(f"SELECT count(*) FROM {t}")
        row = cur.fetchone()
        print(f"{t}: {row[0]}")
    except Exception as e:
        print(f"{t}: (Error: {e})")

# 2. Refill Progress
print("\n[Refill Progress Summary]")
try:
    df_refill = pd.read_sql("SELECT status, count(*) as cnt, min(updated_at) as oldest_update, max(updated_at) as newest_update FROM refill_progress GROUP BY status", conn)
    print(df_refill.to_string())
except Exception as e:
    print(f"Error reading refill_progress: {e}")

conn.close()