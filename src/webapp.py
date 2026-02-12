from __future__ import annotations

import sqlite3
from pathlib import Path
from flask import Flask, render_template_string, request
import pandas as pd


DB_PATH = Path("data/market_data.db")

PAGE = """
<!doctype html>
<html><head>
  <title>BNF-K Viewer</title>
  <style>
    body { font-family: sans-serif; margin: 24px; }
    table { border-collapse: collapse; }
    th, td { padding: 6px 10px; border: 1px solid #ccc; }
    form { margin-bottom: 12px; }
  </style>
</head><body>
  <h2>일봉 조회</h2>
  <form method="get">
    코드: <input name="code" value="{{code}}" size="8" />
    최근 N일: <input name="days" value="{{days}}" size="4" />
    <button type="submit">조회</button>
  </form>
  {% if df is not none %}
  <p>{{len}} rows</p>
  <table>
    <tr>{% for c in df.columns %}<th>{{c}}</th>{% endfor %}</tr>
    {% for _, row in df.iterrows() %}
      <tr>{% for c in df.columns %}<td>{{row[c]}}</td>{% endfor %}</tr>
    {% endfor %}
  </table>
  {% endif %}
</body></html>
"""


def load_prices(code: str, days: int):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT date, open, high, low, close, volume, amount
        FROM daily_price
        WHERE code=?
        ORDER BY date DESC
        LIMIT ?
        """,
        conn,
        params=(code, days),
    )
    return df


app = Flask(__name__)


@app.route("/")
def index():
    code = request.args.get("code", "005930")
    days = int(request.args.get("days", "30"))
    df = load_prices(code, days)
    return render_template_string(PAGE, df=df, code=code, days=days, len=len(df))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
