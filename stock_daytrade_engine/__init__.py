"""Stock day-trade recommendation engine (LONG only).

Core idea (daily bars, next-day daytrade):
- Signal at day t close: uptrend + short-term oversold (Connors-style)
- Place next-day limit buy at: close[t] - entry_k * ATR[t]
- Bracket exits on day t+1:
    * stop = entry - stop_mult * ATR[t]
    * take-profit = entry + target_mult * ATR[t]
    * if neither hit, exit at day t+1 close (EOD)
- Objective: maximize win rate (primary), with light tie-breakers.
"""

__all__ = [
    "config",
    "db",
    "indicators",
    "backtester",
    "recommender",
]
