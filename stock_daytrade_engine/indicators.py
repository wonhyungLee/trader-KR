from __future__ import annotations

from typing import Sequence
import numpy as np

def rolling_sma(values: Sequence[float], period: int) -> np.ndarray:
    """Simple moving average aligned to each index (NaN until enough bars)."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period or period <= 0:
        return out
    cumsum = np.cumsum(arr)
    out[period - 1 :] = (cumsum[period - 1 :] - np.concatenate(([0.0], cumsum[: -period]))) / period
    return out

def rsi_sma(values: Sequence[float], period: int = 2) -> np.ndarray:
    """RSI using simple average of the last `period` gains/losses (Connors RSI2 style).

    Returns array with NaN for first `period` bars where RSI is undefined.

    Note:
      - This uses a simple moving average (SMA) of gains/losses, not Wilder's RMA.
      - It's intentionally consistent with many Connors RSI(2) formulations.
    """
    c = np.asarray(values, dtype=float)
    n = len(c)
    out = np.full(n, np.nan)
    if n < period + 1 or period <= 0:
        return out

    d = np.diff(c)
    gains = np.clip(d, 0, None)
    losses = np.clip(-d, 0, None)

    # map diffs (len n-1) to indices 1..n-1
    g = np.zeros(n, dtype=float)
    l = np.zeros(n, dtype=float)
    g[1:] = gains
    l[1:] = losses

    gsum = np.cumsum(g)
    lsum = np.cumsum(l)

    # RSI at index i uses gains/losses over (i-period+1..i)
    for i in range(period, n):
        g_win = gsum[i] - gsum[i - period]
        l_win = lsum[i] - lsum[i - period]
        g_avg = g_win / period
        l_avg = l_win / period
        if l_avg == 0.0 and g_avg == 0.0:
            out[i] = 50.0
        elif l_avg == 0.0:
            out[i] = 100.0
        else:
            rs = g_avg / l_avg
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out

def atr_sma(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> np.ndarray:
    """ATR using simple moving average of True Range over `period` bars.
    NaN until index >= period.

    True Range:
      TR = max(high-low, abs(high-prev_close), abs(low-prev_close))

    Note:
      - We set TR[0]=0 and start ATR from index=period (needs period diffs).
    """
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    n = len(c)
    out = np.full(n, np.nan)
    if n < period + 1 or period <= 0:
        return out

    prev_close = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))

    tr0 = np.zeros(n, dtype=float)
    tr0[1:] = tr[1:]

    s = float(np.sum(tr0[1 : period + 1]))
    out[period] = s / period
    for i in range(period + 1, n):
        s += float(tr0[i] - tr0[i - period])
        out[i] = s / period
    return out
