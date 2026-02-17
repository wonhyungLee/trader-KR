"""Daytrade-only backtest entrypoint.

Legacy `src.analyzer.backtest_runner` is intentionally kept as a compatibility
alias, but the underlying engine is now ONLY:
- Active Universe + daytrade (ATR-based limit entry, same-day exit)

Primary outputs:
- data/backtest_active_universe_daytrade/trade_log.csv
- data/backtest_active_universe_daytrade/equity_curve.csv
- data/backtest_active_universe_daytrade/annual_metrics.csv

Compatibility output:
- data/backtest_annual_validation.csv (same annual metrics, daytrade-based)
"""

from __future__ import annotations

from src.analyzer.backtest_runner_active_universe import main as _daytrade_main


def main() -> None:
    _daytrade_main()


if __name__ == "__main__":
    main()
