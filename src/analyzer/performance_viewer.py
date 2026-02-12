"""성과 분석 및 시각화.

- equity_curve.csv, trade_log.csv를 읽어 월별 수익률/낙폭을 계산하고 report.png를 생성한다.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(base_dir: Path = Path("data")):
    equity = pd.read_csv(base_dir / "equity_curve.csv", parse_dates=["date"])
    trades = pd.read_csv(base_dir / "trade_log.csv", parse_dates=["entry_date","exit_date"])
    return equity, trades


def monthly_stats(equity: pd.DataFrame):
    equity = equity.copy()
    equity.set_index("date", inplace=True)
    equity["ret"] = equity["equity"].pct_change()
    monthly = (1 + equity["ret"]).resample("M").prod() - 1
    mdd = (equity["equity"] / equity["equity"].cummax() - 1).resample("M").min()
    return monthly, mdd


def plot_equity(equity: pd.DataFrame, base_dir: Path = Path("data")):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(equity["date"], equity["equity"], label="Equity")
    ax[0].set_title("Equity Curve")
    ax[0].grid(True)

    dd = equity["equity"] / equity["equity"].cummax() - 1
    ax[1].fill_between(equity["date"], dd, color="red", alpha=0.3)
    ax[1].set_title("Drawdown")
    ax[1].grid(True)

    fig.tight_layout()
    out = base_dir / "report.png"
    fig.savefig(out)
    print(f"saved {out}")


def main():
    equity, trades = load_data()
    monthly, mdd = monthly_stats(equity)
    print("Monthly return:")
    print(monthly.tail())
    print("Monthly MDD:")
    print(mdd.tail())
    plot_equity(equity)


if __name__ == "__main__":
    main()
