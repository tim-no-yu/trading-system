"""
Pairs trading backtest: spread z-score strategy.
Uses beta from training only, rolling z-score (no lookahead), trades on z thresholds.
Outputs train vs test metrics and saves plots to outputs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from analysis.pairs_regression import (
    TRAIN_FRAC,
    load_and_align,
    ols_residuals,
    train_test_split,
)

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
Z_ENTRY_LONG = -2.0
Z_ENTRY_SHORT = 2.0
Z_EXIT = 0.0
ROLLING_WINDOW = 20


def compute_spread_zscore(
    spread: pd.Series,
    window: int = ROLLING_WINDOW,
) -> pd.Series:
    """Rolling z-score without lookahead: mean/std use only past data (shift(1))."""
    rolling_mean = spread.rolling(window, min_periods=2).mean().shift(1)
    rolling_std = spread.rolling(window, min_periods=2).std().shift(1)
    z = (spread - rolling_mean) / rolling_std
    return z.replace([np.inf, -np.inf], np.nan)


def run_backtest(
    spread: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """
    Trade: long when z < -2, short when z > 2, exit when z crosses 0.
    Returns (position array, list of trade dicts).
    """
    n = len(spread)
    position = np.zeros(n)
    trades: list[dict] = []

    pos = 0
    entry_idx = -1
    entry_spread = 0.0

    for i in range(1, n):
        z_curr = z[i]
        z_prev = z[i - 1]

        if np.isnan(z_curr):
            position[i] = pos
            continue

        # Exit when z crosses 0
        if pos != 0 and (z_prev * z_curr <= 0 or z_curr == 0):
            pnl = pos * (spread[i] - entry_spread)
            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": i,
                "side": pos,
                "pnl": pnl,
                "duration": i - entry_idx,
            })
            pos = 0
            position[i] = 0
            continue

        # Entry when flat
        if pos == 0:
            if z_curr < Z_ENTRY_LONG:
                pos = 1
                entry_idx = i
                entry_spread = spread[i]
            elif z_curr > Z_ENTRY_SHORT:
                pos = -1
                entry_idx = i
                entry_spread = spread[i]

        position[i] = pos

    return position, trades


def compute_metrics(
    spread: np.ndarray,
    position: np.ndarray,
    trades: list[dict],
) -> dict:
    """Compute PnL, Sharpe, max drawdown, win rate, avg duration, #trades."""
    pnl = np.zeros(len(spread))
    for i in range(1, len(spread)):
        pnl[i] = position[i - 1] * (spread[i] - spread[i - 1])

    pnl_sum = pnl.sum()
    pnl_std = pnl.std()
    sharpe = (pnl_sum / len(pnl)) / pnl_std * np.sqrt(252) if pnl_std > 0 else 0.0

    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_dd = drawdown.max()

    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = wins / len(trades) if trades else 0.0
    avg_duration = np.mean([t["duration"] for t in trades]) if trades else 0.0

    return {
        "total_pnl": pnl_sum,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "avg_duration": avg_duration,
        "n_trades": len(trades),
    }


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_and_align()
    train, test = train_test_split(df, TRAIN_FRAC)

    # Beta from training only
    alpha, beta, _ = ols_residuals(train["NVDA"].values, train["AMD"].values)

    # Spread = NVDA - beta * AMD (fixed beta)
    train_spread = train["NVDA"].values - beta * train["AMD"].values
    test_spread = test["NVDA"].values - beta * test["AMD"].values

    # Z-score: rolling mean/std from training window only (no lookahead)
    # Train: rolling with shift(1). Test: fixed mean/std from train (no lookahead).
    train_z = compute_spread_zscore(pd.Series(train_spread), ROLLING_WINDOW).values
    train_mean = float(np.nanmean(train_spread))
    train_std = float(np.nanstd(train_spread))
    if train_std <= 0:
        train_std = 1e-10
    test_z = (test_spread - train_mean) / train_std

    # Backtest train
    train_pos, train_trades = run_backtest(train_spread, train_z)
    train_metrics = compute_metrics(train_spread, train_pos, train_trades)

    # Backtest test
    test_pos, test_trades = run_backtest(test_spread, test_z)
    test_metrics = compute_metrics(test_spread, test_pos, test_trades)

    # Print metrics
    print("Pairs backtest (beta from train only)")
    print("=" * 60)
    print(f"{'Metric':<20} {'Train':>15} {'Test':>15}")
    print("-" * 60)
    print(f"{'Total PnL':<20} {train_metrics['total_pnl']:>15.2f} {test_metrics['total_pnl']:>15.2f}")
    print(f"{'Sharpe':<20} {train_metrics['sharpe']:>15.2f} {test_metrics['sharpe']:>15.2f}")
    print(f"{'Max drawdown':<20} {train_metrics['max_drawdown']:>15.2f} {test_metrics['max_drawdown']:>15.2f}")
    print(f"{'Win rate':<20} {train_metrics['win_rate']*100:>14.1f}% {test_metrics['win_rate']*100:>14.1f}%")
    print(f"{'Avg duration':<20} {train_metrics['avg_duration']:>15.1f} {test_metrics['avg_duration']:>15.1f}")
    print(f"{'# trades':<20} {train_metrics['n_trades']:>15} {test_metrics['n_trades']:>15}")

    # Plots
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: spread + z-score
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    train_ts = train["ts"].values
    test_ts = test["ts"].values

    ax1.plot(train_ts, train_spread, label="Train spread", alpha=0.8)
    ax1.plot(test_ts, test_spread, label="Test spread", alpha=0.8)
    ax1.set_ylabel("Spread (NVDA - β×AMD)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_ts, train_z, label="Train z", alpha=0.8)
    ax2.plot(test_ts, test_z, label="Test z", alpha=0.8)
    ax2.axhline(Z_ENTRY_SHORT, color="r", linestyle="--", alpha=0.5)
    ax2.axhline(Z_ENTRY_LONG, color="g", linestyle="--", alpha=0.5)
    ax2.axhline(Z_EXIT, color="k", linestyle="-", alpha=0.3)
    ax2.set_ylabel("Z-score")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Pairs: Spread and Z-Score")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "pairs_spread_zscore.png", dpi=150)
    plt.close(fig)

    # Plot 2: equity curve
    train_pnl = np.zeros(len(train_spread))
    for i in range(1, len(train_spread)):
        train_pnl[i] = train_pos[i - 1] * (train_spread[i] - train_spread[i - 1])
    test_pnl = np.zeros(len(test_spread))
    for i in range(1, len(test_spread)):
        test_pnl[i] = test_pos[i - 1] * (test_spread[i] - test_spread[i - 1])

    train_equity = np.cumsum(train_pnl)
    test_equity = np.cumsum(test_pnl)

    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_ts, train_equity, label="Train equity", alpha=0.8)
    ax.plot(test_ts, test_equity, label="Test equity", alpha=0.8)
    ax.set_ylabel("Cumulative PnL")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Pairs: Equity Curve")
    fig2.tight_layout()
    fig2.savefig(OUTPUTS_DIR / "pairs_equity_curve.png", dpi=150)
    plt.close(fig2)

    print(f"\nPlots saved to {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
