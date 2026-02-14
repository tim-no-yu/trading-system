"""
Pairs regression: OLS of NVDA on AMD, ADF on residuals, half-life estimate.
Loads cleaned CSVs, aligns by timestamp, 70/30 train/test split, reports
beta, ADF stat, p-value, and residual half-life.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_FRAC = 0.70


def _find_csv(symbol: str) -> Path:
    """Find NVDA or AMD clean CSV (1Day, 1Hour, or 1Min)."""
    for tf in ("1Day", "1Hour", "1Min"):
        p = DATA_DIR / f"{symbol}_{tf}_stock_alpaca_clean.csv"
        if p.exists():
            return p
    raise FileNotFoundError(f"No clean CSV found for {symbol} in {DATA_DIR}")


def load_and_align() -> pd.DataFrame:
    """Load NVDA and AMD CSVs, align by timestamp."""
    nvda_path = _find_csv("NVDA")
    amd_path = _find_csv("AMD")

    nvda = pd.read_csv(nvda_path)
    amd = pd.read_csv(amd_path)

    # Normalize timestamp column (Datetime or timestamp)
    for df in (nvda, amd):
        if "Datetime" in df.columns:
            df["ts"] = pd.to_datetime(df["Datetime"], utc=True)
        elif "timestamp" in df.columns:
            df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
        else:
            raise ValueError("CSV must have Datetime or timestamp column")

    nvda = nvda[["ts", "Close"]].rename(columns={"Close": "NVDA"})
    amd = amd[["ts", "Close"]].rename(columns={"Close": "AMD"})

    merged = pd.merge(nvda, amd, on="ts", how="inner")
    merged = merged.dropna().sort_values("ts").reset_index(drop=True)
    return merged


def train_test_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC):
    """Chronological 70/30 split."""
    n = len(df)
    split_idx = int(n * train_frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def ols_residuals(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """OLS of y on x. Returns (alpha, beta, residuals)."""
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    alpha, beta = model.params[0], model.params[1]
    residuals = y - (alpha + beta * x)
    return alpha, beta, residuals


def half_life(residuals: np.ndarray) -> float | None:
    """
    Estimate half-life of mean reversion from AR(1): spread_t = phi * spread_{t-1} + c.
    Half-life = -log(2) / log(phi) in periods.
    """
    lag = residuals[:-1]
    diff = np.diff(residuals)
    # Regress diff on lag: diff_t = theta * (mu - lag) ≈ -theta * lag + const
    # Or: residuals_t = phi * residuals_{t-1} => phi from regression of res_t on res_{t-1}
    x = sm.add_constant(lag)
    model = sm.OLS(residuals[1:], x).fit()
    phi = model.params[1]
    if phi <= 0 or phi >= 1:
        return None
    return -math.log(2) / math.log(phi)


def main() -> None:
    df = load_and_align()
    train, _ = train_test_split(df, TRAIN_FRAC)

    y = train["NVDA"].values
    x = train["AMD"].values

    alpha, beta, residuals = ols_residuals(y, x)

    adf_result = adfuller(residuals, autolag="AIC", maxlag=None)
    adf_stat, pvalue = adf_result[0], adf_result[1]

    hl = half_life(residuals)

    print("Pairs regression (NVDA ~ AMD, train set)")
    print("-" * 50)
    print(f"  beta:        {beta:.6f}")
    print(f"  ADF stat:    {adf_stat:.4f}")
    print(f"  p-value:     {pvalue:.6f}")
    hl_str = f"{hl:.2f} periods" if hl is not None else "N/A (phi not in (0,1))"
    print(f"  half-life:   {hl_str}")


if __name__ == "__main__":
    main()
