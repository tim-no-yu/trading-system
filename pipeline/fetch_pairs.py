"""
Fetch NVDA and AMD stock bars from Alpaca (1Day preferred, 1Hour fallback)
for the last 2–5 years, clean to simple CSVs, and save to data/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on path when run as script
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from pipeline.alpaca import (
    DATA_DIR,
    _normalize_bars,
    _parse_timeframe,
    _to_rfc3339,
    get_rest,
)

SYMBOLS = ["NVDA", "AMD"]
PREFERRED_TIMEFRAME = "1Day"
FALLBACK_TIMEFRAME = "1Hour"
YEARS_BACK = 5
BARS_PER_PAGE = 10000


def _fetch_bars_paginated(
    api,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    feed: str,
) -> pd.DataFrame:
    """Fetch all bars in [start, end] with pagination."""
    tf = _parse_timeframe(timeframe)
    all_dfs: list[pd.DataFrame] = []
    current_start = start

    while current_start < end:
        bars = api.get_bars(
            symbol,
            tf,
            start=_to_rfc3339(current_start),
            end=_to_rfc3339(end),
            limit=BARS_PER_PAGE,
            feed=feed,
        )
        raw = bars.df if hasattr(bars, "df") else bars
        if raw is None or (hasattr(raw, "empty") and raw.empty):
            break
        df = _normalize_bars(raw, symbol)
        if df.empty:
            break
        all_dfs.append(df)
        if len(df) < BARS_PER_PAGE:
            break
        last_ts = df["Datetime"].max()
        # Advance by one bar to avoid overlap
        if "Day" in timeframe or "D" in timeframe:
            current_start = last_ts + pd.Timedelta(days=1)
        elif "Hour" in timeframe or "H" in timeframe:
            current_start = last_ts + pd.Timedelta(hours=1)
        else:
            current_start = last_ts + pd.Timedelta(minutes=1)

    if not all_dfs:
        return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])

    out = pd.concat(all_dfs, ignore_index=True)
    out.drop_duplicates(subset=["Datetime"], inplace=True)
    out.sort_values("Datetime", inplace=True)
    return out


def _clean_to_simple_csv(
    df: pd.DataFrame,
    out_path: Path,
    include_ohlcv: bool = True,
) -> None:
    """Write cleaned CSV with Datetime, Close, and optionally OHLCV."""
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df.dropna(subset=["Datetime"], inplace=True)
    df.drop_duplicates(subset=["Datetime"], inplace=True)
    df.sort_values("Datetime", inplace=True)

    if include_ohlcv:
        cols = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    else:
        cols = ["Datetime", "Close"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out_path, index=False)


def fetch_pairs(
    years_back: int = YEARS_BACK,
    preferred: str = PREFERRED_TIMEFRAME,
    fallback: str = FALLBACK_TIMEFRAME,
    symbols: list[str] | None = None,
    feed: str | None = None,
) -> dict[str, Path]:
    """
    Fetch NVDA and AMD bars, clean, and save to data/.

    Returns dict mapping symbol -> output path.
    """
    symbols = symbols or SYMBOLS
    feed = feed or os.environ.get("ALPACA_DATA_FEED", "iex")
    api = get_rest()

    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=years_back * 365)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    for symbol in symbols:
        symbol = symbol.upper()
        df = None
        used_timeframe = None

        for tf in [preferred, fallback]:
            try:
                df = _fetch_bars_paginated(api, symbol, tf, start, end, feed)
                if df is not None and not df.empty:
                    used_timeframe = tf
                    break
            except Exception as e:
                print(f"  {symbol} {tf}: {e}")
                continue

        if df is None or df.empty:
            print(f"  {symbol}: No data for {preferred} or {fallback}")
            continue

        out_path = DATA_DIR / f"{symbol}_{used_timeframe}_stock_alpaca_clean.csv"
        _clean_to_simple_csv(df, out_path, include_ohlcv=True)
        results[symbol] = out_path
        print(f"  {symbol}: {len(df)} bars -> {out_path}")

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch NVDA and AMD bars from Alpaca (1Day/1Hour), clean, save to data/"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=YEARS_BACK,
        help=f"Years of history (default: {YEARS_BACK})",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=SYMBOLS,
        help="Symbols to fetch (default: NVDA AMD)",
    )
    args = parser.parse_args()

    print("Fetching NVDA and AMD (1Day preferred, 1Hour fallback)...")
    results = fetch_pairs(years_back=args.years, symbols=args.symbols)
    print(f"Done. Saved {len(results)} files.")


if __name__ == "__main__":
    main()
