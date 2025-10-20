"""
realized_vol.py
Functions to compute realized volatility from price data.

Functions:
- fetch_price_data(ticker, period='2y', interval='1d')
- compute_daily_returns(df) -> df with 'returns'
- realized_vol_rolling(df, window=21) -> df with 'RV' (annualized)
- realized_vol_from_intraday(df_intraday, freq='5min') -> daily RV from intraday (if you have intraday)
"""

from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf


def fetch_price_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Download price data using yfinance (returns dataframe with DatetimeIndex).
    interval examples: '1d', '1h', '30m', '5m' (5m requires more permissions and may be limited)
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} with period={period} interval={interval}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]  # take only the first level ('Close', 'Open', etc.)
    return df


def compute_daily_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Compute log returns on price_col.
    Returns a copy with a column 'returns' (in decimal, e.g. 0.01 == 1%).
    """
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    df['returns'] = df[price_col].pct_change()
    df = df.dropna(subset=['returns'])
    return df


def realized_vol_rolling(df: pd.DataFrame, window: int = 21, price_col: str = "Close") -> pd.DataFrame:
    """
    Compute rolling realized volatility:
    RV_t = sqrt(sum_{i=t-window+1..t} r_i^2)  (sampled over window days)
    We annualize by sqrt(252) if the data is daily.

    Parameters:
    - df: must contain 'returns' (use compute_daily_returns)
    - window: number of days in rolling window (typical: 21 trading days ~ 1 month)
    Returns df with a new column 'RV' (annualized).
    """
    df = df.copy()
    if "returns" not in df.columns:
        df = compute_daily_returns(df, price_col=price_col)

    # realized variance over window (sum of squared returns)
    df["rv_squared"] = df["returns"] ** 2
    df["rv_roll_sum"] = df["rv_squared"].rolling(window=window).sum()

    # annualize: sqrt(sum) * sqrt(trading_days_per_year / window)
    # Another equivalent: sqrt( (252/window) * sum(returns^2) )
    # We'll compute annualized RV (as standard deviation)
    trading_days = 252
    df["RV"] = np.sqrt((trading_days / window) * df["rv_roll_sum"])
    df = df.dropna(subset=["RV"])
    # Clean helper columns optionally left for debugging
    df = df.drop(columns=["rv_squared", "rv_roll_sum"])
    return df


def realized_vol_from_intraday(df_intraday: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    """
    Given intraday returns (timestamp-indexed, with 'price' column), compute daily realized variance:
    RV_day = sum_{i in day} r_{i}^2   (where r_i are intraday log returns)
    and annualize if desired.

    Input df_intraday should have DatetimeIndex or a column convertible to Datetime.
    Returns DataFrame indexed by date with columns: 'rv_daily' (not annualized) and 'RV' (annualized)
    """
    df = df_intraday.copy()
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Intraday df must have a DatetimeIndex")

    # compute log returns from mid/price column (assume column 'Close' or 'price' present)
    price_cols = [c for c in df.columns if c.lower() in ("close", "price")]
    if price_cols:
        pcol = price_cols[0]
    else:
        raise ValueError("No price column found (expect 'Close' or 'price') in intraday df")

    df["intraday_ret"] = np.log(df[pcol] / df[pcol].shift(1))
    df = df.dropna(subset=["intraday_ret"])
    # group by date and sum squared intraday returns -> daily realized variance
    rv_daily = df["intraday_ret"].pow(2).groupby(df.index.date).sum()
    rv_daily.index = pd.to_datetime(rv_daily.index)
    rv_df = pd.DataFrame({"rv_daily": rv_daily})
    # annualize assuming trading_days per year
    trading_days = 252
    rv_df["RV"] = np.sqrt(rv_df["rv_daily"] * trading_days)
    return rv_df
