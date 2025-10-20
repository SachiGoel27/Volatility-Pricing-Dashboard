import yfinance as yf
import datetime
from typing import Dict, Optional
import pandas as pd

def get_implied_vol_series(ticker: str, period: str = "6mo", max_days: int = 250) -> Dict[str, float]:
    """
    Return a simple time series of daily implied volatilities for the given ticker.
    This matches the last `max_days` from historical price data.

    Returns:
        dict: {date_str: implied_vol_decimal}, e.g. {'2025-10-01': 0.25}
    """
    try:
        # 1) Get historical price data
        df_price = yf.download(ticker, period=period, interval="1d", progress=False)
        if df_price.empty:
            return {}

        df_price = df_price.tail(max_days)
        dates = df_price.index

        # 2) Get nearest option expiration
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return {}
        selected_exp = expirations[0]

        option_chain = tk.option_chain(selected_exp)
        calls = option_chain.calls
        puts = option_chain.puts

        if calls.empty and puts.empty:
            return {}

        # 3) Take average IV for the expiration as a simple proxy
        # You can improve by interpolating or matching maturities
        # print(calls.columns)
        iv_avg = pd.concat([calls['impliedVolatility'], puts['impliedVolatility']]).mean()
        implied_vol_series = {date.strftime("%Y-%m-%d"): float(iv_avg) for date in dates}

        return implied_vol_series

    except Exception as e:
        print(f"[get_implied_vol_series] Failed for {ticker}: {e}")
        return {}
