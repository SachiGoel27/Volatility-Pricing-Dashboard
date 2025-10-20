from django.shortcuts import render
import yfinance as yf
import plotly.graph_objs as go
import plotly.offline as opy
from py_vollib.black_scholes.implied_volatility import implied_volatility as bs_iv
import datetime
from .analytics.realized_vol import fetch_price_data, compute_daily_returns, realized_vol_rolling
from .analytics.ms_garch_model import fit_ms_garch_model
from .analytics.implied_vol import get_implied_vol_series
import pandas as pd
import numpy as np

def page_b_view(request):
    ticker = request.GET.get("ticker", "AAPL")
    period = request.GET.get("period", "6mo")
    
    # Initialize variables
    rv_series = {}
    ms_vol_series = {}
    implied_vol_series = {}
    forecast_vol = None
    rv_div = ms_vol_div = overlay_div = None
    error = None

    try:
        # 1) Fetch price and realized volatility
        df = fetch_price_data(ticker, period=period)
        df = compute_daily_returns(df)
        df = realized_vol_rolling(df, window=21)

        # 2) Fit MS-GARCH
        ms_results = fit_ms_garch_model(df, n_regimes=2)

        # 3) Build MS-GARCH volatility time series
        posterior = ms_results["posterior"]
        forecasts = ms_results["forecasts"]
        ms_vol_list = []
        for idx, row in posterior.iterrows():
            vol = 0.0
            weight_sum = 0.0
            for r, sigma_r in forecasts.items():
                if r < len(row):
                    w = row.iloc[r]
                    vol += w * sigma_r
                    weight_sum += w
            ms_vol_list.append(vol / weight_sum if weight_sum > 0 else None)
        
        ms_vol_df = pd.Series(ms_vol_list, index=posterior.index).tail(250)
        ms_vol_series = {date.strftime("%Y-%m-%d"): float(val) if pd.notna(val) else None 
                         for date, val in ms_vol_df.items()}

        # 4) Realized volatility
        rv_df = df["RV"].tail(250)
        rv_series = {date.strftime("%Y-%m-%d"): float(val) if pd.notna(val) else None
                     for date, val in rv_df.items()}

        # 5) Implied volatility
        implied_vol_series_full = get_implied_vol_series(ticker, period=period)
        dates_str = list(rv_series.keys())
        implied_vol_series = {d: implied_vol_series_full.get(d, None) for d in dates_str}

        # 6) Forecast
        forecast_vol = float(ms_results.get("mixture_forecast", None) * np.sqrt(252))

        # --- Plotly charts ---
        # Realized Vol
        rv_trace = go.Scatter(
            x=list(rv_series.keys()), 
            y=list(rv_series.values()), 
            mode='lines', 
            name='Realized Volatility', 
            line=dict(color='blue')
        )
        rv_fig = go.Figure(data=[rv_trace])
        rv_fig.update_layout(title='Realized Volatility', xaxis_title='Date', yaxis_title='Volatility')
        rv_div = opy.plot(rv_fig, auto_open=False, output_type='div')

        # MS-GARCH
        ms_trace = go.Scatter(
            x=list(ms_vol_series.keys()), 
            y=list(ms_vol_series.values()), 
            mode='lines', 
            name='MS-GARCH Volatility', 
            line=dict(color='red')
        )
        ms_fig = go.Figure(data=[ms_trace])
        ms_fig.update_layout(title='MS-GARCH Volatility', xaxis_title='Date', yaxis_title='Volatility')
        ms_vol_div = opy.plot(ms_fig, auto_open=False, output_type='div')

        # Overlay
        iv_trace = go.Scatter(
            x=list(implied_vol_series.keys()), 
            y=list(implied_vol_series.values()), 
            mode='lines', 
            name='Implied Volatility', 
            line=dict(color='green')
        )
        ms_trace_overlay = go.Scatter(
            x=list(ms_vol_series.keys()), 
            y=list(ms_vol_series.values()), 
            mode='lines', 
            name='MS-GARCH Forecast', 
            line=dict(color='red')
        )
        overlay_fig = go.Figure(data=[iv_trace, ms_trace_overlay])
        overlay_fig.update_layout(title='Overlay: Implied vs MS-GARCH', xaxis_title='Date', yaxis_title='Volatility')
        overlay_div = opy.plot(overlay_fig, auto_open=False, output_type='div')

        print(f"Successfully processed {ticker}")
        print(f"RV series length: {len(rv_series)}")
        print(f"MS-GARCH series length: {len(ms_vol_series)}")
        print(f"Implied Vol series length: {len(implied_vol_series)}")

    except Exception as e:
        error = str(e)
        print(f"Error in page_b_view: {e}")
        import traceback
        traceback.print_exc()

    context = {
        "ticker": ticker,
        "period": period,
        "rv_div": rv_div,
        "ms_vol_div": ms_vol_div,
        "overlay_div": overlay_div,
        "forecast_vol": forecast_vol,
        "error": error,
    }

    return render(request, "dashboard/page_b.html", context)
def page_c_view(request):
    context = {
        'title': 'Realized vs Implied Volatility Analysis',
        'description': (
            "This page will show realized volatility computed from historical "
            "returns and compare it against implied volatility surfaces over time."
        ),
    }
    return render(request, "dashboard/page_c.html", context)

def dashboard_view(request):
    ticker = request.GET.get('ticker', 'AAPL')
    period = request.GET.get('period', '6mo')  # default to 6 months
    selected_exp = request.GET.get('expiration', None)

    # Define the available timeframes
    timeframes = {
        "1M": "1mo",
        "6M": "6mo",
        "1Y": "1y",
        "5Y": "5y",
        "All": "max",
    }

    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)

    # Price chart
    price_trace = go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Price')
    price_fig = go.Figure(data=[price_trace])
    price_fig.update_layout(title=f"{ticker} Price History ({period})",
                            xaxis_title='Date',
                            yaxis_title='Price ($)')
    price_div = opy.plot(price_fig, auto_open=False, output_type='div')

    iv_div = "<p>No options data available.</p>"
    expirations = []

    try:
        expirations = stock.options
        if not expirations:
            raise ValueError("No options available")

        if not selected_exp or selected_exp not in expirations:
            selected_exp = expirations[0]

        option_chain = stock.option_chain(selected_exp)
        calls = option_chain.calls
        puts = option_chain.puts

        S = hist['Close'][-1]
        r = 0.03
        today = datetime.datetime.today()

        # Calls
        call_strikes, call_iv = [], []
        for _, row in calls.iterrows():
            K = row['strike']
            market_price = row['lastPrice']
            T = (datetime.datetime.strptime(selected_exp, "%Y-%m-%d") - today).days / 365
            try:
                if market_price > 0 and T > 0:
                    iv = bs_iv(market_price, S, K, T, r, 'c')
                    call_strikes.append(K)
                    call_iv.append(iv)
            except:
                continue

        # Puts
        put_strikes, put_iv = [], []
        for _, row in puts.iterrows():
            K = row['strike']
            market_price = row['lastPrice']
            T = (datetime.datetime.strptime(selected_exp, "%Y-%m-%d") - today).days / 365
            try:
                if market_price > 0 and T > 0:
                    iv = bs_iv(market_price, S, K, T, r, 'p')
                    put_strikes.append(K)
                    put_iv.append(iv)
            except:
                continue

        # IV Plot
        if call_strikes or put_strikes:
            call_trace = go.Scatter(x=call_strikes, y=call_iv, mode='markers+lines', name='Calls')
            put_trace = go.Scatter(x=put_strikes, y=put_iv, mode='markers+lines', name='Puts')
            atm_line = go.Scatter(x=[S, S], y=[0, max(call_iv + put_iv, default=0)], 
                                  mode='lines', name='ATM', line=dict(dash='dot'))

            iv_fig = go.Figure(data=[call_trace, put_trace, atm_line])
            iv_fig.update_layout(title=f"Implied Volatility Curve ({selected_exp})",
                                 xaxis_title='Strike Price',
                                 yaxis_title='Implied Volatility')
            iv_div = opy.plot(iv_fig, auto_open=False, output_type='div')

    except Exception as e:
        print(f"IV calculation failed: {e}")

    context = {
        'ticker': ticker,
        'period': period,
        'price_div': price_div,
        'iv_div': iv_div,
        'expirations': expirations,
        'selected_exp': selected_exp,
        'timeframes': timeframes,
    }
    return render(request, 'dashboard/dashboard.html', context)
