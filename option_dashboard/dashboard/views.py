from django.shortcuts import render
import yfinance as yf
import plotly.graph_objs as go
import plotly.offline as opy
from py_vollib.black_scholes.implied_volatility import implied_volatility as bs_iv
import datetime
# from .models import OptionData
def page_b_view(request):
    context = {
        'title': 'Markov + GARCH Model',
        'description': (
            "This page will display the regime-switching volatility model "
            "with GARCH dynamics. Future functionality may include parameter "
            "inputs, regime plots, and simulated volatility paths."
        ),
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
