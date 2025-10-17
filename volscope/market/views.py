from django.shortcuts import render
from django.shortcuts import render
from .services.data_fetcher import get_price_data, get_option_chain
import plotly.graph_objs as go
import plotly.offline as opy

def dashboard(request):
    ticker = request.GET.get("ticker", "AAPL")
    price_data = get_price_data(ticker)
    option_chain = get_option_chain(ticker)

    # Plot price graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_data.index, y=price_data['Close'], mode='lines', name='Price'))
    fig.update_layout(title=f"{ticker} Price History", xaxis_title='Date', yaxis_title='Price')
    price_plot = opy.plot(fig, auto_open=False, output_type='div')

    # Plot IV distribution
    if option_chain:
        ivs = [x['impliedVolatility'] for x in option_chain['calls']]
        strikes = [x['strike'] for x in option_chain['calls']]
        iv_fig = go.Figure()
        iv_fig.add_trace(go.Scatter(x=strikes, y=ivs, mode='lines+markers', name='Implied Vol'))
        iv_fig.update_layout(title=f"{ticker} Implied Volatility Curve", xaxis_title='Strike', yaxis_title='IV')
        iv_plot = opy.plot(iv_fig, auto_open=False, output_type='div')
    else:
        iv_plot = "<p>No option data available</p>"

    return render(request, 'market/dashboard.html', {
        "ticker": ticker,
        "price_plot": price_plot,
        "iv_plot": iv_plot
    })

