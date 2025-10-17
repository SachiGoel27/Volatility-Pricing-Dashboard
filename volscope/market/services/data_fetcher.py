import yfinance as yf
import pandas as pd

def get_price_data(ticker: str):
    """Fetch 6-month daily price data for a given ticker"""
    data = yf.download(ticker, period="6mo", interval="1d")
    return data[['Close']]

def get_option_chain(ticker: str):
    """Fetch option chain with implied volatilities"""
    stock = yf.Ticker(ticker)
    expirations = stock.options
    chain_data = []

    if expirations:
        first_exp = expirations[0]  # only fetch near-term for now
        opt_chain = stock.option_chain(first_exp)
        calls = opt_chain.calls[['strike', 'lastPrice', 'impliedVolatility']]
        puts = opt_chain.puts[['strike', 'lastPrice', 'impliedVolatility']]
        chain_data = {
            "expiration": first_exp,
            "calls": calls.to_dict(orient="records"),
            "puts": puts.to_dict(orient="records")
        }
    return chain_data
