"""
plot_utils.py
Utility functions for visualizing volatility and model regimes.

Functions:
- plot_realized_vol(df)
- plot_regimes(df, posterior, regimes)
- plot_iv_vs_forecast(df_iv, forecast_vol)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_realized_vol(df: pd.DataFrame):
    """
    Plot realized volatility over time.
    Expects df with column 'RV' (annualized).
    """
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["RV"], label="Realized Volatility", color="blue", linewidth=1.5)
    plt.title("Rolling Realized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_regimes(df: pd.DataFrame, posterior: pd.DataFrame, regimes: pd.Series):
    """
    Overlay the detected regimes from the HMM on the realized volatility.
    Inputs:
    - df: DataFrame with 'RV'
    - posterior: DataFrame with regime probabilities (from HMM)
    - regimes: Series of 0/1 labels per date
    """
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["RV"], color="black", linewidth=1.2, label="Realized Volatility")

    # Highlight regime 1 (high-vol regime) areas
    plt.fill_between(df.index, 0, df["RV"],
                     where=regimes == 1,
                     color="red", alpha=0.3,
                     label="Regime 1 (High Vol)")

    plt.title("Markov-Switching Regimes on Realized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_iv_vs_forecast(df_iv: pd.DataFrame, forecast_vol: float):
    """
    Compare implied volatility curve (from options) to forecasted volatility (from MS-GARCH).
    Expects:
    - df_iv: DataFrame with columns ['Maturity', 'ImpliedVol'] (annualized, decimal form)
    - forecast_vol: scalar, MS-GARCH mixture forecast (annualized)
    """
    plt.figure(figsize=(8, 5))
    plt.plot(df_iv["Maturity"], df_iv["ImpliedVol"], marker="o", label="Implied Vol Curve")
    plt.axhline(y=forecast_vol, color="orange", linestyle="--", label="MS-GARCH Forecast Vol")

    plt.title("Implied Volatility Curve vs MS-GARCH Forecast")
    plt.xlabel("Time to Maturity (Days)")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
