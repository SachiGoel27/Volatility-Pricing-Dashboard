"""
ms_garch_model.py
Two-stage Markov-Switching (HMM) + GARCH hybrid implementation.

Main function:
- fit_ms_garch_model(df, n_regimes=2, min_obs_per_regime=100, scale_returns=True)

Inputs:
- df: DataFrame with 'returns' column (decimal returns, e.g. 0.01)
- n_regimes: number of hidden regimes
- min_obs_per_regime: minimum observations to fit GARCH in a regime
- scale_returns: scale returns by 100 for arch package (percent) — recommended

Outputs: dictionary containing:
- 'hmm' : fitted HMM object
- 'regime_probs' : posterior probabilities (T x n_regimes)
- 'regimes' : hard regime assignment (T,)
- 'garch_fits' : dict of fitted arch results per regime (res objects)
- 'forecasts' : dict of one-step-ahead sigma per regime (decimal)
- 'mixture_forecast' : weighted mixture sigma (decimal)
- 'transition_matrix' : HMM transition matrix
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from arch import arch_model


def fit_hmm_on_series(returns: np.ndarray, n_regimes: int = 2, n_iter: int = 200, random_state: int = 42):
    """
    Fit Gaussian HMM on 1-D returns series (shape (T,1)).
    Returns fitted model and posterior probabilities (gamma: T x n_regimes)
    """
    model = GaussianHMM(n_components=n_regimes, covariance_type="diag", n_iter=n_iter, random_state=random_state)
    model.fit(returns)
    gamma = model.predict_proba(returns)
    return model, gamma


def fit_garch_for_regime(returns: pd.Series, p: int = 1, q: int = 1, scale: bool = True):
    """
    Fit a GARCH(p,q) on returns for a given regime.
    Note: arch arch_model commonly expects returns in percent (not decimal), so scaling by 100 helps.
    We recommend mean='Zero' for returns that are demeaned; modify as needed.
    Returns the fitted result object.
    """
    if scale:
        train_ret = returns * 100.0  # percent
    else:
        train_ret = returns
    am = arch_model(train_ret, mean="Zero", vol="GARCH", p=p, q=q, dist="normal")
    res = am.fit(disp="off")
    return res


def fit_ms_garch_model(
    df: pd.DataFrame,
    n_regimes: int = 2,
    min_obs_per_regime: int = 100,
    p: int = 1,
    q: int = 1,
    scale_returns: bool = True,
) -> Dict[str, Any]:
    """
    Two-stage MS + GARCH fit.

    Steps:
    1) Fit HMM on returns (1-d)
    2) Get posterior probs and hard regime labels
    3) For each regime, fit GARCH on observations assigned to that regime (hard assignment)
    4) Forecast one-step-ahead conditional variance per regime
    5) Combine forecasts using latest posterior or one-step-ahead regime distribution

    Returns results dict (see docstring).
    """
    # Validate input
    if "returns" not in df.columns:
        raise ValueError("Input DataFrame must have 'returns' column (decimal returns)")

    returns = df["returns"].dropna().values.reshape(-1, 1)  # shape (T,1)
    dates = df.index
    print(df['returns'].head(10))
    print(df['returns'].std(), df['returns'].mean())

    if len(returns) < 30:
        print(f"[Warning] Not enough data to fit MS-GARCH (len={len(returns)})")
        return {
            "posterior": pd.DataFrame(index=df.index, data={"p_regime_1": np.nan, "p_regime_2": np.nan}),
            "mixture_forecast": np.full(len(df), np.nan),
        }

    # 1) Fit HMM
    hmm, gamma = fit_hmm_on_series(returns, n_regimes=n_regimes)
    posterior = pd.DataFrame(gamma, index=dates, columns=[f"regime_{i}" for i in range(n_regimes)])

    # 2) Hard assign regimes
    hard_regimes = posterior.values.argmax(axis=1)
    regime_series = pd.Series(hard_regimes, index=dates, name="regime")

    # 3) Fit GARCH per regime (hard assignment)
    garch_fits = {}
    forecasts = {}
    for r in range(n_regimes):
        mask = hard_regimes == r
        if mask.sum() < min_obs_per_regime:
            # skip regime if too few obs
            continue
        # returns as pandas Series for arch
        rs = pd.Series(df["returns"].values[mask], index=dates[mask])
        try:
            res = fit_garch_for_regime(rs, p=p, q=q, scale=scale_returns)
            garch_fits[r] = res

            # 4) one-step-ahead forecast (variance)
            f = res.forecast(horizon=1, reindex=False)
            # res.forecast(...).variance is DataFrame with shape (#observations, horizon)
            var_1 = f.variance.values[-1, 0]  # if we scaled by 100, var is in (percent^2)
            if scale_returns:
                sigma = np.sqrt(var_1) / 100.0  # back to decimal
            else:
                sigma = np.sqrt(var_1)
            forecasts[r] = float(sigma)
        except Exception as e:
            # if GARCH fails for that regime, skip
            print(f"Failed to fit GARCH for regime {r}: {e}")
            continue

    # 5) Mixture forecast: we can use today's posterior (gamma[-1]) or the one-step-ahead regime dist
    last_posterior = gamma[-1]  # shape (n_regimes,)
    # Optionally use transition matrix to project one step ahead: pi_next = last_posterior @ P
    try:
        P = hmm.transmat_
        pi_next = last_posterior @ P
    except Exception:
        pi_next = last_posterior

    # Use pi_next if you want one-step-ahead regime distribution; else use last_posterior
    weights = pi_next

    # Build mixture (only include regimes we have forecast for)
    mix_num = 0.0
    weight_sum = 0.0
    for r, sigma_r in forecasts.items():
        w = float(weights[r]) if r < len(weights) else 0.0
        mix_num += w * sigma_r
        weight_sum += w

    mixture_forecast = float(mix_num / weight_sum) if weight_sum > 0 else None

    for r,res in garch_fits.items():
        print("Regime", r)
        print(res.params)                # show omega/alpha/beta
        f = res.forecast(horizon=1, reindex=False)
        print("forecast variance shape", getattr(f, "variance", None).shape)
        var1 = f.variance.values[-1,0]
        print("raw var1:", var1)
        sigma = (np.sqrt(var1) / 100.0)  # if you scaled by 100 for fit
        print("sigma (decimal):", sigma)
    print(forecasts)

    forecasts_annualized = {r: sigma * np.sqrt(252) for r, sigma in forecasts.items()}
    mixture_forecast_annualized = mixture_forecast * np.sqrt(252) if mixture_forecast else None     
    results = {
        "hmm": hmm,
        "posterior": posterior,  # DataFrame (T x n_regimes)
        "regimes": regime_series,  # hard assignment
        "garch_fits": garch_fits,
        "forecasts": forecasts_annualized,     # dict regime -> sigma (decimal)
        "mixture_forecast": mixture_forecast_annualized,
        "transition_matrix": getattr(hmm, "transmat_", None),
    }
    return results