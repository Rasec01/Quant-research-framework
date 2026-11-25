"""
Factor-tilt portfolios, rebalanced monthly and chosen walk-forward.

A tilt portfolio holds the market and adds long-short factor positions:

    return = RF + (Mkt-RF) + sum_k w_k * factor_k - costs

Weights for month t use only factor returns up to month t-1:

* "equal":  the same tilt to every factor, whatever the history says
* "mv":     long-only tilts in proportion to the maximum-Sharpe mix of the
            factors over a trailing window (shrunk towards equal tilts)

Costs: each unit of tilt turnover costs `trade_cost`; holding long-short
factor positions costs `carry_cost` per year per unit of gross tilt, a rough
allowance for the trading and shorting that real factor portfolios need.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.data import FACTORS


def _mv_tilts(window: pd.DataFrame, total: float, shrink: float) -> np.ndarray:
    mu, cov = window.mean().to_numpy(), window.cov().to_numpy()
    k = len(mu)
    neg_sharpe = lambda w: -(w @ mu) / np.sqrt(w @ cov @ w)
    res = minimize(neg_sharpe, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    w = res.x if res.success and np.isfinite(res.fun) else np.full(k, 1 / k)
    w = (1 - shrink) * np.clip(w, 0, None) + shrink / k
    return total * w / w.sum()


def tilt_weights(ff: pd.DataFrame, method: str = "equal", total_tilt: float = 0.5,
                 window: int = 120, shrink: float = 0.5, factors=FACTORS) -> pd.DataFrame:
    """Weights for each month, decided with data that ends the month before."""
    w = pd.DataFrame(np.nan, index=ff.index, columns=factors)
    for i in range(window, len(ff)):
        if method == "equal":
            w.iloc[i] = total_tilt / len(factors)
        elif method == "mv":
            w.iloc[i] = _mv_tilts(ff[factors].iloc[i - window:i], total_tilt, shrink)
        else:
            raise ValueError("method must be 'equal' or 'mv'")
    return w.dropna()


def run(ff: pd.DataFrame, weights: pd.DataFrame, trade_cost: float = 0.001,
        carry_cost: float = 0.005) -> pd.DataFrame:
    """Monthly returns of the tilt portfolio and of the market."""
    f = ff.loc[weights.index]
    market = f["RF"] + f["Mkt-RF"]
    tilt = (weights * f[weights.columns]).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.iloc[0].abs().sum())
    costs = turnover * trade_cost + weights.abs().sum(axis=1) * carry_cost / 12
    return pd.DataFrame({"strategy": market + tilt - costs, "market": market,
                         "rf": f["RF"], "turnover": turnover})


def performance(monthly: pd.Series, rf: pd.Series) -> dict:
    wealth = (1 + monthly).cumprod()
    years = len(monthly) / 12
    excess = monthly - rf
    return {
        "annual_return": float(wealth.iloc[-1] ** (1 / years) - 1),
        "annual_volatility": float(monthly.std() * np.sqrt(12)),
        "sharpe": float(excess.mean() / excess.std() * np.sqrt(12)),
        "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
    }


def summary(res: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame({"Factor tilt": performance(res["strategy"], res["rf"]),
                      "Market": performance(res["market"], res["rf"])})
    active = res["strategy"] - res["market"]
    t.loc["tracking_error", "Factor tilt"] = active.std() * np.sqrt(12)
    t.loc["information_ratio", "Factor tilt"] = active.mean() / active.std() * np.sqrt(12)
    t.loc["turnover_per_year", "Factor tilt"] = res["turnover"].mean() * 12
    return t
