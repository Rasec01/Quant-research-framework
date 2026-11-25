"""Time-series factor regression with Newey-West standard errors."""
import numpy as np
import pandas as pd


def factor_regression(excess_returns: pd.Series, factors: pd.DataFrame, lags: int = 6) -> pd.DataFrame:
    """
    Regress excess returns on factors: r - rf = alpha + sum(beta_k * f_k) + e.
    Returns coefficients, Newey-West t-statistics, and alpha annualised (x12).
    """
    data = pd.concat([excess_returns.rename("y"), factors], axis=1, join="inner").dropna()
    y = data["y"].to_numpy()
    X = np.column_stack([np.ones(len(data)), data[factors.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n, k = X.shape

    # Newey-West (Bartlett kernel) covariance of the coefficients
    XtX_inv = np.linalg.inv(X.T @ X)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        G = Xe[lag:].T @ Xe[:-lag]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv * n / (n - k)
    se = np.sqrt(np.diag(cov))

    out = pd.DataFrame({"coef": beta, "t_stat": beta / se}, index=["alpha"] + list(factors.columns))
    out.loc["alpha", "annualised"] = beta[0] * 12
    r2 = 1 - e.var() / y.var()
    out.attrs["r2"] = float(r2)
    out.attrs["months"] = n
    return out
