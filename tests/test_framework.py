import numpy as np
import pandas as pd
import pytest

from src.data import parse_ff_csv
from src.regression import factor_regression
from src.strategy import run, summary, tilt_weights

SAMPLE = """This file was created by CMPT_ME_BEME_OP_INV_RETS using the 202409 CRSP database.
The 1-month TBill return is from Ibbotson and Associates Inc.

,Mkt-RF,SMB,HML,RMW,CMA,RF
196307,   -0.39,   -0.48,   -0.81,    0.64,   -1.15,    0.27
196308,    5.07,   -0.80,    1.70,    0.40,   -0.38,    0.25

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RMW,CMA,RF
  1964,   12.51,    0.21,    9.14,    3.42,    6.85,    3.54
"""


def synthetic_ff(n=360, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    data = rng.normal([0.006, 0.002, 0.003, 0.003, 0.002, 0.003], [0.045, 0.03, 0.03, 0.02, 0.02, 0.001], (n, 6))
    return pd.DataFrame(data, index=idx, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])


def test_parser_reads_monthly_block_only_in_decimals():
    df = parse_ff_csv(SAMPLE)
    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("1963-07-31")
    assert df.loc["1963-08-31", "Mkt-RF"] == pytest.approx(0.0507)


def test_zero_tilt_without_costs_is_the_market():
    ff = synthetic_ff()
    w = tilt_weights(ff, "equal", total_tilt=0.0, window=60)
    res = run(ff, w, trade_cost=0, carry_cost=0)
    pd.testing.assert_series_equal(res["strategy"], res["market"], check_names=False)


def test_mv_weights_use_only_past_months():
    ff = synthetic_ff()
    changed = ff.copy()
    changed.iloc[300:, 1:5] += 0.05  # make every factor look great from month 300 on
    w1 = tilt_weights(ff, "mv", 0.5, window=60)
    w2 = tilt_weights(changed, "mv", 0.5, window=60)
    pd.testing.assert_frame_equal(w1.iloc[: 300 - 60 + 1], w2.iloc[: 300 - 60 + 1])


def test_tilts_are_long_only_and_sum_to_total():
    w = tilt_weights(synthetic_ff(), "mv", 0.6, window=60)
    assert (w >= -1e-9).all().all()
    assert np.allclose(w.sum(axis=1), 0.6)


def test_regression_recovers_known_loadings():
    ff = synthetic_ff(600)
    rng = np.random.default_rng(1)
    y = 0.001 + 1.0 * ff["Mkt-RF"] + 0.5 * ff["HML"] - 0.3 * ff["SMB"] + rng.normal(0, 0.002, len(ff))
    reg = factor_regression(y, ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
    assert reg.loc["Mkt-RF", "coef"] == pytest.approx(1.0, abs=0.02)
    assert reg.loc["HML", "coef"] == pytest.approx(0.5, abs=0.03)
    assert reg.loc["SMB", "coef"] == pytest.approx(-0.3, abs=0.03)
    assert reg.loc["alpha", "annualised"] == pytest.approx(0.012, abs=0.004)


def test_summary_has_information_ratio():
    ff = synthetic_ff()
    res = run(ff, tilt_weights(ff, "equal", 0.5, 60))
    assert np.isfinite(summary(res).loc["information_ratio", "Factor tilt"])
