"""
Fama-French factor-tilt research.

    python main.py                                  # walk-forward max-Sharpe tilts, 1973 onwards
    python main.py --method equal                   # fixed equal tilts only
    python main.py --total-tilt 0.6 --chart outputs/tilt.png
    python main.py --start 2000-01 --trade-cost 0.002 --carry-cost 0.01
"""
import argparse
from pathlib import Path

from src.data import FACTORS, load_ff5
from src.regression import factor_regression
from src.strategy import run, summary, tilt_weights


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["equal", "mv"], default="mv")
    p.add_argument("--total-tilt", type=float, default=0.5, help="sum of long-short factor weights")
    p.add_argument("--window", type=int, default=120, help="months of history used for each decision")
    p.add_argument("--shrink", type=float, default=0.5, help="'mv' only: pull towards equal tilts (0-1)")
    p.add_argument("--trade-cost", type=float, default=0.001, help="cost per unit of tilt turnover")
    p.add_argument("--carry-cost", type=float, default=0.005, help="annual cost per unit of gross tilt")
    p.add_argument("--start", default=None, help="first month to report, e.g. 2000-01")
    p.add_argument("--chart", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    ff = load_ff5()
    w = tilt_weights(ff, a.method, a.total_tilt, a.window, a.shrink)
    res = run(ff, w, a.trade_cost, a.carry_cost)
    if a.start:
        res = res.loc[a.start:]

    print(f"\nFactor tilt ({a.method}, total tilt {a.total_tilt}) vs market, "
          f"{res.index[0]:%Y-%m} to {res.index[-1]:%Y-%m}, after costs:")
    table = summary(res)
    if a.method != "equal":
        # Baseline: the same total tilt spread equally, so the table shows what the walk-forward choice adds.
        base = run(ff, tilt_weights(ff, "equal", a.total_tilt, a.window), a.trade_cost, a.carry_cost).loc[res.index]
        table.insert(1, "Equal tilt", summary(base)["Factor tilt"])
    print(table.round(4).to_string())
    print("\nAverage tilt weights:")
    print(w.loc[res.index].mean().round(3).to_string())

    if a.method == "equal":
        # Fixed tilts are an exact mix of the factors, so a factor regression would only give the weights back.
        print("\nNo factor regression: with fixed tilts it would just return the weights above.")
    else:
        reg = factor_regression(res["strategy"] - res["rf"], ff[["Mkt-RF"] + FACTORS])
        print(f"\nFive-factor regression of the strategy (R-squared {reg.attrs['r2']:.3f}, {reg.attrs['months']} months).")
        print("Alpha is what the month-by-month choice of tilts adds beyond average factor exposure, after costs:")
        print(reg.round(4).to_string())

    if a.chart:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        (1 + res["strategy"]).cumprod().plot(ax=ax, logy=True, label="Factor tilt (after costs)")
        (1 + res["market"]).cumprod().plot(ax=ax, logy=True, label="Market")
        ax.set_ylabel("Growth of 1 (log scale)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        Path(a.chart).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.chart, dpi=150)
        print(f"\nChart saved to {a.chart}")


if __name__ == "__main__":
    main()
