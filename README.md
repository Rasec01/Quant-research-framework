# Quant Research Framework

Factor-investing research on the Fama-French five-factor data: build factor-tilt portfolios month by month using only information available at the time, compare them with the market after costs, and explain any return with a five-factor regression.

This is the approach I used to tilt client portfolios towards factors at St. James International, rebuilt on public data so anyone can rerun it.

## What it does

| Piece | Detail |
|---|---|
| Data | Monthly Fama-French five factors (market, size, value, profitability, investment) and the T-bill rate from Kenneth French's data library, downloaded on first run and cached |
| Portfolios | Market plus long-short factor tilts; total tilt set by `--total-tilt` |
| Weighting | `equal`: the same tilt to each factor. `mv`: long-only maximum-Sharpe mix of the factors over a trailing window, shrunk towards equal tilts |
| No look-ahead | The weights for month *t* use factor returns up to month *t−1* only; a test checks it |
| Costs | A cost per unit of turnover and a yearly carry cost per unit of gross tilt, both configurable |
| Output | Annual return, volatility, Sharpe ratio, maximum drawdown, tracking error, information ratio, turnover; a five-factor regression with Newey-West t-statistics; an optional growth chart |

## Quick start

```bash
pip install -r requirements.txt
python main.py                                       # equal tilts
python main.py --method mv --total-tilt 0.6 --chart outputs/tilt.png
python main.py --start 2000-01 --trade-cost 0.002 --carry-cost 0.01
```

## Reading the output

* **Information ratio** is the tilt's average return above the market divided by how much it strays from the market. It is the number that says whether the tilts were worth the tracking error.
* **The regression** should show loadings close to the tilts you chose and an alpha near minus the costs. A large positive alpha would mean something is wrong, because the strategy is built from the factors themselves.
* Factor premia vary a lot across decades. Compare `--start 1990-01` with `--start 2010-01` before drawing conclusions.

## Results

Run `python main.py --method mv --chart outputs/tilt.png` and paste the tables and chart here.

## Tests

```bash
pytest -q
```

The tests check the Ken French file parser, that a zero tilt without costs equals the market, that changing future factor returns leaves earlier weights untouched, that tilts are long-only and sum to the chosen total, and that the regression recovers known loadings from simulated data.

## Project structure

```
├── main.py
├── src/
│   ├── data.py         # download and parse the Ken French five-factor file
│   ├── strategy.py     # walk-forward tilt weights, costs, performance table
│   └── regression.py   # factor regression with Newey-West standard errors
└── tests/test_framework.py
```

## References

Fama, E. F. and French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1–22.

## Disclaimer

For research and education only. Not investment advice.

## License

MIT
