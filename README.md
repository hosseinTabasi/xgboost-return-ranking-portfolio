# XGBoost Return Ranking vs Traditional Factors for Adaptive Multi-Asset Portfolio Construction

**Author:** Hossein Tabasi  
**Licence:** MIT (Copyright 2026 Hossein Tabasi)  
**Data mode of the checked-in run:** FULL-public (yfinance daily adjusted closes)

This repository studies whether an expanding-window XGBoost ranker, using only information known at month-end, improves out-of-sample risk-adjusted returns relative to three non-ML rules after one-way transaction costs. All numbers below are copied from `results/tables/performance.csv` produced by `python -m src.main`. They are not live trading profit and loss.

## Research question

Does an XGBoost model that ranks assets using lagged technical, macro, and cross-asset features produce better out-of-sample risk-adjusted returns than (a) equal-weight, (b) simple momentum, and (c) historical mean-variance, after 15 bps (stocks/ETFs) and 40 bps (crypto) one-way costs?

## How to run

```bash
# from this directory; the pack venv is /workspace/.venv
/workspace/.venv/bin/pip install -e ".[dev]"
MPLBACKEND=Agg /workspace/.venv/bin/python -m src.main
/workspace/.venv/bin/python -m pytest -q
```

Equivalent: `python src/main.py` or `python scripts/run_experiment.py`. CI (`.github/workflows/ci.yml`) runs pytest only and does not download prices.

Configuration: `configs/universe.yaml` (tickers, 2016-01-01 to 2026-08-01 request window, cost bps, seed 42, XGB grid).

## Universe and sample (this run)

- **Investable (18):** AAPL, MSFT, AMZN, GOOGL, JPM, JNJ, XOM, NVDA, XLK, XLF, XLE, XLV, GLD, TLT, IEF, LQD, BTC-USD, ETH-USD. No ticker was dropped.
- **Macro (feature-only):** ^VIX, ^TNX. Both downloaded.
- **Price files:** 2016-01-01 to 2026-07-31 (BTC includes 2016-01-01; ETH-USD starts 2017-11-09; equities 2016-01-04 to 2026-07-31).
- **OOS backtest:** 77 month-end decisions, 2020-02-28 to 2026-06-30 (36 months of expanding-window history required before the first test month; the last decision needs a realised next month, so July 2026 is unused as a decision date).
- **2017–2019** is empty in the subperiod table because the walk-forward test window starts in February 2020.

## Findings (net of costs)

Costs: 15 bps stocks/ETFs/gold/bonds, 40 bps crypto, charged as \(\sum_i |Δw_i| \times c_i\) at each monthly rebalance. Start wealth = 1. Source: `results/tables/performance.csv`.

| strategy | ann ret | vol | Sharpe | Sortino | max DD | Calmar | avg TO | cost drag (Sharpe) | end wealth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| equal_weight | 0.250 | 0.189 | **1.322** | 2.483 | -0.239 | 1.044 | 0.006 | 0.003 | 4.19 |
| hist_mv | 0.347 | 0.254 | **1.365** | 3.004 | -0.221 | 1.569 | 0.190 | 0.045 | 6.77 |
| mom_top5_ew | 0.384 | 0.321 | 1.196 | 2.575 | -0.350 | 1.097 | 0.212 | 0.042 | 8.05 |
| sixty_forty | 0.151 | 0.155 | 0.973 | 1.721 | -0.269 | 0.563 | 0.006 | 0.003 | 2.47 |
| xgb_top5_ew | 0.297 | 0.353 | 0.840 | 1.442 | -0.482 | 0.615 | 0.399 | 0.066 | 5.30 |
| xgb_mv | 0.249 | 0.274 | 0.908 | 1.628 | -0.341 | 0.729 | 0.505 | 0.106 | 4.16 |
| xgb_top30_sw | 0.318 | 0.413 | 0.771 | 1.269 | -0.591 | 0.538 | 0.504 | 0.088 | 5.89 |
| xgb_ls | 0.072 | 0.159 | 0.455 | 0.728 | -0.270 | 0.268 | 0.477 | 0.144 | 1.56 |
| ridge_top5_ew | 0.301 | 0.340 | 0.883 | 1.531 | -0.451 | 0.667 | 0.375 | 0.068 | 5.40 |
| rf_top5_ew | 0.445 | 0.375 | 1.187 | 2.302 | -0.371 | 1.199 | 0.300 | 0.056 | 10.60 |

**Does any XGB edge survive 15/40 bps costs?** On this sample, no, not on risk-adjusted metrics. XGB top-5 equal-weight earned a higher compound return than equal-weight (29.7% vs 25.0% annualised) but with almost double volatility (35.3% vs 18.9%) and a deeper max drawdown (−48.2% vs −23.9%). Net Sharpe was 0.84 versus 1.32 for equal-weight, 1.20 for 12–1 momentum top-5, and 1.37 for historical mean-variance. Cost drag on Sharpe was 0.066 for XGB top-5 (gross Sharpe 0.91) against 0.003 for equal-weight, in line with 40% versus 0.6% average one-way turnover. The long-short XGB book did not blow up; it finished at wealth 1.56 with net Sharpe 0.46.

Last-fold XGB used `n_estimators=100`, `max_depth=3`, `learning_rate=0.05` (seed 42). Highest gain features: `tnx_chg_1m`, `mom_3m`, `dist_ma200` (`results/tables/feature_importance.csv`).

## Figures

- `results/figures/equity_curves.png` — EW, 60/40, momentum top-5, historical MV, XGB top-5
- `results/figures/equity_curves_xgb_family.png` — XGB variants plus Ridge/RF top-5
- `results/figures/feature_importance.png`
- `results/figures/shap_summary.png` — SHAP on the last train window

## Layout

```
src/           data_loader, features, models, portfolio, backtest, metrics, main
configs/       universe.yaml
tests/         test_no_lookahead.py, test_costs.py
data/          raw/ (gitignored dumps), processed/ panel and prices
models/        xgb_last.json
results/       tables/ and figures/ from the run
docs/REPORT.md workshop write-up
```

## Limitations (short)

Look-ahead is restricted to the target column `y`; features at t use prices on or before t (`tests/test_no_lookahead.py`). Remaining issues: Yahoo corporate-action adjustments, a 18-name universe with overlapping sector ETFs and single stocks, multiple testing across ten portfolio rules, monthly decision dates that ignore intra-month paths, crypto weekend alignment by as-of fill, duplicate `mom_1m` / `lag_target_ret_1m` columns, and a 2020–2026 OOS window that includes one crypto/tech boom-bust. See `docs/REPORT.md`.
