# XGBoost return ranking versus traditional factors for adaptive multi-asset portfolios

Hossein Tabasi  
28 August 2026 (IST)  
Workshop report. MIT Licence, copyright Hossein Tabasi 2026.

This note records a single walk-forward experiment. Every performance number is taken from `results/tables/performance.csv` and `results/tables/subperiods.csv` after `python -m src.main`. The run is labelled **FULL-public**: daily adjusted closes were downloaded with yfinance for the requested universe. The study does not report live trading profit and loss, does not use a broker, and does not claim a deployable allocation rule.

## 1. Research question

Does an XGBoost model that ranks assets using technical, macro, and cross-asset features known at month-end produce better out-of-sample risk-adjusted returns than (a) equal-weight, (b) simple momentum, and (c) historical mean-variance, after 15 basis points one-way costs on stocks, ETFs, gold and bonds and 40 basis points one-way on crypto?

The question is about *ranking for portfolio construction*, not about point-forecast R-squared. A model can have weak month-ahead R-squared and still help if it orders names well enough that a long-only book beats a dumb benchmark after turnover. Conversely, a high raw return that comes from concentrating in the two crypto names is not an improvement if volatility and drawdown deteriorate by enough to lower Sharpe. Both possibilities are left open; the tables decide.

## 2. Design

Decisions are monthly. At the last equity trading day of month \(t\), features are computed from prices dated on or before that day. The supervised target is the simple return from that close to the next month-end close. Crypto prints seven days a week; they are aligned to the equity month-end by taking the last available crypto close on or before that calendar date (forward-fill capped at seven days). The portfolio is held until the next month-end and then rebalanced.

Walk-forward is expanding. A training row at decision date \(s\) uses features from data \(\le s\) and a target equal to the return from \(s\) to \(s+1\). When predicting at \(t\), only rows with \(s < t\) are used, so every training target has already been realised by time \(t\). The first test month is the first month-end that has at least 36 complete training months. Seed is 42. CPU only.

This is not a claim that the protocol is free of every form of snooping. The feature list, the XGB grid, the universe, and the portfolio rules were specified before the run, but they were specified with knowledge of the 2016–2026 public-market narrative. Multiple testing is discussed in Section 8.

## 3. Data

`configs/universe.yaml` requests 2016-01-01 through 2026-08-01. yfinance succeeded for every name; the Yahoo Chart API fallback was not needed. Actual file ranges:

- Equities, sector ETFs, GLD, TLT, IEF, LQD, ^VIX, ^TNX: 2016-01-04 to 2026-07-31 (about 2,659 daily bars).
- BTC-USD: 2016-01-01 to 2026-07-31 (3,865 daily bars, including weekends).
- ETH-USD: 2017-11-09 to 2026-07-31 (3,187 bars). ETH therefore enters the ranking universe only after it has enough history for a 200-day moving average and a 12-month momentum.

The investable set that was actually used has 18 names: AAPL, MSFT, AMZN, GOOGL, JPM, JNJ, XOM, NVDA, XLK, XLF, XLE, XLV, GLD, TLT, IEF, LQD, BTC-USD, ETH-USD. Macro series ^VIX and ^TNX are feature-only. Download notes are in `results/tables/download_log.csv`. Processed files are under `data/processed/` (`prices.parquet`, `returns_daily.parquet`, `panel.parquet`, `monthly_returns.parquet`).

Yahoo/yfinance closes are split- and dividend-adjusted as supplied by the vendor. That is a known limitation: total-return series constructed this way are not identical to a point-in-time unadjusted series plus separate dividends, and corporate actions can be restated.

After dropping rows with missing required features, the month-end panel has 1,934 asset-date rows, 114 distinct month-ends, and 18 assets. The out-of-sample window used for every strategy in the comparison table is the XGB prediction index: 77 months from 2020-02-28 to 2026-06-30. The last requested calendar month is unused as a *decision* date because the August 2026 return is not in the file.

## 4. Features

All predictors are lagged. The target column is named `y` and is the only column that uses the next month-end price. `lag_target_ret_1m` is the previous month's realised return of the same asset; it is not `y`. `tests/test_no_lookahead.py` checks three properties: `y` matches the one-month price ratio, no required feature is numerically equal to `y` or almost collinear with it at correlation 0.95, and mutating all prices *after* a decision date leaves features at that date unchanged.

The feature list at each month-end, per asset:

- Momentum: 1m, 3m, 6m, 12m simple returns from month-end prices, plus 12–1 momentum \(P_{t-1}/P_{t-12}-1\).
- Volatility: 21-day and 63-day realised volatility of daily returns, annualised by \(\sqrt{252}\); 12-month rolling maximum drawdown on daily prices (252-day window).
- Moving-average distance: close versus 50-day and 200-day moving averages, sampled at month-end.
- RSI(14) and a MACD histogram (12–26–9) sampled at month-end from the daily series of that asset.
- Cross-sectional percentile ranks, that month, of 12m momentum, 21-day volatility, and 1m return.
- Macro: one-month change in the VIX *level* and in the 10-year yield *level*, plus the equal-weight 1m return of the investable universe (`mkt_ret_1m`).

`mom_1m` and `lag_target_ret_1m` are identical by construction. Trees can split gain across duplicates; Ridge cannot. The duplication is left in the panel because the specification asked for a clearly named lagged target as well as 1m momentum. It is a defect of the feature matrix, not a source of look-ahead.

## 5. Models

The main model is `xgboost.XGBRegressor` (`tree_method=hist`, `objective=reg:squarederror`, seed 42). At each test date the last 12 months of the training window are held out as a validation tail. A grid over `n_estimators ∈ {100, 200}`, `max_depth ∈ {3, 5}`, `learning_rate ∈ {0.05, 0.1}` is scored by validation MSE. The winning triple is refit on the full training window and used to predict the next-month return of every asset with a complete feature row. The last-fold model is saved at `models/xgb_last.json`. In this run the last-fold parameters were `n_estimators=100`, `max_depth=3`, `learning_rate=0.05`. The engine was xgboost 3.4.1; a HistGradientBoosting substitute exists in `src/models.py` but was not used.

Baselines that *do* use the same walk-forward split: Ridge (`StandardScaler` + `Ridge(alpha=1)`), RandomForestRegressor (`n_estimators=200`, `max_depth=5`, `min_samples_leaf=2`). A non-ML score is 12–1 momentum (with 12m momentum as fallback). Momentum is not trained; it is only ranked.

## 6. Portfolios and costs

From each model's scores, four books are formed each month:

1. Long-only top-5 equal-weight.
2. Long-only top 30% of the universe, weighted by clipped-positive scores.
3. Long-short: long top 5 / short bottom 5, 10% per name, 50/50 book (gross 1, net 0).
4. Long-only mean-variance: predicted means, 12-month daily covariance scaled by 21, SLSQP maximising \(w^\top\mu - (\gamma/2) w^\top\Sigma w\) with \(\gamma=5\), \(0 \le w_i \le 0.20\), \(\sum w=1\).

Three research baselines do not use XGB: (a) equal-weight all names with a non-missing next-month return; (b) 12–1 momentum top-5 equal-weight; (c) historical mean-variance using trailing 12-month realised means rather than predicted means, same covariance and weight caps. An optional 60/40 book puts 40% in TLT and 60% equal-weight in the risky sleeve {US large-cap, sector ETFs, GLD, BTC-USD, ETH-USD}. IEF is the configured fallback if TLT is missing; TLT was present throughout.

Start wealth is 1. Costs are applied on one-way turnover as \(\sum_i |w_{t,i}-w_{t-1,i}| \times c_i\) with \(c_i=15\) bps except 40 bps for BTC-USD and ETH-USD. Initial weights are zero, so the first month pays full entry. Net return is \((1+r_{\mathrm{gross}})(1-\mathrm{cost})-1\). `tests/test_costs.py` checks that a full rotation from a 15 bp name into a 40 bp name charges 55 bps and that wealth with costs is strictly below wealth without.

Average one-way turnover reported in the table is the conventional \(0.5\sum_i |Δw_i|\).

## 7. Full-sample results

Assumptions printed on every table: one-way costs 15 bps stocks/ETFs, 40 bps crypto; OOS 2020-02-28 to 2026-06-30; data_mode=FULL-public; prices 2016-01-01 to 2026-07-31. Annualisation uses 12 monthly observations, geometric mean return, sample standard deviation, Sortino with the full-sample downside second moment, max drawdown on the net wealth path, Calmar = annualised return / |max DD|.

Copied from `results/tables/performance.csv` (net of costs unless noted):

- equal_weight: ann. return 0.250, vol 0.189, Sharpe 1.322, Sortino 2.483, max DD −0.239, Calmar 1.044, avg turnover 0.006, cost-drag on Sharpe 0.003, end wealth 4.185.
- hist_mv: return 0.347, vol 0.254, Sharpe 1.365, Sortino 3.004, max DD −0.221, Calmar 1.569, turnover 0.190, drag 0.045, wealth 6.769.
- mom_top5_ew: return 0.384, vol 0.321, Sharpe 1.196, Sortino 2.575, max DD −0.350, Calmar 1.097, turnover 0.212, drag 0.042, wealth 8.049.
- sixty_forty: return 0.151, vol 0.155, Sharpe 0.973, Sortino 1.721, max DD −0.269, Calmar 0.563, turnover 0.006, drag 0.003, wealth 2.468.
- xgb_top5_ew: return 0.297, vol 0.353, Sharpe 0.840, Sortino 1.442, max DD −0.482, Calmar 0.615, turnover 0.399, drag 0.066, wealth 5.300. Gross Sharpe 0.906.
- xgb_mv: return 0.249, vol 0.274, Sharpe 0.908, Sortino 1.628, max DD −0.341, Calmar 0.729, turnover 0.505, drag 0.106, wealth 4.161.
- xgb_top30_sw: return 0.318, vol 0.413, Sharpe 0.771, max DD −0.591, wealth 5.889.
- xgb_ls: return 0.072, vol 0.159, Sharpe 0.455, max DD −0.270, wealth 1.564. The long-short book did not blow up.
- ridge_top5_ew: Sharpe 0.883, wealth 5.402.
- rf_top5_ew: return 0.445, vol 0.375, Sharpe 1.187, max DD −0.371, wealth 10.600. Highest terminal wealth; still below equal-weight and historical MV on Sharpe.

No strategy produced a non-finite wealth path. Historical mean-variance has the highest net Sharpe (1.365) and the shallowest max drawdown among the high-return books (−22.1%). Equal-weight is second on Sharpe (1.322) with almost no turnover. Momentum wins on terminal wealth among the three research baselines but not on Sharpe. XGB top-5 sits between equal-weight and momentum on *return* and last on *Sharpe* among those four.

## 8. Subperiods

The 2017–2019 slice is empty: the expanding window forbids a test month until 36 months of feature rows exist, and that date is February 2020. Subperiod statistics use the same net monthly series, sliced by decision date (`results/tables/subperiods.csv`).

**2020** (11 months, from 28 February). This is a short, high-dispersion window that includes the COVID recovery and a crypto rally. Momentum top-5 annualised at 1.692 with Sharpe 4.16. Historical MV: 1.032 / 2.98. Equal-weight: 0.618 / 2.19. XGB top-5: 0.884 / 1.76. XGB participated in the rally but with more volatility than equal-weight (vol 0.501 vs 0.283) and a worse Sharpe.

**2021** (12 months). Equal-weight Sharpe 2.14 (return 0.331, vol 0.155). XGB top-5 Sharpe 1.06 (return 0.355, vol 0.336). Again a higher or similar return at roughly double volatility. Random forest top-5 had Sharpe 1.63; still below equal-weight.

**2022** (12 months). This is the year that distinguishes concentration from diversification. XGB top-5 lost 13.1% with vol 0.520 and max DD −0.419. XGB top-30% lost 15.0% with max DD −0.482. Ridge top-5 lost 22.6%. Equal-weight lost 6.9% (max DD −0.234). Momentum lost 6.3%. Historical MV *gained* 6.5% (Sharpe 0.26). The 60/40 book lost 12.3%. XGB ranking did not provide crash protection in the year when crypto, duration, and long-duration growth all failed together.

**2023–2026** (42 months, through 30 June 2026). Equal-weight Sharpe 1.97 (return 0.248, vol 0.126, max DD −0.071). XGB top-5 Sharpe 1.23 (return 0.302, vol 0.245). XGB top-30% Sharpe 1.45. Random forest top-5 Sharpe 1.84, the closest ML book to equal-weight on this slice, still short of 1.97. Historical MV Sharpe 1.38. The ranking models continue to buy volatility that is not compensated, on a Sharpe metric, by the extra return.

## 9. Feature importance and SHAP

Last-fold gain (`results/tables/feature_importance.csv`, `results/figures/feature_importance.png`) is relatively flat. The top three are `tnx_chg_1m` (0.188), `mom_3m` (0.179), and `dist_ma200` (0.175). Macro and market-wide features (`vix_chg_1m`, `mkt_ret_1m`) sit in the upper half. Cross-sectional 12m-momentum rank is near the bottom (0.085). `lag_target_ret_1m` is last (0.078), which is consistent with it being a duplicate of `mom_1m` (0.107): the booster splits a shared signal. A SHAP summary for the last training window was written to `results/figures/shap_summary.png`. Gain and SHAP both describe the last fold only; they are not a causal attribution and they are not stable by construction across earlier folds.

## 10. Did an XGB ranking edge survive 15/40 bps?

No, not against the three named baselines on risk-adjusted returns over 2020-02-28 to 2026-06-30.

Net Sharpe: XGB top-5 0.840 versus equal-weight 1.322, momentum top-5 1.196, historical MV 1.365. The same ordering holds for Calmar (0.615 vs 1.044 / 1.097 / 1.569) and for maximum drawdown (−48.2% vs −23.9% / −35.0% / −22.1%). Gross-of-cost Sharpe for XGB top-5 is 0.906, still below all three baselines net of costs. Cost drag of 0.066 Sharpe units is real — turnover of 0.40 one-way per month is expensive at 15/40 bps — but it is not the reason the comparison fails. An XGB mean-variance overlay (Sharpe 0.908) and a score-weighted top 30% (0.771) do not reverse the conclusion. Random forest, which is a related tree ensemble without the small XGB grid, reaches Sharpe 1.187 and wealth 10.60; that is a higher return, not a higher Sharpe than equal-weight.

A reader who cares only about compound wealth would note that XGB top-5 finished at 5.30 versus 4.19 for equal-weight. That is not the research question as posed, and it is not robust to 2022.

The honest summary is therefore: on this 18-asset public sample, after realistic one-way costs, XGBoost ranking did not improve out-of-sample Sharpe, Sortino, max drawdown, or Calmar relative to equal-weight, 12–1 momentum, or historical mean-variance. Historical mean-variance was the strongest of the three research baselines on Sharpe and drawdown. Equal-weight was close behind and almost costless.

## 11. Limitations

Look-ahead. Features at \(t\) do not use prices after \(t\), and the target is stored only as `y`. That is necessary, not sufficient. Sampling daily indicators at month-end still uses the same close that a trader might not have known until the cash-equity close, and crypto as-of alignment can mix a weekend print with a Friday equity close.

Snooping and multiple testing. Ten portfolio rules are reported. Even with a pre-specified grid, the chance that *some* ML book looks competitive is material. No Deflated Sharpe Ratio or haircut for the search is applied. The XGB grid is small, which limits overfit to hyperparameters but also limits the model's capacity.

Universe. Eighteen names, with XLK overlapping AAPL/MSFT/NVDA/GOOGL and with two crypto assets whose volatility dominates any top-k that includes them. Results should not be read as a statement about a 500-name equity universe or about futures.

Sample. The test window is February 2020 to June 2026. It contains one pandemic crash, one inflation shock, and a large crypto cycle. The 36-month training minimum deletes 2017–2019 from the test set, so the comparison is silent on that period.

Vendor data. yfinance adjusted closes incorporate splits and dividends as Yahoo records them. Point-in-time restatements are not audited here. ETH is missing 2016–2017 by construction of the Yahoo series.

Costs. One-way 15/40 bps on monthly turnover ignores spreads that widen in 2020-03 and 2022-05, ignores borrow fees on the long-short book, and ignores the difference between BTC-USD (a Yahoo index-like series) and a tradable perpetual. Capacity is not modelled.

Model. Ridge sees a duplicated column. Last-fold depth 3 and 100 trees is a conservative booster; a different loss (ranking loss, listwise) was not tried. Predicted-mean mean-variance uses a utility with \(\gamma=5\) rather than a fully specified risk-target problem; SLSQP failures fall back to a clipped-mean heuristic.

None of these caveats is a licence to replace the table with a more flattering statistic. The table is the result.

## 12. Reproduction

```
MPLBACKEND=Agg python -m src.main
python -m pytest -q
```

CI runs only pytest. The pack venv used for this write-up is `/workspace/.venv` with xgboost 3.4.1 and shap 0.52.0. Seed 42. If a future download fails entirely, `src/data_loader.py` emits a synthetic panel labelled TOY and every table must be read as TOY; that branch was not taken here.

## 13. Conclusion

The experiment is a complete, lagged, cost-aware comparison of an XGBoost next-month return ranker with equal-weight, simple momentum, and historical mean-variance on 18 public assets from 2016 through July 2026, evaluated out of sample from February 2020. After 15/40 bps one-way costs, XGB top-5 equal-weight had net Sharpe 0.84 against 1.32 / 1.20 / 1.37 for those three rules. The extra compound return of the XGB long-only books was purchased with higher volatility, higher turnover, and a 2022 drawdown that equal-weight and historical MV did not match. That is the finding that belongs in a workshop discussion: a flexible ranker with a standard technical-plus-macro panel did not, on this sample, beat the boring books once costs and risk are counted.
