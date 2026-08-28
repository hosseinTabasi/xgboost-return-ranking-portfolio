# Data

Public daily adjusted closes are downloaded at run time (yfinance first, Yahoo Chart API fallback). Raw per-ticker dumps land in `data/raw/` and are gitignored. The aligned panel is written to `data/processed/`.

- `prices.parquet` / `prices.csv`: daily adjusted close, columns = tickers (including macro if available).
- `returns_daily.parquet`: daily simple returns.
- `panel.parquet`: month-end feature panel (long format) plus next-month target `y`.
- `monthly_returns.parquet`: next-month returns, wide (decision date × asset).

If every public download fails, `src/data_loader.py` writes a **TOY** synthetic panel and every table/figure is labelled TOY. Do not treat TOY numbers as a market backtest.

Yahoo/yfinance closes are split- and dividend-adjusted as provided by the vendor. That adjustment is a known limitation (see `docs/REPORT.md`).
