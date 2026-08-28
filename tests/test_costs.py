"""Turnover times bps must reduce wealth relative to the zero-cost path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import cost_bps_series, run_backtest
from src.metrics import annualised_return, sharpe


def test_turnover_bps_reduces_wealth() -> None:
    dates = pd.DatetimeIndex(["2020-01-31", "2020-02-28", "2020-03-31"])
    assets = ["AAPL", "BTC-USD"]
    # Full rotation each month: 100% AAPL -> 100% BTC -> 100% AAPL.
    weights = pd.DataFrame(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        index=dates,
        columns=assets,
    )
    rets = pd.DataFrame(
        [
            [0.02, 0.02],
            [0.02, 0.02],
            [0.02, 0.02],
        ],
        index=dates,
        columns=assets,
    )
    bps = cost_bps_series(assets, crypto=["BTC-USD"], default_bps=15.0, crypto_bps=40.0)
    net = run_backtest(weights, rets, bps, name="net", charge_costs=True)
    gross = run_backtest(weights, rets, bps, name="gross", charge_costs=False)

    assert float(net.equity.iloc[-1]) < float(gross.equity.iloc[-1])
    # Month 0: buy 100% AAPL, cost = 1.0 * 15bps.
    assert abs(float(net.monthly.iloc[0]["cost"]) - 0.0015) < 1e-12
    # Month 1: sell AAPL 100% and buy BTC 100% -> 1.0*15bps + 1.0*40bps.
    assert abs(float(net.monthly.iloc[1]["cost"]) - 0.0055) < 1e-12
    # One-way turnover 0.5 * L1; month 1 L1 = 2.0 so turnover = 1.0.
    assert abs(float(net.monthly.iloc[1]["turnover_one_way"]) - 1.0) < 1e-12


def test_zero_turnover_zero_cost_after_entry() -> None:
    dates = pd.DatetimeIndex(["2020-01-31", "2020-02-28"])
    weights = pd.DataFrame({"AAPL": [1.0, 1.0]}, index=dates)
    rets = pd.DataFrame({"AAPL": [0.01, 0.01]}, index=dates)
    bps = pd.Series({"AAPL": 15.0})
    net = run_backtest(weights, rets, bps, name="net")
    assert float(net.monthly.iloc[0]["cost"]) > 0
    assert abs(float(net.monthly.iloc[1]["cost"])) < 1e-15


def test_sharpe_net_not_above_gross_when_costs_positive() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-31", periods=24, freq="ME")
    w = pd.DataFrame({"AAPL": rng.uniform(0.2, 0.8, size=len(dates))}, index=dates)
    w["MSFT"] = 1.0 - w["AAPL"]
    r = pd.DataFrame(
        {
            "AAPL": rng.normal(0.008, 0.04, size=len(dates)),
            "MSFT": rng.normal(0.007, 0.035, size=len(dates)),
        },
        index=dates,
    )
    bps = pd.Series({"AAPL": 15.0, "MSFT": 15.0})
    net = run_backtest(w, r, bps, name="net")
    gross = run_backtest(w, r, bps, name="gross", charge_costs=False)
    assert annualised_return(net.monthly["net_return"]) < annualised_return(gross.monthly["net_return"])
    # With positive costs, net Sharpe should not exceed gross on this path.
    assert sharpe(net.monthly["net_return"]) <= sharpe(gross.monthly["net_return"]) + 1e-12
