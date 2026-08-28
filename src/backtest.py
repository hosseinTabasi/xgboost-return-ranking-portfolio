"""Monthly rebalanced backtest with one-way transaction costs.

Cost at rebalance t is sum_i |w_{t,i} - w_{t-1,i}| * c_i, i.e. both buys and
sells are charged at the one-way bps rate (15 bps stocks/ETFs, 40 bps crypto).
Conventional one-way turnover is 0.5 * sum_i |Δw_i|.

Net return uses (1 + r_gross) * (1 - cost) - 1. Initial weights are zero, so
the first month pays full entry costs. Start wealth = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    monthly: pd.DataFrame
    start_wealth: float = 1.0


def cost_bps_series(
    assets: list[str],
    crypto: list[str],
    default_bps: float = 15.0,
    crypto_bps: float = 40.0,
) -> pd.Series:
    bps = pd.Series(float(default_bps), index=assets, dtype=float)
    for c in crypto:
        if c in bps.index:
            bps.loc[c] = float(crypto_bps)
    return bps


def run_backtest(
    weights: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    cost_bps: pd.Series,
    *,
    name: str,
    start_wealth: float = 1.0,
    charge_costs: bool = True,
) -> BacktestResult:
    """`monthly_returns` is indexed by decision date; values are next-month asset returns."""
    assets = [c for c in weights.columns if c in monthly_returns.columns]
    w_df = weights[assets].sort_index()
    r_df = monthly_returns[assets]
    dates = w_df.index.intersection(r_df.index)
    bps = cost_bps.reindex(assets).fillna(float(cost_bps.median() if len(cost_bps) else 15.0))
    c = (bps / 10000.0) if charge_costs else pd.Series(0.0, index=assets)

    w_prev = pd.Series(0.0, index=assets, dtype=float)
    wealth = float(start_wealth)
    recs: list[dict] = []
    equity_pts = []
    for t in dates:
        w = w_df.loc[t].reindex(assets).fillna(0.0).astype(float)
        r = r_df.loc[t].reindex(assets).astype(float)
        dw = w - w_prev
        cost = float((dw.abs() * c).sum()) if charge_costs else 0.0
        turnover = 0.5 * float(dw.abs().sum())
        # Holdings with a missing realised return contribute 0 (weight should be 0).
        gross = float(np.nansum(w.to_numpy() * r.to_numpy()))
        net = (1.0 + gross) * (1.0 - cost) - 1.0
        wealth = wealth * (1.0 + net)
        recs.append(
            {
                "date": t,
                "gross_return": gross,
                "cost": cost,
                "net_return": net,
                "turnover_one_way": turnover,
                "wealth": wealth,
            }
        )
        equity_pts.append((t, wealth))
        w_prev = w

    monthly = pd.DataFrame(recs)
    if monthly.empty:
        equity = pd.Series(dtype=float, name=name)
    else:
        monthly = monthly.set_index("date").sort_index()
        equity = pd.Series({d: v for d, v in equity_pts}, name=name).sort_index()
    return BacktestResult(name=name, equity=equity, monthly=monthly, start_wealth=start_wealth)


def align_equal_weight_weights(
    monthly_returns: pd.DataFrame,
    dates: pd.Index | None = None,
) -> pd.DataFrame:
    idx = dates if dates is not None else monthly_returns.index
    rows = []
    for t in idx:
        if t not in monthly_returns.index:
            continue
        r = monthly_returns.loc[t]
        assets = r.replace([np.inf, -np.inf], np.nan).dropna().index.tolist()
        w = pd.Series(0.0, index=monthly_returns.columns, dtype=float)
        if assets:
            w.loc[assets] = 1.0 / len(assets)
        w.name = t
        rows.append(w)
    return pd.DataFrame(rows).sort_index() if rows else pd.DataFrame()
