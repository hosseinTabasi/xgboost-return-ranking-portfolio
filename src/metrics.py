"""Performance statistics from monthly backtest files. No invented numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import BacktestResult

PERIODS_PER_YEAR = 12


def annualised_return(r: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    years = len(r) / float(periods)
    if years <= 0:
        return float("nan")
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return float("nan")
    return growth ** (1.0 / years) - 1.0


def annualised_vol(r: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    r = r.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe(r: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    v = annualised_vol(r, periods)
    if not np.isfinite(v) or v == 0:
        return float("nan")
    return annualised_return(r, periods) / v


def sortino(r: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    downside = r.to_numpy(dtype=float)
    downside = np.minimum(downside, 0.0)
    dd = float(np.sqrt(np.mean(downside**2)) * np.sqrt(periods))
    if dd == 0:
        return float("nan")
    return annualised_return(r, periods) / dd


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if eq.empty:
        return float("nan")
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def calmar(r: pd.Series, equity: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    mdd = max_drawdown(equity)
    if not np.isfinite(mdd) or mdd >= 0:
        return float("nan")
    return annualised_return(r, periods) / abs(mdd)


def summarise_result(res: BacktestResult, res_gross: BacktestResult | None = None) -> dict[str, float | str]:
    net = res.monthly["net_return"] if not res.monthly.empty else pd.Series(dtype=float)
    out: dict[str, float | str] = {
        "strategy": res.name,
        "n_months": int(len(net)),
        "start": str(res.equity.index.min().date()) if len(res.equity) else "",
        "end": str(res.equity.index.max().date()) if len(res.equity) else "",
        "ann_return": annualised_return(net),
        "ann_vol": annualised_vol(net),
        "sharpe": sharpe(net),
        "sortino": sortino(net),
        "max_dd": max_drawdown(res.equity),
        "calmar": calmar(net, res.equity),
        "avg_turnover": float(res.monthly["turnover_one_way"].mean()) if not res.monthly.empty else float("nan"),
        "avg_cost": float(res.monthly["cost"].mean()) if not res.monthly.empty else float("nan"),
        "end_wealth": float(res.equity.iloc[-1]) if len(res.equity) else float("nan"),
    }
    if res_gross is not None and not res_gross.monthly.empty:
        g = res_gross.monthly["net_return"]  # costs off => this is gross
        out["sharpe_gross"] = sharpe(g)
        out["sharpe_net"] = out["sharpe"]
        sg = out["sharpe_gross"]
        sn = out["sharpe_net"]
        out["cost_drag_sharpe"] = (
            float(sg) - float(sn) if np.isfinite(sg) and np.isfinite(sn) else float("nan")
        )
    else:
        out["sharpe_gross"] = float("nan")
        out["sharpe_net"] = out["sharpe"]
        out["cost_drag_sharpe"] = float("nan")
    return out


def subperiod_table(
    results: dict[str, BacktestResult],
    periods: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """`periods` is a list of (label, start_inclusive, end_inclusive) ISO dates."""
    rows = []
    for label, start, end in periods:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        for name, res in results.items():
            if res.monthly.empty:
                continue
            sl = res.monthly.loc[(res.monthly.index >= s) & (res.monthly.index <= e)]
            if sl.empty:
                continue
            eq = (1.0 + sl["net_return"]).cumprod()
            r = sl["net_return"]
            rows.append(
                {
                    "period": label,
                    "start": str(sl.index.min().date()),
                    "end": str(sl.index.max().date()),
                    "strategy": name,
                    "n_months": int(len(sl)),
                    "ann_return": annualised_return(r),
                    "ann_vol": annualised_vol(r),
                    "sharpe": sharpe(r),
                    "max_dd": max_drawdown(eq),
                }
            )
    return pd.DataFrame(rows)


def format_pct(x: float, digits: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"
