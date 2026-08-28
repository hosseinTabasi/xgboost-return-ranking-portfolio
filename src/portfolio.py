"""Monthly portfolio rules from scores or historical moments.

All constructors return a weight Series indexed by asset (missing names => 0).
Long-only weights sum to 1 when at least one name is eligible. The long-short
book is 50/50 (gross 1, net 0) when both legs can be filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _eligible(scores: pd.Series) -> pd.Series:
    return scores.replace([np.inf, -np.inf], np.nan).dropna()


def top_k_equal(scores: pd.Series, k: int = 5) -> pd.Series:
    s = _eligible(scores)
    w = pd.Series(0.0, index=scores.index, dtype=float)
    if s.empty:
        return w
    k_use = min(int(k), len(s))
    picked = s.nlargest(k_use).index
    w.loc[picked] = 1.0 / k_use
    return w.fillna(0.0)


def top_frac_score_weighted(scores: pd.Series, frac: float = 0.30) -> pd.Series:
    s = _eligible(scores)
    w = pd.Series(0.0, index=scores.index, dtype=float)
    if s.empty:
        return w
    n = max(1, int(np.ceil(float(frac) * len(s))))
    picked = s.nlargest(n)
    clipped = picked.clip(lower=0.0)
    if float(clipped.sum()) <= 0.0:
        w.loc[picked.index] = 1.0 / len(picked)
    else:
        w.loc[clipped.index] = clipped / clipped.sum()
    return w.fillna(0.0)


def long_short_top_bottom(scores: pd.Series, k: int = 5) -> pd.Series:
    s = _eligible(scores)
    w = pd.Series(0.0, index=scores.index, dtype=float)
    if len(s) < 2:
        return w
    k_use = min(int(k), max(1, len(s) // 2))
    long_idx = s.nlargest(k_use).index
    short_idx = s.nsmallest(k_use).index
    # Guard against overlap on a tiny universe.
    short_idx = [i for i in short_idx if i not in set(long_idx)]
    if not len(long_idx) or not len(short_idx):
        return w
    w.loc[list(long_idx)] = 0.5 / len(long_idx)
    w.loc[list(short_idx)] = -0.5 / len(short_idx)
    return w.fillna(0.0)


def equal_weight(assets: list[str] | pd.Index) -> pd.Series:
    assets = list(assets)
    if not assets:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(assets), index=assets, dtype=float)


def sixty_forty(
    available: list[str],
    *,
    bond: str,
    risky: list[str],
    bond_weight: float = 0.40,
    bond_fallback: str | None = None,
) -> pd.Series:
    bond_name = bond if bond in available else bond_fallback
    risky_use = [a for a in risky if a in available and a != bond_name]
    w = pd.Series(0.0, index=available, dtype=float)
    if bond_name is None or bond_name not in available or not risky_use:
        return equal_weight(available)
    w.loc[bond_name] = float(bond_weight)
    w.loc[risky_use] = (1.0 - float(bond_weight)) / len(risky_use)
    return w


def mean_variance_long_only(
    mu: pd.Series,
    cov: pd.DataFrame,
    *,
    max_weight: float = 0.20,
    gamma: float = 5.0,
) -> pd.Series:
    """Max w'μ - (γ/2) w'Σw, long-only, sum weights = 1, cap per name."""
    names = [n for n in mu.index if n in cov.index and n in cov.columns]
    mu = mu.reindex(names).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cov = cov.reindex(index=names, columns=names).astype(float)
    cov = cov.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n = len(names)
    if n == 0:
        return pd.Series(dtype=float)
    cap = float(max_weight)
    if n * cap < 1.0:
        cap = 1.0 / n
    sigma = cov.to_numpy(dtype=float)
    # Numerical ridge so SLSQP does not see a singular Hessian.
    sigma = sigma + 1e-8 * np.eye(n)
    mu_v = mu.to_numpy(dtype=float)

    def objective(w: np.ndarray) -> float:
        return float(0.5 * gamma * w @ sigma @ w - w @ mu_v)

    cons = {"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}
    bounds = [(0.0, cap)] * n
    w0 = np.clip(np.ones(n) / n, 0.0, cap)
    w0 = w0 / w0.sum()
    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 250, "ftol": 1e-9, "disp": False},
    )
    if res.success:
        w = np.clip(res.x, 0.0, cap)
        s = float(w.sum())
        w = w / s if s > 0 else w0
    else:
        # Least-bad: clip-normalise positive means, else equal weight.
        pos = np.clip(mu_v, 0.0, None)
        if pos.sum() <= 0:
            w = w0
        else:
            w = pos / pos.sum()
            w = np.clip(w, 0.0, cap)
            w = w / w.sum()
    return pd.Series(w, index=names, dtype=float)


def trailing_cov_monthly(
    daily_returns: pd.DataFrame,
    decision: pd.Timestamp,
    assets: list[str],
    lookback_days: int = 252,
) -> pd.DataFrame:
    """Daily covariance up to *and including* the decision date, scaled to monthly."""
    hist = daily_returns.loc[daily_returns.index <= decision, assets]
    hist = hist.tail(int(lookback_days)).dropna(axis=1, how="all")
    cov_d = hist.cov(min_periods=max(40, lookback_days // 4))
    return cov_d * 21.0


def trailing_mean_monthly(
    monthly_rets_wide: pd.DataFrame,
    decision: pd.Timestamp,
    assets: list[str],
    lookback_months: int = 12,
) -> pd.Series:
    """Mean of *already realised* monthly returns strictly before the next-month target.

    `monthly_rets_wide` is indexed by decision date and stores the *next* month
    return, so rows with date < decision have realised by `decision`.
    """
    hist = monthly_rets_wide.loc[monthly_rets_wide.index < decision, assets]
    hist = hist.tail(int(lookback_months))
    return hist.mean(axis=0)


def weights_from_scores(
    scores: pd.DataFrame,
    rule: str,
    *,
    top_k: int = 5,
    top_frac: float = 0.30,
) -> pd.DataFrame:
    rows = []
    for dt, row in scores.iterrows():
        if rule == "top5":
            w = top_k_equal(row, k=top_k)
        elif rule == "top30":
            w = top_frac_score_weighted(row, frac=top_frac)
        elif rule == "ls":
            w = long_short_top_bottom(row, k=top_k)
        else:
            raise ValueError(rule)
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_index().fillna(0.0)


def mv_from_means(
    means: pd.DataFrame,
    daily_returns: pd.DataFrame,
    *,
    max_weight: float = 0.20,
    gamma: float = 5.0,
) -> pd.DataFrame:
    rows = []
    for dt, mu_row in means.iterrows():
        assets = [a for a in mu_row.dropna().index if a in daily_returns.columns]
        if len(assets) < 3:
            continue
        cov = trailing_cov_monthly(daily_returns, pd.Timestamp(dt), assets)
        assets = [a for a in assets if a in cov.index]
        if len(assets) < 3:
            continue
        w = mean_variance_long_only(
            mu_row.reindex(assets),
            cov,
            max_weight=max_weight,
            gamma=gamma,
        )
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_index().fillna(0.0)
