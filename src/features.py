"""Month-end feature panel with a strictly lagged target.

Every feature at decision date t is computed from prices on or before t.
The target `y` is the next calendar-month return (look-ahead only as y).
`lag_target_ret_1m` is the *previous* month's realised return of the same
asset; it is not the future target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Core columns that must be finite before the first walk-forward date.
REQUIRED_FEATURES = [
    "mom_1m",
    "mom_3m",
    "mom_6m",
    "mom_12m",
    "vol_21d",
    "vol_63d",
    "mdd_12m",
    "dist_ma50",
    "dist_ma200",
    "rsi_14",
    "macd_hist",
    "rank_mom_12m",
    "rank_vol_21d",
    "rank_mom_1m",
    "mkt_ret_1m",
    "lag_target_ret_1m",
]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_hist(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    return macd - signal


def _max_drawdown_rolling(close: pd.Series, window: int = 252, min_periods: int = 200) -> pd.Series:
    roll_max = close.rolling(window, min_periods=min_periods).max()
    dd = close / roll_max - 1.0
    return dd.rolling(window, min_periods=min_periods).min()


def _per_asset(close: pd.DataFrame, fn) -> pd.DataFrame:
    pieces = {}
    for col in close.columns:
        s = close[col].dropna()
        if s.empty:
            continue
        pieces[col] = fn(s)
    return pd.DataFrame(pieces)


def equity_month_ends(prices: pd.DataFrame, calendar_tickers: list[str]) -> pd.DatetimeIndex:
    """Last date in each calendar month on which at least one equity/ETF printed."""
    cols = [c for c in calendar_tickers if c in prices.columns]
    if not cols:
        cols = list(prices.columns)
    ref = prices[cols].dropna(how="all")
    grouped = ref.groupby(ref.index.to_period("M"))
    ends = grouped.apply(lambda g: g.index.max())
    ends = pd.DatetimeIndex(pd.to_datetime(ends.values)).sort_values()
    return ends


def sample_asof(df: pd.DataFrame, dates: pd.DatetimeIndex, limit_days: int = 7) -> pd.DataFrame:
    """Forward-fill at most `limit_days` onto the decision calendar (crypto weekends)."""
    if df.empty:
        return pd.DataFrame(index=dates, columns=df.columns)
    union = df.index.union(dates).sort_values()
    aligned = df.reindex(union).ffill(limit=limit_days)
    return aligned.reindex(dates)


@dataclass
class FeatureBundle:
    panel: pd.DataFrame  # long: date, asset, features, y
    feature_cols: list[str]
    monthly_returns: pd.DataFrame  # wide: decision date x asset = next-month return
    month_end_prices: pd.DataFrame
    daily_returns: pd.DataFrame
    month_ends: pd.DatetimeIndex


def build_feature_panel(
    prices: pd.DataFrame,
    investable: list[str],
    macro: list[str],
    crypto: list[str],
) -> FeatureBundle:
    inv = [t for t in investable if t in prices.columns]
    px = prices[inv + [m for m in macro if m in prices.columns]].sort_index()
    calendar_tickers = [t for t in inv if t not in crypto]
    month_ends = equity_month_ends(px, calendar_tickers)

    close_inv = px[inv]
    daily_ret = close_inv.pct_change()

    vol_21 = _per_asset(close_inv, lambda s: s.pct_change().rolling(21, min_periods=15).std() * np.sqrt(252.0))
    vol_63 = _per_asset(close_inv, lambda s: s.pct_change().rolling(63, min_periods=40).std() * np.sqrt(252.0))
    mdd = _per_asset(close_inv, lambda s: _max_drawdown_rolling(s, 252, 200))
    ma50 = _per_asset(close_inv, lambda s: s.rolling(50, min_periods=50).mean())
    ma200 = _per_asset(close_inv, lambda s: s.rolling(200, min_periods=200).mean())
    rsi = _per_asset(close_inv, lambda s: _rsi(s, 14))
    macd = _per_asset(close_inv, lambda s: _macd_hist(s))

    px_me = sample_asof(close_inv, month_ends)
    vol21_me = sample_asof(vol_21, month_ends)
    vol63_me = sample_asof(vol_63, month_ends)
    mdd_me = sample_asof(mdd, month_ends)
    ma50_me = sample_asof(ma50, month_ends)
    ma200_me = sample_asof(ma200, month_ends)
    rsi_me = sample_asof(rsi, month_ends)
    macd_me = sample_asof(macd, month_ends)

    # Month-end simple returns. y is the *next* interval; features use current and lagged.
    mom_1m = px_me.pct_change(1)
    mom_3m = px_me.pct_change(3)
    mom_6m = px_me.pct_change(6)
    mom_12m = px_me.pct_change(12)
    mom_12_1 = px_me.shift(1) / px_me.shift(12) - 1.0
    y = px_me.pct_change(1).shift(-1)

    dist_ma50 = px_me / ma50_me - 1.0
    dist_ma200 = px_me / ma200_me - 1.0

    rank_mom12 = mom_12m.rank(axis=1, pct=True)
    rank_vol21 = vol21_me.rank(axis=1, pct=True)
    rank_mom1 = mom_1m.rank(axis=1, pct=True)
    mkt_ret_1m = mom_1m.mean(axis=1)

    vix_chg = None
    tnx_chg = None
    if "^VIX" in px.columns:
        vix_me = sample_asof(px[["^VIX"]], month_ends)["^VIX"]
        vix_chg = vix_me.diff(1)
    if "^TNX" in px.columns:
        tnx_me = sample_asof(px[["^TNX"]], month_ends)["^TNX"]
        tnx_chg = tnx_me.diff(1)

    frames = []
    for asset in inv:
        row = pd.DataFrame(
            {
                "mom_1m": mom_1m[asset],
                "mom_3m": mom_3m[asset],
                "mom_6m": mom_6m[asset],
                "mom_12m": mom_12m[asset],
                "mom_12_1": mom_12_1[asset],
                "vol_21d": vol21_me[asset],
                "vol_63d": vol63_me[asset],
                "mdd_12m": mdd_me[asset],
                "dist_ma50": dist_ma50[asset],
                "dist_ma200": dist_ma200[asset],
                "rsi_14": rsi_me[asset],
                "macd_hist": macd_me[asset],
                "rank_mom_12m": rank_mom12[asset],
                "rank_vol_21d": rank_vol21[asset],
                "rank_mom_1m": rank_mom1[asset],
                "mkt_ret_1m": mkt_ret_1m,
                # Previous month's realised return of *this* asset (lagged target, not y).
                "lag_target_ret_1m": mom_1m[asset],
                "y": y[asset],
            },
            index=month_ends,
        )
        if vix_chg is not None:
            row["vix_chg_1m"] = vix_chg
        if tnx_chg is not None:
            row["tnx_chg_1m"] = tnx_chg
        row["asset"] = asset
        frames.append(row)

    panel = pd.concat(frames, axis=0)
    panel.index.name = "date"
    panel = panel.reset_index()

    feature_cols = [c for c in panel.columns if c not in {"date", "asset", "y"}]
    # Drop rows that are missing required (non-macro) features. Macro NaNs drop the
    # optional columns later only if an entire series is missing.
    required = [c for c in REQUIRED_FEATURES if c in panel.columns]
    panel = panel.dropna(subset=required).copy()
    # Keep rows with y NaN only if we want a last-month score; drop them for training.
    panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)

    monthly_returns = y.loc[month_ends, inv]
    return FeatureBundle(
        panel=panel,
        feature_cols=feature_cols,
        monthly_returns=monthly_returns,
        month_end_prices=px_me,
        daily_returns=daily_ret,
        month_ends=month_ends,
    )


def save_panel(bundle: FeatureBundle, root) -> None:
    from pathlib import Path

    proc = Path(root) / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    bundle.panel.to_csv(proc / "panel.csv", index=False)
    bundle.monthly_returns.to_csv(proc / "monthly_returns.csv")
    try:
        bundle.panel.to_parquet(proc / "panel.parquet", index=False)
        bundle.monthly_returns.to_parquet(proc / "monthly_returns.parquet")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] panel parquet failed ({exc})", flush=True)
