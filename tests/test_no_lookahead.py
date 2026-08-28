"""Target is a one-month shift; features at t cannot include next-month return."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import make_toy_prices
from src.features import REQUIRED_FEATURES, build_feature_panel


INV = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "XOM", "NVDA", "GLD", "TLT", "BTC-USD"]
CRYPTO = ["BTC-USD"]
MACRO = ["^VIX", "^TNX"]


def _bundle():
    px = make_toy_prices(INV, MACRO, CRYPTO, "2016-01-01", "2022-12-31", seed=42)
    return build_feature_panel(px, INV, MACRO, CRYPTO), px


def test_target_is_next_month_return() -> None:
    bundle, _ = _bundle()
    panel = bundle.panel
    px_me = bundle.month_end_prices
    sample = panel.dropna(subset=["y"]).head(40)
    for _, row in sample.iterrows():
        t = row["date"]
        asset = row["asset"]
        # Next month-end after t.
        future = px_me.index[px_me.index > t]
        if future.empty:
            continue
        t1 = future[0]
        p0 = float(px_me.loc[t, asset])
        p1 = float(px_me.loc[t1, asset])
        expected = p1 / p0 - 1.0
        assert np.isfinite(row["y"])
        assert abs(float(row["y"]) - expected) < 1e-10


def test_lag_target_is_previous_month_not_y() -> None:
    panel = _bundle()[0].panel.dropna(subset=["y", "lag_target_ret_1m"])
    # They must not be identical on the same row (except by chance).
    corr = float(panel["lag_target_ret_1m"].corr(panel["y"]))
    assert abs(corr) < 0.85
    # lag_target equals mom_1m by construction.
    assert np.allclose(panel["lag_target_ret_1m"], panel["mom_1m"], equal_nan=True)


def test_no_feature_equals_target() -> None:
    panel = _bundle()[0].panel.dropna(subset=["y"])
    y = panel["y"].to_numpy(dtype=float)
    for col in REQUIRED_FEATURES:
        if col not in panel.columns:
            continue
        x = panel[col].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        assert mask.sum() > 20
        assert not np.allclose(x[mask], y[mask], atol=1e-8, rtol=1e-5), col
        if np.std(x[mask]) > 0 and np.std(y[mask]) > 0:
            corr = abs(float(np.corrcoef(x[mask], y[mask])[0, 1]))
            assert corr < 0.95, f"{col} corr with y is {corr}"


def test_mutating_future_prices_does_not_change_features_at_t() -> None:
    px = make_toy_prices(INV, MACRO, CRYPTO, "2016-01-01", "2021-12-31", seed=7)
    bundle = build_feature_panel(px, INV, MACRO, CRYPTO)
    panel = bundle.panel.dropna(subset=["y"]).sort_values("date")
    t = panel["date"].iloc[len(panel) // 2]
    feat_cols = [c for c in bundle.feature_cols if c in panel.columns]
    before = panel.loc[panel["date"] == t, ["asset"] + feat_cols].set_index("asset").sort_index()

    px2 = px.copy()
    future = px2.index > t
    # Break the future path; features at t must be invariant.
    px2.loc[future] = px2.loc[future] * 3.7 + 50.0
    bundle2 = build_feature_panel(px2, INV, MACRO, CRYPTO)
    after = (
        bundle2.panel.loc[bundle2.panel["date"] == t, ["asset"] + feat_cols]
        .set_index("asset")
        .sort_index()
    )
    common = before.index.intersection(after.index)
    assert len(common) >= 5
    # Drop rank columns that can shift if future NaN patterns change the month-end calendar.
    compare_cols = [c for c in feat_cols if not c.startswith("rank_")]
    delta = (before.loc[common, compare_cols] - after.loc[common, compare_cols]).abs().max().max()
    assert delta < 1e-9, delta
