"""Run the full ranking experiment and write tables/figures from the backtest."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.backtest import (  # noqa: E402
    align_equal_weight_weights,
    cost_bps_series,
    run_backtest,
)
from src.data_loader import assemble_prices, load_config, save_prices  # noqa: E402
from src.features import build_feature_panel, save_panel  # noqa: E402
from src.metrics import subperiod_table, summarise_result  # noqa: E402
from src.models import (  # noqa: E402
    gain_importance,
    momentum_scores,
    save_last_booster,
    walk_forward,
)
from src.portfolio import (  # noqa: E402
    mv_from_means,
    sixty_forty,
    trailing_mean_monthly,
    weights_from_scores,
)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _oos_equal_weight(monthly_returns: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    return align_equal_weight_weights(monthly_returns, dates)


def _oos_sixty_forty(
    monthly_returns: pd.DataFrame,
    dates: pd.Index,
    *,
    bond: str,
    bond_fallback: str,
    risky: list[str],
    bond_weight: float,
) -> pd.DataFrame:
    rows = []
    for t in dates:
        if t not in monthly_returns.index:
            continue
        r = monthly_returns.loc[t]
        available = r.replace([np.inf, -np.inf], np.nan).dropna().index.tolist()
        w = sixty_forty(
            available,
            bond=bond,
            risky=[a for a in risky if a in available],
            bond_weight=bond_weight,
            bond_fallback=bond_fallback,
        )
        w = w.reindex(monthly_returns.columns).fillna(0.0)
        w.name = t
        rows.append(w)
    return pd.DataFrame(rows).sort_index() if rows else pd.DataFrame()


def _hist_mv_weights(
    monthly_returns: pd.DataFrame,
    daily_returns: pd.DataFrame,
    dates: pd.Index,
    *,
    max_weight: float,
    gamma: float,
) -> pd.DataFrame:
    means = []
    for t in dates:
        mu = trailing_mean_monthly(monthly_returns, pd.Timestamp(t), list(monthly_returns.columns))
        mu.name = t
        means.append(mu)
    if not means:
        return pd.DataFrame()
    mean_df = pd.DataFrame(means).sort_index()
    return mv_from_means(mean_df, daily_returns, max_weight=max_weight, gamma=gamma)


def _plot_equity(
    series: dict[str, pd.Series],
    path: Path,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for name, eq in series.items():
        if eq is None or eq.empty:
            continue
        ax.plot(eq.index, eq.values, label=name, linewidth=1.4)
    ax.set_ylabel("Wealth (start = 1)")
    ax.set_xlabel("Decision date (return realised over the next month)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_importance(imp: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    data = imp.sort_values("gain", ascending=True)
    ax.barh(data["feature"], data["gain"], color="#3b6d9a")
    ax.set_xlabel("Gain")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _try_shap(result, path: Path, notes: list[str]) -> None:
    if result.last_model is None or result.last_X_train.empty:
        notes.append("SHAP skipped: no last-fold model.")
        return
    try:
        import shap
        from xgboost import XGBRegressor
    except Exception as exc:  # noqa: BLE001
        notes.append(f"SHAP skipped: import failed ({exc}).")
        return
    model = result.last_model
    if not isinstance(model, XGBRegressor):
        notes.append("SHAP skipped: last model is not XGBRegressor.")
        return
    try:
        X = result.last_X_train
        # Bound SHAP cost on very large last windows.
        if len(X) > 2500:
            X = X.iloc[-2500:]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        plt.figure(figsize=(8.5, 5.5))
        shap.summary_plot(shap_values, X, show=False, max_display=18)
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=140, bbox_inches="tight")
        plt.close()
        notes.append(f"SHAP summary written to {path}.")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"SHAP skipped: {type(exc).__name__}: {exc}")
        traceback.print_exc()


def main(argv: list[str] | None = None) -> int:
    del argv
    cfg = load_config(ROOT / "configs" / "universe.yaml")
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    tables = ROOT / "results" / "tables"
    figures = ROOT / "results" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    print("=== download ===", flush=True)
    prices, meta = assemble_prices(cfg, ROOT)
    save_prices(prices, ROOT)
    pd.DataFrame(
        [{"ticker": k, "source": v} for k, v in meta.sources.items()]
    ).to_csv(tables / "download_log.csv", index=False)

    print(f"data_mode={meta.data_mode} universe={meta.investable}", flush=True)
    print(f"actual dates {meta.start} -> {meta.end}  n_days={len(prices)}", flush=True)

    print("=== features ===", flush=True)
    bundle = build_feature_panel(prices, meta.investable, meta.macro, meta.crypto)
    save_panel(bundle, ROOT)
    print(
        f"panel rows={len(bundle.panel)} dates={bundle.panel['date'].nunique()} "
        f"assets={bundle.panel['asset'].nunique()}",
        flush=True,
    )
    print(f"features={bundle.feature_cols}", flush=True)

    wf = cfg["walk_forward"]
    grid = cfg["xgb_grid"]
    port_cfg = cfg["portfolio"]
    notes = list(meta.notes)

    print("=== walk-forward XGB ===", flush=True)
    xgb_res = walk_forward(
        bundle.panel,
        bundle.feature_cols,
        kind="xgb",
        min_train_months=int(wf["min_train_months"]),
        val_tail_months=int(wf["val_tail_months"]),
        seed=seed,
        grid=grid,
    )
    notes.extend(xgb_res.notes)
    notes.append(f"XGB engine={xgb_res.engine} last_params={xgb_res.last_params}")
    save_note = save_last_booster(xgb_res, ROOT / "models" / "xgb_last.json")
    notes.append(save_note)
    print(save_note, flush=True)

    print("=== walk-forward Ridge ===", flush=True)
    ridge_res = walk_forward(
        bundle.panel,
        bundle.feature_cols,
        kind="ridge",
        min_train_months=int(wf["min_train_months"]),
        val_tail_months=0,
        seed=seed,
    )
    print("=== walk-forward RandomForest ===", flush=True)
    rf_res = walk_forward(
        bundle.panel,
        bundle.feature_cols,
        kind="rf",
        min_train_months=int(wf["min_train_months"]),
        val_tail_months=0,
        seed=seed,
    )
    mom_scores = momentum_scores(bundle.panel)

    oos_index = xgb_res.predictions.index
    if len(oos_index) == 0:
        notes.append("XGB produced no OOS predictions; aborting backtest.")
        _write_json(tables / "run_meta.json", {**meta.to_dict(), "notes": notes})
        print("No OOS predictions.", flush=True)
        return 1

    # Restrict momentum / other scores to the XGB OOS window for a common sample.
    def clip(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(oos_index)

    xgb_sc = clip(xgb_res.predictions)
    ridge_sc = clip(ridge_res.predictions)
    rf_sc = clip(rf_res.predictions)
    mom_sc = clip(mom_scores)
    r_m = bundle.monthly_returns.reindex(oos_index)

    top_k = int(port_cfg["top_k"])
    top_frac = float(port_cfg["top_frac"])
    max_w = float(port_cfg["max_weight"])
    gamma = float(port_cfg.get("mv_risk_aversion", 5.0))

    print("=== portfolios ===", flush=True)
    weights: dict[str, pd.DataFrame] = {}
    weights["xgb_top5_ew"] = weights_from_scores(xgb_sc, "top5", top_k=top_k)
    weights["xgb_top30_sw"] = weights_from_scores(xgb_sc, "top30", top_frac=top_frac)
    weights["xgb_ls"] = weights_from_scores(xgb_sc, "ls", top_k=top_k)
    weights["xgb_mv"] = mv_from_means(xgb_sc, bundle.daily_returns, max_weight=max_w, gamma=gamma)
    weights["ridge_top5_ew"] = weights_from_scores(ridge_sc, "top5", top_k=top_k)
    weights["rf_top5_ew"] = weights_from_scores(rf_sc, "top5", top_k=top_k)
    weights["mom_top5_ew"] = weights_from_scores(mom_sc, "top5", top_k=top_k)
    weights["equal_weight"] = _oos_equal_weight(r_m, oos_index)
    weights["hist_mv"] = _hist_mv_weights(
        bundle.monthly_returns, bundle.daily_returns, oos_index, max_weight=max_w, gamma=gamma
    )

    groups = cfg["investable"]
    risky = list(groups.get("us_large_cap", [])) + list(groups.get("sector_etfs", []))
    risky += list(groups.get("gold", [])) + list(groups.get("crypto", []))
    sf = cfg["sixty_forty"]
    weights["sixty_forty"] = _oos_sixty_forty(
        r_m,
        oos_index,
        bond=str(sf["bond"]),
        bond_fallback=str(sf.get("bond_fallback", "IEF")),
        risky=risky,
        bond_weight=float(sf["bond_weight"]),
    )
    notes.append(
        "60/40 composition: 60% equal-weight of available names in "
        f"{risky} and 40% in {sf['bond']} (fallback {sf.get('bond_fallback')})."
    )

    bps = cost_bps_series(
        meta.investable,
        meta.crypto,
        default_bps=float(cfg["costs"]["default_bps"]),
        crypto_bps=float(cfg["costs"]["crypto_bps"]),
    )
    start_w = float(port_cfg.get("start_wealth", 1.0))

    print("=== backtest ===", flush=True)
    net_results = {}
    gross_results = {}
    for name, w in weights.items():
        if w is None or w.empty:
            notes.append(f"strategy {name} produced no weights.")
            continue
        w.to_csv(tables / f"weights_{name}.csv")
        net_results[name] = run_backtest(w, r_m, bps, name=name, start_wealth=start_w, charge_costs=True)
        gross_results[name] = run_backtest(
            w, r_m, bps, name=name, start_wealth=start_w, charge_costs=False
        )
        net_results[name].monthly.to_csv(tables / f"monthly_{name}.csv")

    rows = [
        summarise_result(net_results[k], gross_results.get(k))
        for k in net_results
    ]
    perf = pd.DataFrame(rows)
    cost_note = (
        f"one-way costs {cfg['costs']['default_bps']} bps stocks/ETFs, "
        f"{cfg['costs']['crypto_bps']} bps crypto; "
        f"OOS {oos_index.min().date()} to {oos_index.max().date()}; "
        f"data_mode={meta.data_mode}; prices {meta.start} to {meta.end}"
    )
    perf.insert(1, "assumptions", cost_note)
    perf.to_csv(tables / "performance.csv", index=False)
    print(perf.to_string(index=False), flush=True)

    periods = [
        ("2017-2019", "2017-01-01", "2019-12-31"),
        ("2020", "2020-01-01", "2020-12-31"),
        ("2021", "2021-01-01", "2021-12-31"),
        ("2022", "2022-01-01", "2022-12-31"),
        ("2023-2026", "2023-01-01", "2026-12-31"),
    ]
    sub = subperiod_table(net_results, periods)
    if not sub.empty:
        sub.insert(1, "assumptions", cost_note)
        sub.to_csv(tables / "subperiods.csv", index=False)

    # Feature importance
    imp = gain_importance(xgb_res)
    imp.to_csv(tables / "feature_importance.csv", index=False)
    if not imp.empty and imp["gain"].notna().any():
        _plot_importance(
            imp,
            figures / "feature_importance.png",
            f"XGB last-fold gain ({meta.data_mode}; last params {xgb_res.last_params})",
        )

    _try_shap(xgb_res, figures / "shap_summary.png", notes)

    plot_keys = [
        ("equal_weight", "Equal-weight"),
        ("sixty_forty", "60/40"),
        ("mom_top5_ew", "Momentum top-5 EW"),
        ("hist_mv", "Historical MV"),
        ("xgb_top5_ew", "XGB top-5 EW"),
    ]
    eq = {}
    for key, label in plot_keys:
        if key in net_results and not net_results[key].equity.empty:
            eq[label] = net_results[key].equity
    _plot_equity(
        eq,
        figures / "equity_curves.png",
        f"Net wealth after costs ({cost_note})",
    )

    extra = {}
    for key, label in [
        ("xgb_top5_ew", "XGB top-5"),
        ("xgb_top30_sw", "XGB top-30% SW"),
        ("xgb_ls", "XGB long-short"),
        ("xgb_mv", "XGB MV"),
        ("ridge_top5_ew", "Ridge top-5"),
        ("rf_top5_ew", "RF top-5"),
    ]:
        if key in net_results and not net_results[key].equity.empty:
            extra[label] = net_results[key].equity
    if extra:
        _plot_equity(
            extra,
            figures / "equity_curves_xgb_family.png",
            f"XGB-family and ML baselines, net of costs ({meta.data_mode})",
        )

    xgb_sc.to_csv(tables / "predictions_xgb.csv")
    ridge_sc.to_csv(tables / "predictions_ridge.csv")
    rf_sc.to_csv(tables / "predictions_rf.csv")

    universe_df = pd.DataFrame(
        {
            "ticker": meta.investable,
            "role": [
                "crypto" if t in meta.crypto else "investable" for t in meta.investable
            ],
        }
    )
    universe_df.to_csv(tables / "universe.csv", index=False)

    # Honest one-line comparison used by the report.
    def _sharpe(name: str) -> float:
        if name not in net_results:
            return float("nan")
        row = perf.loc[perf["strategy"] == name]
        if row.empty:
            return float("nan")
        return float(row["sharpe"].iloc[0])

    comparison = {
        "xgb_top5_ew_sharpe_net": _sharpe("xgb_top5_ew"),
        "equal_weight_sharpe_net": _sharpe("equal_weight"),
        "mom_top5_ew_sharpe_net": _sharpe("mom_top5_ew"),
        "hist_mv_sharpe_net": _sharpe("hist_mv"),
        "sixty_forty_sharpe_net": _sharpe("sixty_forty"),
    }
    _write_json(
        tables / "run_meta.json",
        {
            **meta.to_dict(),
            "oos_start": str(pd.Timestamp(oos_index.min()).date()),
            "oos_end": str(pd.Timestamp(oos_index.max()).date()),
            "n_oos_months": int(len(oos_index)),
            "feature_cols": bundle.feature_cols,
            "xgb_engine": xgb_res.engine,
            "xgb_last_params": xgb_res.last_params,
            "comparison": comparison,
            "notes": notes,
            "cost_note": cost_note,
        },
    )
    print("=== done ===", flush=True)
    print("\n".join(notes), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
