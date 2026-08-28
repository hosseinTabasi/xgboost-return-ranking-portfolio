"""Walk-forward expanding-window return models.

Train rows use decision dates strictly before t. The target of a training
row at s is the return from s to s+1, which is known by t whenever s < t.
Hyper-parameters for the gradient booster are chosen on the last 12 months
of the training window (MSE) and the winner is refit on the full train set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelKind = Literal["xgb", "ridge", "rf", "hgb"]

XGB_AVAILABLE = True
XGB_IMPORT_ERROR = ""
try:
    from xgboost import XGBRegressor
except Exception as exc:  # noqa: BLE001
    XGB_AVAILABLE = False
    XGB_IMPORT_ERROR = str(exc)
    XGBRegressor = None  # type: ignore[assignment,misc]


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame  # date x asset
    last_model: Any
    last_feature_names: list[str]
    last_X_train: pd.DataFrame
    last_params: dict[str, Any]
    engine: str  # "xgboost" or "hist_gradient_boosting" or model kind
    notes: list[str] = field(default_factory=list)


def _xgb_estimator(params: dict[str, Any], seed: int):
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def _hgb_estimator(params: dict[str, Any], seed: int):
    return HistGradientBoostingRegressor(
        max_iter=int(params.get("n_estimators", 200)),
        max_depth=int(params.get("max_depth", 3)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        random_state=seed,
    )


def _grid_fit(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    grid: dict[str, list],
    seed: int,
    use_xgb: bool,
) -> tuple[Any, dict[str, Any]]:
    combos = list(
        product(grid["n_estimators"], grid["max_depth"], grid["learning_rate"])
    )
    best_mse = float("inf")
    best_params = {
        "n_estimators": grid["n_estimators"][0],
        "max_depth": grid["max_depth"][0],
        "learning_rate": grid["learning_rate"][0],
    }
    if len(y_val) < 10:
        params = best_params
        model = _xgb_estimator(params, seed) if use_xgb else _hgb_estimator(params, seed)
        model.fit(X_tr, y_tr)
        return model, params
    for n_est, depth, lr in combos:
        params = {"n_estimators": n_est, "max_depth": depth, "learning_rate": lr}
        model = _xgb_estimator(params, seed) if use_xgb else _hgb_estimator(params, seed)
        model.fit(X_tr, y_tr)
        pred = np.asarray(model.predict(X_val), dtype=float)
        mse = float(mean_squared_error(y_val, pred))
        if mse < best_mse:
            best_mse = mse
            best_params = params
    # Refit on train+val happens in the caller.
    return None, best_params


def _fit_predict_kind(
    kind: ModelKind,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    seed: int,
    grid: dict[str, list],
    val_tail_months: int,
    train_dates: pd.Series,
) -> tuple[np.ndarray, Any, dict[str, Any], str]:
    notes_engine = kind
    last_params: dict[str, Any] = {}
    if kind == "ridge":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0, random_state=seed)),
            ]
        )
        model.fit(X_tr, y_tr)
        return np.asarray(model.predict(X_te), dtype=float), model, last_params, notes_engine
    if kind == "rf":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        )
        model.fit(X_tr, y_tr)
        return np.asarray(model.predict(X_te), dtype=float), model, last_params, notes_engine

    use_xgb = kind == "xgb" and XGB_AVAILABLE
    engine = "xgboost" if use_xgb else "hist_gradient_boosting"
    unique_dates = pd.Index(train_dates.unique()).sort_values()
    if val_tail_months > 0 and len(unique_dates) > val_tail_months + 6:
        cut = unique_dates[-val_tail_months]
        inner = train_dates < cut
        val = ~inner
        if inner.sum() >= 50 and val.sum() >= 10:
            _, last_params = _grid_fit(
                X_tr.loc[inner],
                y_tr[inner.to_numpy()],
                X_tr.loc[val],
                y_tr[val.to_numpy()],
                grid,
                seed,
                use_xgb,
            )
        else:
            last_params = {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.05,
            }
    else:
        last_params = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
        }
    model = _xgb_estimator(last_params, seed) if use_xgb else _hgb_estimator(last_params, seed)
    model.fit(X_tr, y_tr)
    return np.asarray(model.predict(X_te), dtype=float), model, last_params, engine


def walk_forward(
    panel: pd.DataFrame,
    feature_cols: list[str],
    *,
    kind: ModelKind,
    min_train_months: int = 36,
    val_tail_months: int = 12,
    seed: int = 42,
    grid: dict[str, list] | None = None,
) -> WalkForwardResult:
    grid = grid or {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }
    work = panel.dropna(subset=["y"] + [c for c in feature_cols if c in panel.columns]).copy()
    feat = [c for c in feature_cols if c in work.columns]
    dates = pd.Index(sorted(work["date"].unique()))
    pred_rows: list[dict[str, Any]] = []
    last_model = None
    last_X = pd.DataFrame()
    last_params: dict[str, Any] = {}
    engine = kind
    notes: list[str] = []
    if kind == "xgb" and not XGB_AVAILABLE:
        notes.append(
            f"xgboost import failed ({XGB_IMPORT_ERROR}); using HistGradientBoostingRegressor."
        )

    for i, t in enumerate(dates):
        train = work[work["date"] < t]
        n_months = train["date"].nunique()
        if n_months < min_train_months:
            continue
        test = work[work["date"] == t]
        if test.empty:
            continue
        X_tr = train[feat]
        y_tr = train["y"].to_numpy(dtype=float)
        X_te = test[feat]
        preds, model, params, engine = _fit_predict_kind(
            kind,
            X_tr,
            y_tr,
            X_te,
            seed,
            grid,
            val_tail_months,
            train["date"],
        )
        last_model = model
        last_X = X_tr
        last_params = params
        for asset, yhat in zip(test["asset"].tolist(), preds, strict=True):
            pred_rows.append({"date": t, "asset": asset, "yhat": float(yhat)})
        if (i + 1) % 12 == 0:
            print(f"  [{kind}] fold date={pd.Timestamp(t).date()} train_months={n_months}", flush=True)

    pred_long = pd.DataFrame(pred_rows)
    if pred_long.empty:
        wide = pd.DataFrame()
    else:
        wide = pred_long.pivot(index="date", columns="asset", values="yhat").sort_index()
    return WalkForwardResult(
        predictions=wide,
        last_model=last_model,
        last_feature_names=feat,
        last_X_train=last_X,
        last_params=last_params,
        engine=engine,
        notes=notes,
    )


def momentum_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """12-1 momentum if present, else 12m momentum. No fitting."""
    col = "mom_12_1" if "mom_12_1" in panel.columns else "mom_12m"
    sub = panel[["date", "asset", col]].dropna()
    return sub.pivot(index="date", columns="asset", values=col).sort_index()


def save_last_booster(result: WalkForwardResult, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = result.last_model
    if model is None:
        return "no model"
    if XGB_AVAILABLE and XGBRegressor is not None and isinstance(model, XGBRegressor):
        model.save_model(str(path))
        return f"xgboost json -> {path}"
    import joblib

    alt = path.with_suffix(".joblib")
    joblib.dump(model, alt)
    return f"joblib -> {alt}"


def gain_importance(result: WalkForwardResult) -> pd.DataFrame:
    model = result.last_model
    names = result.last_feature_names
    if model is None:
        return pd.DataFrame(columns=["feature", "gain"])
    if XGB_AVAILABLE and XGBRegressor is not None and isinstance(model, XGBRegressor):
        booster = model.get_booster()
        raw = booster.get_score(importance_type="gain")
        mapped = {k: float(v) for k, v in raw.items()}
        # XGBoost may use f0,f1 or the DataFrame names.
        rows = []
        for i, name in enumerate(names):
            gain = mapped.get(name, mapped.get(f"f{i}", 0.0))
            rows.append({"feature": name, "gain": gain})
        out = pd.DataFrame(rows).sort_values("gain", ascending=False)
        return out
    # sklearn fallback
    est = model
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
        return pd.DataFrame({"feature": names, "gain": imp}).sort_values("gain", ascending=False)
    return pd.DataFrame({"feature": names, "gain": np.nan})
