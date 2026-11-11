from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import regression_metrics

N_SPLITS = 5
RANDOM_STATE = 42


def _make_models() -> dict:
    return {
        "Persistence": None,  # 基线：用当前值当预测
        "Ridge": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "LightGBM": LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1,
        ),
    }


def cross_validate(
    data: pd.DataFrame,
    feat_cols: list[str],
    horizon: int,
    n_splits: int = N_SPLITS,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """对给定预测步长做患者级 5 折交叉验证。

    Returns
    -------
    (oof_by_model, fold_metrics)
        oof_by_model: 模型名 -> 含 patient/timestamp/y_true/y_pred 的 OOF 表
        fold_metrics: 每折每模型的指标
    """
    ycol = f"y_{horizon}"
    d = data.dropna(subset=[ycol]).copy()
    # 目标值必须是真实测量（排除插值点）
    d = d[d["glucose"].notna()]

    X = d[feat_cols]
    y = d[ycol].to_numpy()
    groups = d["patient"].to_numpy()

    models = _make_models()
    oof = {name: [] for name in models}
    fold_rows = []

    gkf = GroupKFold(n_splits=min(n_splits, d["patient"].nunique()))
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        for name, model in models.items():
            if name == "Persistence":
                pred = d["g"].to_numpy()[te]
            else:
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)

            m = regression_metrics(y_te, pred)
            m.update({"model": name, "fold": fold, "horizon_steps": horizon,
                      "n_test_patients": int(pd.Series(groups[te]).nunique())})
            fold_rows.append(m)

            # timestamp 记录「预警发出时刻」，target_time 是预测所指向的时刻，
            # 二者相差 horizon，评估时必须区分，否则提前量会算错
            ts = d["timestamp"].to_numpy()[te].astype("datetime64[ns]")
            oof[name].append(pd.DataFrame({
                "patient": groups[te],
                "segment": d["segment"].to_numpy()[te],
                "timestamp": ts,
                "target_time": ts + pd.Timedelta(minutes=horizon * 15),
                "y_true": y_te,
                "y_pred": pred,
                "fold": fold,
            }))

    oof_by_model = {k: pd.concat(v, ignore_index=True) for k, v in oof.items()}
    return oof_by_model, pd.DataFrame(fold_rows)


def fit_final_model(data: pd.DataFrame, feat_cols: list[str], horizon: int):
    """在全量数据上训练最终 LightGBM，返回模型与特征重要性。"""
    ycol = f"y_{horizon}"
    d = data.dropna(subset=[ycol])
    d = d[d["glucose"].notna()]
    model = LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1,
    )
    model.fit(d[feat_cols], d[ycol])
    imp = (pd.Series(model.feature_importances_, index=feat_cols)
             .sort_values(ascending=False))
    return model, imp