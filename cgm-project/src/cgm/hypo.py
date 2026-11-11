from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupKFold

from .metrics import HYPO_THRESHOLD, find_hypo_events

RANDOM_STATE = 42


def train_hypo_classifier(data: pd.DataFrame, feat_cols: list[str],
                          horizon: int, n_splits: int = 5) -> pd.DataFrame:
    """患者级 CV 训练低血糖分类器，返回带 OOF 概率的表。"""
    ycol = f"y_{horizon}"
    d = data.dropna(subset=[ycol]).copy()
    d = d[d["glucose"].notna()].reset_index(drop=True)
    d["y_hypo"] = (d[ycol] < HYPO_THRESHOLD).astype(int)

    X, y = d[feat_cols], d["y_hypo"].to_numpy()
    groups = d["patient"].to_numpy()
    oof_prob = np.full(len(d), np.nan)

    gkf = GroupKFold(n_splits=min(n_splits, d["patient"].nunique()))
    for tr, te in gkf.split(X, y, groups):
        clf = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=4.0,
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1,
        )
        clf.fit(X.iloc[tr], y[tr])
        oof_prob[te] = clf.predict_proba(X.iloc[te])[:, 1]

    out = d[["patient", "segment", "timestamp", ycol, "y_hypo"]].copy()
    out["target_time"] = out["timestamp"] + pd.Timedelta(minutes=horizon * 15)
    out["y_true"] = out[ycol]
    out["prob"] = oof_prob
    return out


def threshold_sweep(df: pd.DataFrame, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """扫描报警阈值，给出行级精确率/召回率与事件级召回。"""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    y = df["y_hypo"].to_numpy()
    for t in thresholds:
        pred = (df["prob"] >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        # 事件级召回：事件开始前窗口内是否出现概率超阈值
        hit, total = 0, 0
        for _, g in df.groupby(["patient", "segment"]):
            g = g.sort_values("target_time")
            ev = find_hypo_events(g["target_time"], g["y_true"])
            if ev.empty:
                continue
            total += len(ev)
            alarms = g[g["prob"] >= t]
            for _, e in ev.iterrows():
                lo = e["start"] - pd.Timedelta(minutes=45)
                hi = e["start"] - pd.Timedelta(minutes=15)
                if not alarms[(alarms["timestamp"] >= lo) & (alarms["timestamp"] <= hi)].empty:
                    hit += 1

        rows.append({
            "threshold": t,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "event_recall": round(hit / total, 3) if total else np.nan,
            "alarms": int(pred.sum()),
        })
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> dict:
    """整体判别能力。"""
    y = df["y_hypo"].to_numpy()
    p = df["prob"].to_numpy()
    prec, rec, _ = precision_recall_curve(y, p)
    from sklearn.metrics import roc_auc_score
    return {
        "positive_rate": round(float(y.mean()), 4),
        "AUPRC": round(float(average_precision_score(y, p)), 4),
        "AUROC": round(float(roc_auc_score(y, p)), 4),
        "pr_curve": (prec[::max(len(prec) // 50, 1)], rec[::max(len(rec) // 50, 1)]),
    }