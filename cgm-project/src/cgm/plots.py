from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import clarke_zones


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_signal_trace(df: pd.DataFrame, patient: str, path: Path):
    """单个患者的连续血糖轨迹，标注段断裂。"""
    g = df[df["patient"] == patient].sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(g["timestamp"], g["glucose"], lw=0.8, color="#1f3f6e")
    ax.axhline(70, ls="--", lw=0.9, color="#c0392b", label="Hypoglycemia (70 mg/dL)")
    ax.axhline(180, ls="--", lw=0.9, color="#d68910", label="Hyperglycemia (180 mg/dL)")
    for _, s in g.groupby("segment"):
        if len(s) > 1:
            ax.axvspan(s["timestamp"].min(), s["timestamp"].max(), alpha=0.05,
                       color="#2e86c1" if s["segment"].iloc[0] % 2 == 0 else "#aab7b8")
    ax.set_ylim(35, 420)
    ax.set_xlabel("Time")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title(f"CGM signal trace — {patient} (shaded bands = continuous segments)")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.autofmt_xdate()
    _save(fig, path)


def plot_quality(seg_q: pd.DataFrame, path: Path):
    """段落质量：完整度 vs 血糖变异系数，点大小 = 时长。"""
    d = seg_q[seg_q["n_points"] >= 10]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    sc = axes[0].scatter(d["completeness"], d["glucose_cv"],
                         s=np.clip(d["duration_h"], 4, 120),
                         c=d["fast_change"], cmap="viridis", alpha=0.75,
                         edgecolor="k", linewidth=0.3)
    axes[0].set_xlabel("Completeness (observed / expected)")
    axes[0].set_ylabel("Glucose CV")
    axes[0].set_title("Segment quality")
    fig.colorbar(sc, ax=axes[0], label="fast-change points")

    axes[1].hist(seg_q["completeness"], bins=25, color="#5d6d7e", edgecolor="w")
    axes[1].set_xlabel("Completeness")
    axes[1].set_ylabel("Segments")
    axes[1].set_title("Completeness distribution")
    fig.tight_layout()
    _save(fig, path)


def plot_clarke(ref_a, pred_a, ref_b, pred_b, ha: str, hb: str, path: Path):
    """Clarke 误差网格（双 horizon 对比）。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    for ax, ref, pred, title in ((axes[0], ref_a, pred_a, ha), (axes[1], ref_b, pred_b, hb)):
        z = clarke_zones(ref, pred)
        colors = {"A": "#27ae60", "B": "#2e86c1", "C": "#f39c12", "D": "#c0392b", "E": "#7b241c"}
        for zone in "ABCDE":
            m = z == zone
            if m.sum():
                ax.scatter(np.asarray(ref)[m], np.asarray(pred)[m], s=6,
                           alpha=0.45, color=colors[zone],
                           label=f"{zone} ({(m.mean()*100):.1f}%)")
        lim = [30, 420]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Reference glucose (mg/dL)")
        ax.set_ylabel("Predicted glucose (mg/dL)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    _save(fig, path)


def plot_prediction(pred_df: pd.DataFrame, patient: str, horizon_min: int, path: Path):
    """抽样片段的预测曲线对比。"""
    g = pred_df[pred_df["patient"] == patient].sort_values("timestamp")
    g = g.iloc[: 24 * 4 * 3]  # 前 3 天
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(g["timestamp"], g["y_true"], lw=1.1, label="Observed", color="#1f3f6e")
    ax.plot(g["timestamp"], g["y_pred"], lw=1.0, label=f"Predicted (+{horizon_min} min)",
            color="#c0392b", alpha=0.85)
    ax.axhline(70, ls=":", lw=0.8, color="#7f8c8d")
    ax.set_xlabel("Time")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title(f"{horizon_min}-min ahead prediction — {patient}")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    _save(fig, path)


def plot_importance(imp: pd.Series, horizon_min: int, path: Path, top: int = 18):
    fig, ax = plt.subplots(figsize=(6.2, 5))
    s = imp.head(top)[::-1]
    ax.barh(s.index, s.values, color="#2e86c1")
    ax.set_xlabel("LightGBM gain importance")
    ax.set_title(f"Feature importance (+{horizon_min} min)")
    fig.tight_layout()
    _save(fig, path)


def plot_hypo(per_patient: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    d = per_patient.dropna(subset=["event_recall"])
    axes[0].bar(d["patient"], d["event_recall"], color="#c0392b")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Event recall")
    axes[0].set_title("Hypoglycemia event detection by patient")
    axes[0].tick_params(axis="x", rotation=90, labelsize=6)

    axes[1].scatter(d["false_alarm_per_day"], d["event_recall"],
                    s=28, color="#2e86c1", edgecolor="k", linewidth=0.4)
    axes[1].set_xlabel("False alarms per day")
    axes[1].set_ylabel("Event recall")
    axes[1].set_title("Detection vs false alarm trade-off")
    fig.tight_layout()
    _save(fig, path)


def plot_hypo_tradeoff(pr_curve, sweep: pd.DataFrame, path: Path):
    """低血糖检测的 PR 曲线与阈值效应。"""
    prec, rec = pr_curve
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].plot(rec, prec, color="#c0392b", lw=1.6)
    axes[0].set_xlabel("Recall (row level)")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Hypoglycemia detection PR curve (+30 min)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(sweep["threshold"], sweep["recall"], marker="o", ms=3,
                 label="row recall", color="#2e86c1")
    axes[1].plot(sweep["threshold"], sweep["event_recall"], marker="s", ms=3,
                 label="event recall", color="#c0392b")
    axes[1].plot(sweep["threshold"], sweep["precision"], marker="^", ms=3,
                 label="precision", color="#27ae60")
    axes[1].set_xlabel("Alarm threshold (predicted probability)")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Threshold trade-off")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, path)
