# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

SAMPLE_MINUTES = 15
SENSOR_MIN, SENSOR_MAX = 40.0, 500.0
MAX_PHYSIO_ROC = 4.0


def segment_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (patient, seg), g in df.groupby(["patient", "segment"], sort=False):
        g = g.sort_values("timestamp")
        n = len(g)
        dur_h = (g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 3600
        expected = max(int(dur_h * 60 / SAMPLE_MINUTES) + 1, 1)

        dt = g["timestamp"].diff().dt.total_seconds().div(60).dropna()
        roc = g["glucose"].diff().abs().div(dt.replace(0, np.nan)) if len(g) > 1 else pd.Series(dtype=float)

        out_of_range = int(((g["glucose"] < SENSOR_MIN) | (g["glucose"] > SENSOR_MAX)).sum())
        fast = int((roc > MAX_PHYSIO_ROC).sum()) if len(roc) else 0

        rows.append({
            "patient": patient,
            "segment": seg,
            "n_points": n,
            "duration_h": round(dur_h, 2),
            "expected_points": expected,
            "completeness": round(min(n / expected, 1.0), 3),
            "median_dt_min": float(dt.median()) if len(dt) else np.nan,
            "out_of_range": out_of_range,
            "fast_change": fast,
            "glucose_mean": round(float(g["glucose"].mean()), 1),
            "glucose_cv": round(float(g["glucose"].std() / max(g["glucose"].mean(), 1e-6)), 3),
        })
    return pd.DataFrame(rows).sort_values(["patient", "segment"]).reset_index(drop=True)


def detect_compression_low(df, nadir=90.0, drop_rate=1.5, recover_rate=1.5, window_min=90):
    events = []
    for (patient, seg), g in df.groupby(["patient", "segment"], sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < 8:
            continue
        ts = g["timestamp"]
        val = g["glucose"].to_numpy()
        dt = ts.diff().dt.total_seconds().div(60).fillna(SAMPLE_MINUTES).to_numpy()

        for i in range(2, len(g) - 2):
            if val[i] >= nadir:
                continue
            lo = max(0, i - 6)
            hi = min(len(g), i + 7)
            before, after = val[lo:i + 1], val[i:hi]
            if len(before) < 3 or len(after) < 3:
                continue
            d_rate = (before[0] - val[i]) / max(np.nansum(dt[lo:i + 1]), 1)
            r_rate = (after[-1] - val[i]) / max(np.nansum(dt[i:hi]), 1)
            if d_rate >= drop_rate and r_rate >= recover_rate:
                events.append({
                    "patient": patient, "segment": seg, "timestamp": ts.iloc[i],
                    "nadir": float(val[i]), "drop_rate": round(float(d_rate), 2),
                    "recover_rate": round(float(r_rate), 2), "hour": int(ts.iloc[i].hour),
                })
    # 强制返回 DataFrame，即使为空
    return pd.DataFrame(events)


def quality_summary(seg_q: pd.DataFrame, comp) -> pd.DataFrame:
    # 【最强防御】：如果传入的 comp 不是表格，强制转换，防止报错
    if not isinstance(comp, pd.DataFrame):
        comp = pd.DataFrame(comp)

    agg = seg_q.groupby("patient").agg(
        segments=("segment", "nunique"),
        valid_segments=("n_points", lambda s: int((s >= 48).sum())),
        points=("n_points", "sum"),
        completeness=("completeness", "mean"),
        out_of_range=("out_of_range", "sum"),
        fast_change=("fast_change", "sum"),
        glucose_mean=("glucose_mean", "mean"),
        glucose_cv=("glucose_cv", "mean"),
    ).reset_index()

    agg["completeness"] = agg["completeness"].round(3)
    agg["glucose_mean"] = agg["glucose_mean"].round(1)
    agg["glucose_cv"] = agg["glucose_cv"].round(3)

    # 【防错判断】：确保安全访问 empty
    if not comp.empty and "patient" in comp.columns:
        n_comp = comp.groupby("patient").size().rename("compression_low_events")
        agg = agg.merge(n_comp, on="patient", how="left")
    else:
        agg["compression_low_events"] = 0

    return agg.fillna({"compression_low_events": 0})