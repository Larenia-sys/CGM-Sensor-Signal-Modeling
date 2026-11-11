from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HYPO_THRESHOLD = 70.0  # mg/dL，低血糖临床阈值


def regression_metrics(y_true, y_pred) -> dict:
    """回归指标。MARD 按 CGM 惯例以参考值归一。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {}
    mard = np.mean(np.abs(y_pred - y_true) / np.clip(np.abs(y_true), 1e-6, None)) * 100
    return {
        "n": int(len(y_true)),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "MARD_%": round(float(mard), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def clarke_zones(ref, pred) -> np.ndarray:
    """Clarke 误差网格分区（A/B/C/D/E）。

    规则为常用简化实现：
      A  临床准确（低值区同侧，或与参考值相差 <= 20%）
      B  良性误差
      C  不必要治疗（参考正常而预测偏离至低/高值）
      D  危险漏诊（参考低/高而预测落在正常区间）
      E  完全错误（低值预测为高值，或反之）
    """
    r = np.asarray(ref, dtype=float)
    p = np.asarray(pred, dtype=float)
    zone = np.full(r.shape, "B", dtype="<U1")
    band = np.abs(p - r) <= 0.2 * r

    zone[(r <= HYPO_THRESHOLD) & (p <= HYPO_THRESHOLD)] = "A"
    zone[(r > HYPO_THRESHOLD) & band] = "A"

    zone[(r <= HYPO_THRESHOLD) & (p >= 180)] = "E"
    zone[(r >= 180) & (p <= HYPO_THRESHOLD)] = "E"

    zone[(r <= HYPO_THRESHOLD) & (p > HYPO_THRESHOLD) & (p < 180)] = "D"
    zone[(r >= 180) & (p > HYPO_THRESHOLD) & (p < 180)] = "D"

    normal = (r > HYPO_THRESHOLD) & (r < 180)
    zone[normal & (p <= HYPO_THRESHOLD)] = "C"
    zone[normal & (p >= 180)] = "C"
    return zone


def zone_table(ref, pred) -> pd.DataFrame:
    """A–E 区占比（%），并给出临床可接受率 A+B。"""
    z = clarke_zones(ref, pred)
    counts = pd.Series(z).value_counts(normalize=True).reindex(list("ABCDE")).fillna(0) * 100
    out = counts.round(2).to_frame("pct").T
    out["A+B_%"] = round(float(counts.loc[["A", "B"]].sum()), 2)
    return out


def find_hypo_events(ts: pd.Series, glucose: pd.Series,
                     threshold: float = HYPO_THRESHOLD,
                     min_points: int = 2,
                     merge_gap_min: float = 60.0) -> pd.DataFrame:
    """识别低血糖事件并合并相邻片段。

    参数
    ----
    min_points : 至少连续多少个点低于阈值（2 点 = 30 min）
    merge_gap_min : 两个事件的间隔小于该值则合并，避免血糖在阈值附近
                    抖动被切成大量伪事件（T1DM 患者实际约 1-3 次/天）
    """
    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
    g = pd.Series(glucose).reset_index(drop=True)
    below = (g < threshold).to_numpy()
    events, start = [], None
    for i, flag in enumerate(below):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_points:
                events.append({"start": ts.iloc[start], "end": ts.iloc[i - 1],
                               "nadir": float(g.iloc[start:i].min())})
            start = None
    if start is not None and len(below) - start >= min_points:
        events.append({"start": ts.iloc[start], "end": ts.iloc[-1],
                       "nadir": float(g.iloc[start:].min())})

    if not events:
        return pd.DataFrame(columns=["start", "end", "nadir", "duration_min"])

    merged = [events[0]]
    for ev in events[1:]:
        gap = (ev["start"] - merged[-1]["end"]).total_seconds() / 60
        if gap <= merge_gap_min:
            merged[-1]["end"] = max(merged[-1]["end"], ev["end"])
            merged[-1]["nadir"] = min(merged[-1]["nadir"], ev["nadir"])
        else:
            merged.append(ev)

    out = pd.DataFrame(merged)
    out["duration_min"] = (out["end"] - out["start"]).dt.total_seconds() / 60
    return out


def alarm_performance(pred_df: pd.DataFrame, horizon_min: int,
                      threshold: float = HYPO_THRESHOLD,
                      min_points: int = 2,
                      min_lead_min: float = 15.0,
                      max_lead_min: float | None = None) -> dict:
    """低血糖预警评估。

    时间语义
    --------
    每一行代表一次预警：预警在 ``timestamp`` 时刻发出，指向 ``target_time``
    （= timestamp + horizon）的血糖。因此评估事件召回时，必须在
    ``[T0 - max_lead, T0 - min_lead]`` 区间内寻找预警，其中 T0 是事件
    在 target 时间轴上的开始时刻。

    参数
    ----
    min_lead_min : 至少提前多少分钟发出预警才算有效（默认 15 min）
    max_lead_min : 最多提前多少分钟（默认 horizon + 15）
    """
    if max_lead_min is None:
        max_lead_min = horizon_min + 15

    rows = []
    # 按 患者×segment 分组：segment 之间时间不连续，跨段拼接会造出伪事件
    keys = ["patient", "segment"] if "segment" in pred_df.columns else ["patient"]
    for (patient, *_), g in pred_df.groupby(keys):
        g = g.sort_values("target_time")
        events = find_hypo_events(g["target_time"], g["y_true"], threshold, min_points)

        alarms = g[g["y_pred"] < threshold]
        hit, leads = 0, []
        used = set()
        for _, ev in events.iterrows():
            lo = ev["start"] - pd.Timedelta(minutes=max_lead_min)
            hi = ev["start"] - pd.Timedelta(minutes=min_lead_min)
            cand = alarms[(alarms["timestamp"] >= lo) & (alarms["timestamp"] <= hi)]
            cand = cand[~cand.index.isin(used)]
            if not cand.empty:
                hit += 1
                idx = cand.index[0]
                used.add(idx)
                leads.append((ev["start"] - cand.loc[idx, "timestamp"]).total_seconds() / 60)

        days = max((g["target_time"].max() - g["target_time"].min()).total_seconds() / 86400, 1e-6)
        tp = int(((g["y_pred"] < threshold) & (g["y_true"] < threshold)).sum())
        fp = int(((g["y_pred"] < threshold) & (g["y_true"] >= threshold)).sum())

        rows.append({
            "patient": patient,
            "n_events": len(events),
            "event_recall": round(hit / len(events), 3) if len(events) else np.nan,
            "median_lead_min": round(float(np.median(leads)), 1) if leads else np.nan,
            "alarms": len(alarms),
            "precision": round(tp / max(tp + fp, 1), 3),
            "false_alarm_per_day": round(fp / days, 2),
        })

    out = pd.DataFrame(rows).groupby("patient", as_index=False).agg(
        n_events=("n_events", "sum"),
        event_recall=("event_recall", "mean"),
        median_lead_min=("median_lead_min", "median"),
        alarms=("alarms", "sum"),
        precision=("precision", "mean"),
        false_alarm_per_day=("false_alarm_per_day", "sum"),
    )
    summary = {
        "n_events_total": int(out["n_events"].sum()),
        "event_recall_mean": round(float(out["event_recall"].mean()), 3),
        "median_lead_min": round(float(out["median_lead_min"].median()), 1),
        "precision_mean": round(float(out["precision"].mean()), 3),
        "false_alarm_per_day_mean": round(float(out["false_alarm_per_day"].mean()), 2),
    }
    return {"summary": summary, "per_patient": out}