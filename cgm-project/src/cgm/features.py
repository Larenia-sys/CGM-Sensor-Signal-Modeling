from __future__ import annotations

import numpy as np
import pandas as pd

SAMPLE_MINUTES = 15
MIN_SEGMENT_POINTS = 48  # 至少 12 h 的连续记录才参与建模


def _within_group_rolling(s: pd.Series, window: int, how: str) -> pd.Series:
    """组内滚动统计，先 shift(1) 保证不含当前时刻。"""
    return s.shift(1).rolling(window, min_periods=max(2, window // 2)).agg(how)


def build_features(
    df: pd.DataFrame,
    horizons_steps: tuple[int, ...] = (2, 4),
    use_insulin: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """构造特征矩阵与多步预测目标。

    Parameters
    ----------
    df : 含 patient / segment / timestamp / glucose 的长表
    horizons_steps : 预测步长（步长 × 15 min），2 -> 30 min，4 -> 60 min

    Returns
    -------
    (data, feature_cols)
    """
    d = df.sort_values(["patient", "segment", "timestamp"]).reset_index(drop=True).copy()

    # 只保留足够长的连续段
    seg_len = d.groupby(["patient", "segment"])["glucose"].transform("size")
    d = d[seg_len >= MIN_SEGMENT_POINTS].reset_index(drop=True)

    key = [d["patient"], d["segment"]]
    gcol = "glucose"

    # 特征源：段内线性插值（仅填补 1-2 个点的小空洞），目标仍用真实测量值
    d["g"] = (d.groupby(["patient", "segment"])["glucose"]
                .transform(lambda s: s.interpolate(limit=2, limit_area="inside")))

    grp = d.groupby(["patient", "segment"], sort=False)["g"]

    # --- 滞后特征 ---
    for k in (1, 2, 4, 8):
        d[f"lag_{k}"] = grp.shift(k)

    # --- 差分 / 变化率 ---
    d["diff_1"] = d["g"] - d["lag_1"]
    d["diff_2"] = d["g"] - d["lag_2"]
    d["roc_1"] = d["diff_1"] / SAMPLE_MINUTES          # mg/dL/min
    d["accel"] = d["diff_1"] - d["lag_1"].sub(d["lag_2"])

    # --- 滚动统计（窗口 1h / 2h / 4h）---
    for w in (4, 8, 16):
        d[f"roll_mean_{w}"] = grp.transform(lambda s, w=w: _within_group_rolling(s, w, "mean"))
        d[f"roll_std_{w}"] = grp.transform(lambda s, w=w: _within_group_rolling(s, w, "std"))
        d[f"roll_min_{w}"] = grp.transform(lambda s, w=w: _within_group_rolling(s, w, "min"))
        d[f"roll_max_{w}"] = grp.transform(lambda s, w=w: _within_group_rolling(s, w, "max"))

    # --- 时间特征（昼夜节律对血糖影响显著）---
    ts = d["timestamp"]
    d["hour"] = ts.dt.hour
    d["weekday"] = ts.dt.weekday
    d["is_night"] = ((ts.dt.hour < 6) | (ts.dt.hour >= 23)).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)

    # --- 事件特征：近 1h / 3h 的碳水与速效胰岛素累积 ---
    if use_insulin and "carbs" in d.columns:
        for col, name in (("carbs", "carbs"), ("rapid_insulin", "insulin")):
            s = d.groupby(["patient", "segment"], sort=False)[col].transform(
                lambda x: x.fillna(0).rolling(4, min_periods=1).sum())
            d[f"{name}_1h"] = s.fillna(0)
            s3 = d.groupby(["patient", "segment"], sort=False)[col].transform(
                lambda x: x.fillna(0).rolling(12, min_periods=1).sum())
            d[f"{name}_3h"] = s3.fillna(0)

    # --- 多步目标 ---
    for h in horizons_steps:
        d[f"y_{h}"] = grp.shift(-h)

    feat_cols = [c for c in d.columns if c not in
                 {"patient", "segment", "timestamp", "glucose", "g", "record_id",
                  "record_type", "source_file", "dt_min", "fingerstick",
                  "rapid_insulin", "carbs"}
                 and not c.startswith("y_")]
    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(d[c])]

    return d, feat_cols