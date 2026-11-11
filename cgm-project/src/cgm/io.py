from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# LibreView 记录类型编码（Tipo de registro）
RECORD_HISTORICAL = 0  # 15 min 历史血糖，连续监测主序列
RECORD_SCAN = 1        # 用户主动扫描读数

# 列名关键字（按优先级排列：先精确列名，再前缀，再子串）
COLUMN_PATTERNS: dict[str, list[str]] = {
    "record_id": ["ID", "Número de serial"],
    "timestamp": ["Hora", "Sello de tiempo"],
    "record_type": ["Tipo de registro"],
    "glucose": ["Histórico glucosa", "Historial de glucosa"],
    "scan_glucose": ["Glucosa leída", "Escaneo de glucosa"],
    "rapid_insulin": ["Insulina de acción rápida (unidades)"],
    "basal_insulin": ["Insulina de acción lenta (unidades)"],
    "meal_insulin": ["Insulina comida"],
    "correction_insulin": ["Insulina corrección"],
    "carbs": ["Carbohidratos"],
    "fingerstick": ["Glucosa de la tira"],
    "ketones": ["Cetonas"],
    "notes": ["Notas"],
}

SAMPLE_MINUTES = 15


@dataclass
class ParseReport:
    """单文件解析诊断信息。"""

    path: str
    patient: str
    rows_raw: int = 0
    rows_glucose: int = 0
    n_segments: int = 0
    time_jumps: int = 0
    span_days: float = 0.0
    error: str | None = None


def _normalize_text(path: Path) -> str:
    """读取并规范化换行（兼容 CRLF / 早期 Mac CR / UTF-8 BOM）。"""
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")


def _split_head(text: str) -> tuple[str, str]:
    """定位表头行并返回 (数据体, 分隔符)。

    三代格式的表头行都含 "Tipo de registro"，据此锚定起点；
    再按该行中出现次数最多的分隔符判断分隔方式。
    """
    lines = text.split("\n")
    hdr_idx = next(
        (i for i, l in enumerate(lines[:15])
         if "tipo de registro" in l.lower() and "glucosa" in l.lower()),
        None,
    )
    if hdr_idx is None:
        raise ValueError("未找到表头行（缺少 Tipo de registro）")

    hdr = lines[hdr_idx]
    sep = max(("\t", ";", ","), key=hdr.count)

    # 处理「患者 ID 与表头粘在同一行」：形如
    # "HUPA0018P;;;;;;ID;Hora;Tipo de registro;..."，截断到第一个真实列名
    fields = hdr.split(sep)
    start = 0
    for j, f in enumerate(fields):
        if f.strip() in ("ID", "Dispositivo", "Hora"):
            start = j
            break
    hdr = sep.join(fields[start:])

    return "\n".join([hdr] + lines[hdr_idx + 1:]), sep


def _pick(df: pd.DataFrame, key: str) -> pd.Series | None:
    """按「精确列名 > 前缀 > 子串」三级优先级查找列。"""
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for pat in COLUMN_PATTERNS[key]:
        p = pat.lower()
        if p in lowered:
            return df[lowered[p]]
        for low, orig in lowered.items():
            if low.startswith(p):
                return df[orig]
        for low, orig in lowered.items():
            if p in low:
                return df[orig]
    return None


def _to_num(s: pd.Series) -> pd.Series:
    """字符串转数值，兼容西班牙语逗号小数点。"""
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def parse_file(path: str | Path) -> tuple[pd.DataFrame, ParseReport]:
    """解析单个导出文件，返回连续血糖序列与诊断信息。"""
    path = Path(path)
    m = re.match(r"(HUPA\d+P)", path.name)
    patient = m.group(1) if m else path.stem
    rep = ParseReport(path=path.name, patient=patient)

    try:
        body, sep = _split_head(_normalize_text(path))
        raw = pd.read_csv(io.StringIO(body), sep=sep, dtype=str,
                          on_bad_lines="skip", engine="python")
    except Exception as exc:  # noqa: BLE001
        rep.error = f"解析失败: {exc}"
        return pd.DataFrame(), rep

    rep.rows_raw = len(raw)
    col = {k: _pick(raw, k) for k in COLUMN_PATTERNS}
    if col["glucose"] is None or col["timestamp"] is None:
        rep.error = "缺少关键列（glucose/timestamp）"
        return pd.DataFrame(), rep

    def _num(key: str) -> pd.Series:
        s = col.get(key)
        return _to_num(s) if s is not None else pd.Series(np.nan, index=raw.index)

    # 时间格式随版本变化，逐格式尝试
    ts = col["timestamp"].astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M",
                "%d/%m/%Y %H:%M", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        parsed = parsed.fillna(pd.to_datetime(ts, format=fmt, errors="coerce"))
    parsed = parsed.fillna(pd.to_datetime(ts, errors="coerce", dayfirst=True))

    df = pd.DataFrame({
        "record_id": _num("record_id"),
        "timestamp": parsed,
        "record_type": _num("record_type"),
        "glucose": _num("glucose"),
        "rapid_insulin": _num("rapid_insulin"),
        "carbs": _num("carbs"),
        "fingerstick": _num("fingerstick"),
    })
    df["patient"] = patient

    # 只保留历史记录（连续监测主序列）
    hist = df[df["record_type"] == RECORD_HISTORICAL].copy()
    hist = hist.dropna(subset=["timestamp", "glucose"])
    hist = hist[(hist["glucose"] > 0) & (hist["glucose"] < 800)]
    if hist.empty:
        rep.error = "无有效历史记录"
        return pd.DataFrame(), rep

    # 设备内部记录号单调递增，是比时间戳更可靠的顺序来源
    hist = hist.sort_values(["record_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    # --- 时间轴修复：识别设备时钟重置造成的时间断裂 ---
    dt_min = hist["timestamp"].diff().dt.total_seconds() / 60.0
    broken = (dt_min < 0) | (dt_min > 2 * 60)
    hist["segment"] = broken.cumsum()
    rep.time_jumps = int(broken.sum())
    rep.n_segments = int(hist["segment"].nunique())
    rep.rows_glucose = len(hist)
    rep.span_days = float((hist["timestamp"].max() - hist["timestamp"].min()).total_seconds() / 86400)
    hist["dt_min"] = dt_min.fillna(SAMPLE_MINUTES)
    return hist, rep


def load_all(raw_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """批量解析目录下所有导出文件，返回 (长表, 诊断表)。"""
    paths = sorted(Path(raw_dir).glob("*.csv"))
    frames, reports = [], []
    for p in paths:
        try:
            df, rep = parse_file(p)
        except Exception as exc:  # noqa: BLE001
            rep = ParseReport(path=p.name, patient=p.stem, error=str(exc))
            df = pd.DataFrame()
        reports.append(rep.__dict__)
        if not df.empty:
            df["source_file"] = p.name
            frames.append(df)

    if not frames:
        raise RuntimeError(f"{raw_dir} 下没有解析出任何有效数据")

    data = pd.concat(frames, ignore_index=True)
    # 同一患者的不同导出文件可能重叠（长期监测），按 患者+时间 去重
    data = (data.sort_values(["patient", "record_id"])
                .drop_duplicates(subset=["patient", "timestamp"], keep="first")
                .reset_index(drop=True))
    return data, pd.DataFrame(reports)


def to_grid(df: pd.DataFrame, freq: str = f"{SAMPLE_MINUTES}min") -> pd.DataFrame:
    """把单个连续 segment 重采样到等间隔网格，暴露真实缺失。"""
    if df.empty:
        return df
    s = df.set_index("timestamp")["glucose"].sort_index()
    s = s[~s.index.duplicated(keep="first")]
    out = s.resample(freq).mean().to_frame("glucose")
    out["missing"] = out["glucose"].isna()
    out["glucose_filled"] = out["glucose"].interpolate(limit=2, limit_area="inside")
    return out.reset_index()