"""端到端流程：解析 -> 质量评估 -> 特征 -> 患者级交叉验证 -> 临床评价 -> 图表。

用法
----
    python scripts/run_pipeline.py --raw-dir data/raw --out reports
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

from cgm import hypo, io, metrics, model, plots, quality  # noqa: E402
from cgm.features import build_features  # noqa: E402

HORIZONS = {2: 30, 4: 60}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--max-patients", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 解析 ----------
    print(">>> [1/6] 解析 LibreView 导出文件 ...")
    data, reports = io.load_all(args.raw_dir)
    data = data.sort_values(["patient", "timestamp"]).reset_index(drop=True)
    print(f"    患者 {data['patient'].nunique()} 人，记录 {len(data):,} 条，"
          f"文件成功 {int(reports['error'].isna().sum())}/{len(reports)}")
    reports.to_csv(out / "parse_report.csv", index=False, encoding="utf-8-sig")

    # ---------- 2. 信号质量 ----------
    print(">>> [2/6] 信号质量评估 ...")
    seg_q = quality.segment_quality(data)
    comp = quality.detect_compression_low(data)
    summary = quality.quality_summary(seg_q, comp)
    seg_q.to_csv(out / "segment_quality.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "patient_quality.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(out / "compression_low_events.csv", index=False, encoding="utf-8-sig")
    print(f"    段数 {len(seg_q)}，可用段(>=48点) {int((seg_q['n_points'] >= 48).sum())}，"
          f"疑似压力性低值 {len(comp)} 次")

    # ---------- 3. 特征 ----------
    print(">>> [3/6] 特征工程 ...")
    feat_data, feat_cols = build_features(data, horizons_steps=tuple(HORIZONS))
    print(f"    建模样本 {len(feat_data):,}，特征 {len(feat_cols)} 个")

    # ---------- 4. 交叉验证 ----------
    print(">>> [4/6] 患者级 5 折交叉验证 ...")
    all_folds, oofs, perf_rows = [], {}, []
    for h, minutes in HORIZONS.items():
        oof, folds = model.cross_validate(feat_data, feat_cols, h)
        oofs[minutes] = oof
        all_folds.append(folds)
        for name, df in oof.items():
            perf_rows.append({"horizon_min": minutes, "model": name,
                              **metrics.regression_metrics(df["y_true"], df["y_pred"])})
    fold_metrics = pd.concat(all_folds, ignore_index=True)
    fold_metrics.to_csv(out / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")

    perf = pd.DataFrame(perf_rows)
    perf.to_csv(out / "model_comparison.csv", index=False, encoding="utf-8-sig")
    print(perf.to_string(index=False))

    # ---------- 5. 临床评价 ----------
    print(">>> [5/6] Clarke 网格与低血糖预警评估 ...")
    zone_rows, alarm_out = [], {}
    for minutes in HORIZONS.values():
        d = oofs[minutes]["LightGBM"]
        zt = metrics.zone_table(d["y_true"], d["y_pred"])
        zt.insert(0, "horizon_min", minutes)
        zone_rows.append(zt)
        alarm_out[minutes] = metrics.alarm_performance(d, horizon_min=minutes)
    zones = pd.concat(zone_rows, ignore_index=True)
    zones.to_csv(out / "clarke_zones.csv", index=False, encoding="utf-8-sig")
    print(zones.to_string(index=False))

    alarm_summary = pd.DataFrame({
        f"{k}min": v["summary"] for k, v in alarm_out.items()
    }).T
    alarm_summary.to_csv(out / "alarm_summary.csv", encoding="utf-8-sig")
    print(alarm_summary.to_string())

    # ---------- 5b. 低血糖感知建模 ----------
    print(">>> [5b] 低血糖感知分类器（事件检测重构）...")
    clf_df = hypo.train_hypo_classifier(feat_data, feat_cols, horizon=2)
    hsum = hypo.summary(clf_df)
    sweep = hypo.threshold_sweep(clf_df)
    sweep.to_csv(out / "hypo_threshold_sweep.csv", index=False, encoding="utf-8-sig")
    print(f"    AUPRC={hsum['AUPRC']}  AUROC={hsum['AUROC']}  "
          f"正样本率={hsum['positive_rate']}")
    print(sweep.to_string(index=False))

    # ---------- 6. 图表 ----------
    print(">>> [6/6] 生成图表 ...")
    top_patient = summary.sort_values("points", ascending=False)["patient"].iloc[0]
    plots.plot_signal_trace(data, top_patient, fig_dir / "fig1_signal_trace.png")
    plots.plot_quality(seg_q, fig_dir / "fig2_signal_quality.png")
    plots.plot_clarke(
        oofs[30]["LightGBM"]["y_true"], oofs[30]["LightGBM"]["y_pred"],
        oofs[60]["LightGBM"]["y_true"], oofs[60]["LightGBM"]["y_pred"],
        "30 min ahead", "60 min ahead", fig_dir / "fig3_clarke_grid.png",
    )
    plots.plot_prediction(oofs[30]["LightGBM"], top_patient, 30,
                          fig_dir / "fig4_prediction_30min.png")

    _, imp = model.fit_final_model(feat_data, feat_cols, horizon=2)
    plots.plot_importance(imp, 30, fig_dir / "fig5_feature_importance.png")
    plots.plot_hypo(alarm_out[30]["per_patient"], fig_dir / "fig6_hypo_detection.png")
    plots.plot_hypo_tradeoff(hsum["pr_curve"], sweep, fig_dir / "fig7_hypo_tradeoff.png")

    # 指标 JSON，供 README 引用
    body = {
        "dataset": {
            "patients": int(data["patient"].nunique()),
            "records": int(len(data)),
            "segments": int(len(seg_q)),
            "usable_segments": int((seg_q["n_points"] >= 48).sum()),
            "compression_low_events": int(len(comp)),
        },
        "model_comparison": perf.to_dict(orient="records"),
        "clarke": zones.to_dict(orient="records"),
        "alarm": {str(k): v["summary"] for k, v in alarm_out.items()},
        "top_features": imp.head(10).round(1).to_dict(),
        "hypo_aware": {"AUPRC": hsum["AUPRC"], "AUROC": hsum["AUROC"],
                       "positive_rate": hsum["positive_rate"],
                       "sweep": sweep.to_dict(orient="records")},
    }
    (out / "results.json").write_text(json.dumps(body, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"\n完成，结果写入 {out}/")


if __name__ == "__main__":
    main()