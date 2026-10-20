"""W3 分析编排：STDF 生成 → 解析回读 → 全部分析产物（文档 §3.4 + §3.6）。

用法：.venv/bin/python analytics/run_all.py
前置：先跑 main.py 生成 outputs/cp_results.parquet（W2 产物）。

分析层架构原则（文档 §2.1）：只消费 STDF 文件（round-trip 解析后的结果表），
不接触 DUT 模型与测试程序内部对象；parquet 仅作为 STDF writer 的输入。
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:      # 允许 `python analytics/run_all.py` 直接运行
    sys.path.insert(0, str(ROOT))

import pandas as pd

# 模块名 yield 是 Python 关键字，须用 importlib 导入
ayield = importlib.import_module("analytics.yield")
from stdf.cp_meta import build_cp_meta
from stdf.writer import write_cp_stdf
from stdf.reader import read_stdf
from analytics import capability, correlation, spc

OUT = ROOT / "outputs"
ANA = OUT / "analysis"


def main() -> dict:
    t_start = time.perf_counter()
    ANA.mkdir(parents=True, exist_ok=True)

    # ---- 1. STDF 生成 + round-trip（分析层唯一数据源）----
    parquet_path = OUT / "cp_results.parquet"
    if not parquet_path.exists():
        raise SystemExit("先运行 main.py 生成 outputs/cp_results.parquet（W2 产物）")
    results_src = pd.read_parquet(parquet_path)
    meta = build_cp_meta()
    stdf_path = write_cp_stdf(results_src, meta, OUT / "lot_cp.stdf", now=1760000000)
    t_stdf = time.perf_counter() - t_start

    sf = read_stdf(stdf_path)
    results = sf.to_results_frame()          # 分析层结果表（来自 STDF，非 parquet）
    limits = sf.limits_table()               # 测试限/规格限（来自 PTR）
    t_read = time.perf_counter() - t_start - t_stdf

    # round-trip 一致性抽查：测量值/bins 对齐
    n_check = min(5000, len(results))
    assert (results["hard_bin"].to_numpy()[:n_check]
            == results_src["hard_bin"].to_numpy()[:n_check]).all()
    assert abs(results["VOUT1"].iloc[0] - results_src["VOUT1"].iloc[0]) < 1e-4

    # ---- 2. 良率 / bin 帕累托 ----
    ysum = ayield.summary(results)
    hard_pareto, soft_pareto = ayield.bin_pareto(results)
    pareto_png = ayield.plot_bin_pareto(results, ANA / "bin_pareto.png")

    # ---- 3. Cpk / Ppk ----
    cap = capability.capability_table(results, limits)
    cap_csv = capability.save_capability_csv(cap, ANA / "cpk_table.csv")

    # ---- 4. SPC：干净 lot 零报警 + 注入漂移演示 ----
    clean_means = spc.wafer_means(results)
    clean_alarms = spc.western_electric_alarms(clean_means)
    injected = spc.inject_wafer_drift(results)
    inj_means = spc.wafer_means(injected)
    inj_alarms = spc.western_electric_alarms(inj_means)
    spc_png = spc.plot_imr(clean_means, inj_means, ANA / "spc_imr.png")

    # ---- 5. 相关性 / 测试时间优化 ----
    corr = correlation.correlation_matrix(results)
    corr.to_csv(ANA / "correlation_matrix.csv", float_format="%.4f")
    corr_png = correlation.plot_correlation(corr, ANA / "correlation_matrix.png")
    pairs = correlation.redundant_pairs(corr)
    if pairs:  # 有冗余对才执行删项复跑（本仿真模型预期无）
        drop = sorted({p["item_a"] for p in pairs} | {p["item_b"] for p in pairs})
        new_bins = correlation.rerun_bins_without(results, drop, limits,
                                                  meta.exec_order)
        zero_diff = bool((new_bins == results["hard_bin"]).all())
    else:
        drop, zero_diff = [], True
    from testexec.plan import TestPlan     # 只读取 meas_time（configs 外置）
    plan = TestPlan.load()
    meas_times = {it.name: it.meas_time_s for it in plan.items}
    setup_t = sum(a.get("meas_time_s", 0.0) for a in plan.setup_actions)
    time_report = correlation.test_time_report(meas_times, drop,
                                               setup_time_s=setup_t)
    report_json = correlation.save_report(
        {"pairs": pairs, "zero_bin_diff": zero_diff, **time_report},
        ANA / "test_time_optimization.json")

    total = time.perf_counter() - t_start

    # ---- 摘要 ----
    print("===== W3 分析摘要 =====")
    print(f"STDF 文件: {stdf_path}（{stdf_path.stat().st_size / 1e6:.1f} MB，"
          f"写出 {t_stdf:.1f}s，读回解析 {t_read:.1f}s，"
          f"{len(sf.records):,} 条记录）")
    print(f"良率: {ysum['yield'] * 100:.2f}%  FPY: {ysum['fpy'] * 100:.2f}%  "
          f"DPPM: {ysum['dppm']:.0f}")
    print("hard bin 帕累托: " + ", ".join(
        f"H{int(r.hard_bin)} {r['name']}={int(r['count'])}"
        for _, r in hard_pareto.iterrows()))
    print(f"Cpk 最低参数: {cap.iloc[0]['item']} Cpk={cap.iloc[0]['cpk']:.2f} "
          f"Ppk={cap.iloc[0]['ppk']:.2f}（判定 {cap.iloc[0]['judgement']}）→ {cap_csv}")
    print(f"SPC 干净 lot 报警: {len(clean_alarms)}（应为 0）")
    for a in inj_alarms:
        print(f"SPC 注入漂移报警: 第 {a['wafer']} 片触发规则{a['rule']}（{a['detail']}）")
    print(f"相关矩阵 |r| 最大非对角: "
          f"{max((abs(corr.iloc[i, j]) for i in range(len(corr)) for j in range(i + 1, len(corr))), default=0):.3f}")
    print(f"冗余对（|r|>0.95）: {pairs if pairs else '无（本仿真参数噪声独立，如实报告）'}")
    print(f"测试时间优化: {time_report['baseline_s']:.3f}s → "
          f"{time_report['optimized_s']:.3f}s（节省 {time_report['saved_pct'] * 100:.1f}%）"
          f"，删项复跑 bin 零差异: {zero_diff}")
    print(f"产物目录: {ANA}（pareto PNG / cpk CSV / SPC PNG / 相关矩阵 PNG+CSV / 优化 JSON）")
    print(f"总耗时: {total:.1f}s")
    assert not clean_alarms, "干净 lot 不应触发 SPC 报警"
    assert total < 120, "STDF 写出 + 全部分析应 < 2 分钟"
    return {"yield": ysum, "cpk_min": cap.iloc[0].to_dict(),
            "spc_injected_alarms": inj_alarms, "pairs": pairs,
            "time_report": time_report, "total_s": total}


if __name__ == "__main__":
    main()
