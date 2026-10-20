"""一键入口：完整量产测试闭环演练（W1 晶圆生成 → W2 CP 测试 → W3 STDF+量产分析 → W4 特性化+失效分析）。

运行：python main.py
产出：
- outputs/wafermap.png       整批 25 片 wafer map（fail die 空间分布）
- outputs/lot_dies.csv.gz    每 die 存档数据（含缺陷标签，仅存档）
- outputs/cp_results.parquet CP 结果表（不含缺陷标签）
- outputs/lot_cp.stdf        STDF V4 二进制数据文件（850k 记录）
- outputs/analysis/          量产分析产物（bin 帕累托/Cpk 表/SPC 图/相关矩阵/时间优化）
- outputs/char/              特性化产物（Shmoo 图/参数分布图）
- outputs/fa/                FA 产物（失效签名/混淆矩阵/FA 报告）
"""

import time
from pathlib import Path

import numpy as np

from dut import config, pmu300, wafer
from instruments.driver import InstrumentDriver
from report import wafermap
from testexec import cp
from testexec.plan import TestPlan

OUTPUT_DIR = Path(__file__).parent / "outputs"


def _cpk(values: np.ndarray, spec: pmu300.ParamSpec) -> float:
    mu, sd = values.mean(), values.std()
    lo = spec.spec_lo if spec.spec_lo is not None else -np.inf
    hi = spec.spec_hi if spec.spec_hi is not None else np.inf
    return min(hi - mu, mu - lo) / (3 * sd)


def stage_w1() -> tuple[Path, Path]:
    """W1：晶圆生成 + 良率/Cpk 统计 + wafer map。返回 (dies, clean) 供 W2 复用。"""
    print(f"[W1] 生成虚拟晶圆批次 {config.LOT_ID} "
          f"（{config.N_WAFERS} 片 × {config.DIE_PER_WAFER} die，seed={config.SEED}）...")
    dies, wafers = wafer.generate_lot()
    clean, _ = wafer.generate_lot(d0=0.0)  # 无缺陷批：用于自然 Cpk

    total = len(dies)
    yield_ = dies["pass"].mean()
    print(f"\n===== 良率统计（25°C CP，{total:,} dies）=====")
    print(f"整体良率 : {yield_ * 100:.2f}%  （规格限口径，目标区间 85%~97%）")
    print(f"各片良率 : {wafers['yield'].min() * 100:.2f}% ~ "
          f"{wafers['yield'].max() * 100:.2f}%")
    print("缺陷标签存档（测试程序不可见）: " + ", ".join(
        f"{k or '无'}={int(v)}" for k, v in dies["defect"].value_counts().items()))
    latent = dies["defect"].isin(["DEF_DRIFT", "DEF_TIMING"]).sum()
    print(f"潜伏缺陷（25°C CP 逃逸，需多温/FT 检出）: {int(latent)} 颗")

    print(f"\n===== 参数 Cpk（无缺陷批 {len(clean):,} dies，规格限口径，目标 1.0~1.6）=====")
    for name, spec in pmu300.PARAM_SPECS.items():
        print(f"  {name:8s} Cpk = {_cpk(clean[name].to_numpy(dtype=float), spec):.2f}")

    csv_path = OUTPUT_DIR / "lot_dies.csv.gz"
    dies.to_csv(csv_path, index=False, compression="gzip")
    print(f"\n[W1] die 级数据已存档: {csv_path}")

    png_path = wafermap.plot_lot(dies, OUTPUT_DIR / "wafermap.png")
    print(f"[W1] wafer map 已输出: {png_path}")
    assert 0.85 <= yield_ <= 0.97, "整体良率不在 85%~97% 区间，请调 D0/缺陷率（dut/config.py）"
    return dies, clean


def stage_w2_cp(dies) -> None:
    """W2：加载测试计划 → 仪器自检 → 批量 CP（5 万 die × 15 项）→ 摘要 + parquet。"""
    print("\n[W2] 加载 CP 测试计划（configs/cp_plan.yaml，guardband k=3.0）...")
    plan = TestPlan.load()
    driver = InstrumentDriver()          # pyvisa-sim 后端
    idn = driver.idn()                   # 仪器连通性自检（对标上电自检 *IDN?）
    print(f"[W2] 仪器自检: PSU={idn['PSU']}  DMM={idn['DMM']}")
    driver.setup_cp()
    print(f"[W2] SETUP 完成：PSU 编程电压回读 {driver.read_programmed_voltage():.3f} V")

    print(f"[W2] 执行 CP 全流程（{len(dies):,} dies × {len(plan.items)} 项，批量路径）...")
    results = cp.run_cp(dies, plan, driver=driver)   # 默认批量路径（同噪声模型）
    wall = results.attrs["wall_time_s"]
    wall_str = f"{wall * 1000:.0f} ms" if wall < 1 else f"{wall:.1f} s"

    # ---- 摘要 ----
    yield_cp = (results["hard_bin"] == 1).mean()
    hard, soft = cp.bin_summary(results, plan)
    print(f"\n===== CP 摘要（guardband 测试限口径，实测 {wall_str}）=====")
    print(f"整体良率 : {yield_cp * 100:.2f}%  （规格限口径 90.59%，差值即 guardband 过杀）")
    print("hard bin : " + ", ".join(
        f"H{row.hard_bin} {row['name']}={int(row['count'])}"
        for _, row in hard.iterrows()))
    print("soft bin Top5: " + ", ".join(
        f"S{row.soft_bin} {row.soft_bin_name}={int(row['count'])}"
        for _, row in soft.head(5).iterrows()))

    cap = cp.param_capability_with_test_limits(results, plan)
    cap_min = cap.sort_values("cpk_test_limit").iloc[0]
    print(f"Cpk（测试限）: 最低 {cap_min['item']} = {cap_min['cpk_test_limit']:.2f}；"
          f"全表见下")
    for r in cap.itertuples():
        print(f"  {r.item:8s} Cpk(test) = {r.cpk_test_limit:.2f}")

    gb = cp.guardband_margin_table(results, plan).head(3)
    print("guardband 余量最小 Top3（风险最高）:")
    for r in gb.itertuples():
        print(f"  {r.rank}. {r.item:8s} 余量 {r.margin_sigma:.2f}σ"
              f"（guardband 宽 k·σ_meas = {r.k_sig_meas:.2e}）")

    total_time = float(results["test_time_s"].sum())
    print(f"测试时间: 单 die {plan.die_test_time_s * 1000:.0f} ms，"
          f"整 lot 理论机时 {total_time / 3600:.2f} h；本仿真实测 {wall_str}")

    # ---- 检出率统计（事后 join 存档标签，测试程序不可见）----
    rep = cp.detection_report(results, dies)
    print(f"\n===== 检出率统计 =====")
    print(f"CP 可检出缺陷（OFFSET/LEAK/OPEN）: {rep['detectable_caught']}/"
          f"{rep['detectable_total']} = {rep['detection_rate'] * 100:.2f}%  （要求 ≥99%）")
    print(f"guardband 过杀（无缺陷 die 判 fail）: {rep['overkill']}"
          f"/{rep['clean_total']} = {rep['overkill_rate'] * 100:.2f}%")
    print(f"潜伏缺陷逃逸（DRIFT/TIMING，需多温 CP/FT/Shmoo 抓）: "
          f"{rep['latent_escapes']}/{rep['latent_total']}")

    pq_path = OUTPUT_DIR / "cp_results.parquet"
    results.to_parquet(pq_path, index=False)
    print(f"\n[W2] CP 结果表已输出: {pq_path}")
    assert rep["detection_rate"] >= 0.99, "CP 可检出缺陷检出率低于 99%"
    assert wall < 180, "CP 全流程超过 3 分钟"


def stage_w3_stdf_analytics() -> dict:
    """W3：STDF V4 写出 + round-trip + 量产分析（良率/Cpk/SPC/相关性/时间优化）。"""
    print("\n[W3] STDF V4 写出 + round-trip 自检 + 量产分析 ...")
    from analytics import run_all
    t0 = time.perf_counter()
    summary = run_all.main()          # 打印 STDF 摘要（大小/记录数）与各分析结论
    dt = time.perf_counter() - t0
    print(f"[W3] 阶段完成，耗时 {dt:.1f}s")
    return summary


def stage_w4_char_fa() -> dict:
    """W4：特性化（Shmoo + 参数分布）+ 失效分析（签名聚类 + 混淆矩阵）。"""
    print("\n[W4] 特性化（Shmoo 扫描 / 参数分布）+ 失效分析（聚类 / 混淆矩阵）...")
    from char import distribution, shmoo
    from fa import cluster as fa_cluster

    out: dict = {}
    t0 = time.perf_counter()
    shm = shmoo.run()                                   # 打印 Shmoo 结论
    out["shmoo"] = shm
    t_shmoo = time.perf_counter() - t0

    t0 = time.perf_counter()
    dist = distribution.run()                           # 打印参数分布统计
    out["distribution"] = dist
    t_dist = time.perf_counter() - t0

    t0 = time.perf_counter()
    fa = fa_cluster.run_fa()                            # 打印混淆矩阵对角线/轮廓系数
    out["fa"] = fa
    t_fa = time.perf_counter() - t0

    rep = fa["report"]
    print(f"[W4] FA 混淆矩阵对角线命中率: {rep['diagonal_hit_rate'] * 100:.2f}%"
          f"（目标 ≥90%），轮廓系数 {fa['silhouette']:.4f}")
    print(f"[W4] 产物: outputs/char/（Shmoo/分布图）, outputs/fa/（fa_report.md 等）")
    print(f"[W4] 阶段完成：Shmoo {t_shmoo:.1f}s + 分布 {t_dist:.1f}s + FA {t_fa:.1f}s")
    assert rep["diagonal_hit_rate"] >= 0.90, "FA 混淆矩阵对角线命中率低于 90%"
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    t_all = time.perf_counter()
    dies, _ = stage_w1()
    stage_w2_cp(dies)
    stage_w3_stdf_analytics()
    stage_w4_char_fa()
    print(f"\n全流程完成，总耗时 {time.perf_counter() - t_all:.1f}s。")


if __name__ == "__main__":
    main()
