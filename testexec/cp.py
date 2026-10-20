"""CP 执行器：测试计划驱动的整 lot 晶圆测试（W2 交付核心）。

执行链（对标机台 CP 测试）：
晶圆生成器产出的 die 真值（DUT 输出）→ SETUP（加电/量程）→ EXECUTE
（逐项测量：DUT 真值 + 仪器测量噪声 → 判定 vs guardband 后测试限）→
BINNING（首个 fail 项 → soft bin；功能优先 → hard bin）。

两条测量路径（同一噪声模型、同一判定数学）：
- 批量路径（默认，5 万 die）：numpy 向量化，全流程秒级；
- SCPI 路径（use_scpi=True，小 lot）：逐 die 经 instruments/driver.py 的
  SCPI 往返，用于端到端验证仪器层。

输出（outputs/cp_results.parquet，W3 STDF writer 消费）：
die 坐标/编号、12 参数测量值、15 项 pass、soft/hard bin、单 die 测试时间。
注意：结果表不含缺陷标签——标签仅存档，检出率统计由 detection_report()
事后 join 分析（模拟真实产线"测试后拿 STDF 对 FA 标签"的流程）。
"""

import time
import zlib

import numpy as np
import pandas as pd

from dut import config
from instruments.driver import InstrumentDriver
from .plan import TestPlan


def run_cp(lot: pd.DataFrame, plan: TestPlan, seed: int | None = None,
           use_scpi: bool = False, driver: InstrumentDriver | None = None,
           program: str | None = None) -> pd.DataFrame:
    """跑一遍 CP 测试计划，返回 die 级结果表（不含缺陷标签）。

    参数
    ----
    lot : dut.wafer.generate_lot() 的 die 表（真值 + 功能结果）。
    seed : 测量噪声随机种子；None 时用全局 config.SEED（全流程统一种子，可复现）。
    use_scpi : True 时逐 die 走 pyvisa-sim SCPI 往返（小 lot 验证用）。
    driver : 传 None 且 use_scpi=True 时自动创建（默认 pyvisa-sim 后端）。
    """
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    t0 = time.perf_counter()
    if use_scpi:
        measured = _measure_scpi(lot, plan, rng, driver)
    else:
        measured = _measure_batch(lot, plan, rng)

    results = _bin_and_assemble(lot, plan, measured, program=program or plan.program)
    results.attrs["wall_time_s"] = time.perf_counter() - t0
    return results


def plan_program_id(program: str) -> int:
    """程序名的确定性 ID（zlib.crc32，跨进程稳定；用于 STDB MIR 类字段）。"""
    return zlib.crc32(program.encode("utf-8")) & 0x7FFFFFFF


# ---- 测量路径 ----

def _true_param_matrix(lot: pd.DataFrame, plan: TestPlan) -> np.ndarray:
    """die 真值矩阵（n_die × 12 参数，EXECUTE 顺序的 PARM 项，可写副本）。"""
    names = [it.name for it in plan.param_items]
    return np.array(lot[names].to_numpy(dtype=float), dtype=float, copy=True)


def _measure_batch(lot: pd.DataFrame, plan: TestPlan, rng) -> np.ndarray:
    """批量路径：逐参数项向量化叠加测量噪声（与 SCPI 路径同一模型）。"""
    mat = _true_param_matrix(lot, plan)
    for j, item in enumerate(plan.param_items):
        mat[:, j] = plan.noise_for(item).apply(mat[:, j], rng)
    return mat


def _measure_scpi(lot: pd.DataFrame, plan: TestPlan, rng,
                  driver: InstrumentDriver | None) -> np.ndarray:
    """SCPI 路径：逐 die SETUP + 逐项 SIM:INPut→READ? 往返。"""
    if driver is None:
        driver = InstrumentDriver()
    driver.setup_cp()
    mat = _true_param_matrix(lot, plan)
    out = np.empty_like(mat)
    for i in range(len(lot)):
        if i % 200 == 0:  # 每 200 die 重新确认激励（对标机台周期性 setup）
            driver.setup_cp()
        row = {it.name: float(mat[i, j]) for j, it in enumerate(plan.param_items)}
        out[i, :] = list(driver.measure_die(row, rng).values())
    return out


# ---- 判定与分档 ----

def _bin_and_assemble(lot: pd.DataFrame, plan: TestPlan,
                      measured: np.ndarray, program: str) -> pd.DataFrame:
    """对测量值判测试限、soft/hard bin 分档，组装结果表。"""
    n = len(lot)
    param_names = [it.name for it in plan.param_items]
    func_names = [it.name for it in plan.func_items]

    # 参数项判定（向量化：vs guardband 后测试限）
    pass_mat = np.ones((n, len(param_names)), dtype=bool)
    for j, item in enumerate(plan.param_items):
        col = measured[:, j]
        if item.test_lo is not None:
            pass_mat[:, j] &= col >= item.test_lo
        if item.test_hi is not None:
            pass_mat[:, j] &= col <= item.test_hi

    # 功能项判定（不经仪器噪声，取 DUT 功能响应）
    func_pass = lot[func_names].to_numpy(dtype=bool)

    # soft bin：EXECUTE 顺序首个 fail 项
    first_fail = np.full(n, "", dtype=object)
    for j, item in enumerate(plan.param_items):
        newly = pass_mat[:, j] == False  # noqa: E712
        first_fail[(first_fail == "") & newly] = item.name
    for k, name in enumerate(func_names):
        newly = func_pass[:, k] == False  # noqa: E712
        first_fail[(first_fail == "") & newly] = name

    param_fail = ~pass_mat.all(axis=1)
    func_fail = ~func_pass.all(axis=1)

    # hard bin：功能优先（HBin3），其次参数（HBin2），全过 HBin1
    hard_bin = np.where(func_fail, 3, np.where(param_fail, 2, 1))
    sbin_map = {it.name: plan.bins.soft_bin(it.name).sbin for it in plan.items}
    sbin_name_map = {it.name: plan.bins.soft_bin(it.name).name for it in plan.items}
    soft_bin = np.array([sbin_map.get(f, 1) for f in first_fail], dtype=int)
    soft_name = [sbin_name_map.get(f, "PASS") for f in first_fail]

    # 组装结果表（parquet 契约：W3 STDF writer 消费此 schema）
    data: dict[str, object] = {
        "lot": plan.lot_id,
        "program": program,
        "wafer": lot["wafer"].to_numpy(),
        "x": lot["x"].to_numpy(),
        "y": lot["y"].to_numpy(),
        "head_num": 1,          # STDF 标准字段：单头单机（结构支持扩展多 site）
        "site_num": 1,
        "die_id": np.arange(n, dtype=int),
        "test_time_s": plan.die_test_time_s,
    }
    for j, name in enumerate(param_names):
        data[name] = np.round(measured[:, j], 6)
    for j, name in enumerate(param_names):
        data[f"{name}_pass"] = pass_mat[:, j]
    for k, name in enumerate(func_names):
        data[f"{name}_pass"] = func_pass[:, k]
    data["soft_bin"] = soft_bin
    data["soft_bin_name"] = soft_name
    data["hard_bin"] = hard_bin
    return pd.DataFrame(data)


# ---- 汇总报告 ----

def bin_summary(results: pd.DataFrame, plan: TestPlan) -> pd.DataFrame:
    """hard/soft bin 分布表（含名称），按计数降序。"""
    hb = plan.bins.hard_bins
    hard = (results.groupby("hard_bin").size().rename("count").reset_index())
    hard["name"] = hard["hard_bin"].map({b.hbin: b.name for b in hb})
    hard = hard.sort_values("hard_bin")
    soft = (results.groupby(["soft_bin", "soft_bin_name"]).size()
            .rename("count").reset_index().sort_values("count", ascending=False))
    return hard, soft


def param_capability_with_test_limits(results: pd.DataFrame,
                                      plan: TestPlan) -> pd.DataFrame:
    """按 guardband 后测试限计算参数 Cpk（CP 视角的过程能力）。

    统计总体取 HBin1（良品）die —— 工程惯例：过程能力用良品计算；
    缺陷 die 的极端值会污染 μ/σ（如 DEF_LEAK 把 IQ 分布拉飞）。
    注意：guardband 截断使该值略偏乐观，与规格限 Cpk（无缺陷批）对照使用。
    """
    from .limits import margin_sigma
    good = results[results["hard_bin"] == 1]
    rows = []
    for item in plan.param_items:
        v = good[item.name].to_numpy(dtype=float)
        mu, sd = v.mean(), v.std()
        rows.append({
            "item": item.name, "unit": item.unit, "mean": mu, "std": sd,
            "test_lo": item.test_lo, "test_hi": item.test_hi,
            "cpk_test_limit": margin_sigma(v, item.test_lo, item.test_hi),
        })
    return pd.DataFrame(rows)


def guardband_margin_table(results: pd.DataFrame, plan: TestPlan) -> pd.DataFrame:
    """guardband 报告：各参数测试限相对工艺分布的余量（σ 数），升序=风险最高在前。

    统计总体取 HBin1 die（同 Cpk 口径），避免缺陷极端值污染余量。
    """
    from .limits import margin_sigma
    good = results[results["hard_bin"] == 1]
    rows = []
    for item in plan.param_items:
        v = good[item.name].to_numpy(dtype=float)
        rows.append({
            "item": item.name,
            "k_sig_meas": plan.guardband_k * plan.sigma_meas(item),
            "margin_sigma": margin_sigma(v, item.test_lo, item.test_hi),
        })
    df = pd.DataFrame(rows).sort_values("margin_sigma").reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ---- 事后检出率分析（标签仅存档，测试程序不可见）----

DETECTABLE_LABELS = ("DEF_OFFSET", "DEF_LEAK", "DEF_OPEN")
LATENT_LABELS = ("DEF_DRIFT", "DEF_TIMING")


def detection_report(results: pd.DataFrame, lot: pd.DataFrame) -> dict:
    """检出率/过杀率/逃逸统计：结果表与存档标签按 die_id 对齐后分析。"""
    assert len(results) == len(lot), "结果表与 lot 行数不一致"
    labels = lot["defect"].to_numpy()
    fail = results["hard_bin"].to_numpy() != 1

    det_mask = np.isin(labels, DETECTABLE_LABELS)
    det_total = int(det_mask.sum())
    det_caught = int((det_mask & fail).sum())

    latent_mask = np.isin(labels, LATENT_LABELS)
    escapes = int((latent_mask & ~fail).sum())

    clean_mask = labels == ""
    clean_total = int(clean_mask.sum())
    overkill = int((clean_mask & fail).sum())

    return {
        "detectable_total": det_total,
        "detectable_caught": det_caught,
        "detection_rate": det_caught / det_total if det_total else 1.0,
        "latent_total": int(latent_mask.sum()),
        "latent_escapes": escapes,
        "clean_total": clean_total,
        "overkill": overkill,
        "overkill_rate": overkill / clean_total if clean_total else 0.0,
    }
