"""CP 端到端：小 lot（1000 die）SCPI 路径 + 大 lot 批量路径性能 + 结果表字段完整性。"""

import time

import numpy as np
import pandas as pd

from dut import wafer
from instruments.driver import InstrumentDriver
from testexec import cp
from testexec.plan import TestPlan


def _schema_expected(plan: TestPlan) -> set[str]:
    meta = {"lot", "program", "wafer", "x", "y", "head_num", "site_num",
            "die_id", "test_time_s", "soft_bin", "soft_bin_name", "hard_bin"}
    values = {it.name for it in plan.param_items}
    passes = {f"{it.name}_pass" for it in plan.items}
    return meta | values | passes


def test_small_lot_scpi_end_to_end():
    """1000-die 小 lot 走 SCPI 往返路径（仪器层真实参与），全链路跑通。"""
    plan = TestPlan.load()
    lot, _ = wafer.generate_lot()
    small = lot.head(1000).reset_index(drop=True)
    driver = InstrumentDriver()
    t0 = time.perf_counter()
    results = cp.run_cp(small, plan, use_scpi=True, driver=driver)
    dt = time.perf_counter() - t0
    assert len(results) == 1000
    assert dt < 60, f"SCPI 小 lot 耗时 {dt:.1f}s 异常"
    assert set(results["hard_bin"].unique()) <= {1, 2, 3}
    # 与批量路径同种子对比：两路径噪声模型相同、rng 抽样顺序不同，
    # 因此逐值不相等，但 bin 判定应高度一致（>99%）、参数均值应吻合。
    res_batch = cp.run_cp(small, plan)
    agree = (results["hard_bin"] == res_batch["hard_bin"]).mean()
    assert agree > 0.99, f"两条测量路径 bin 一致率 {agree:.2%}"
    for item in plan.param_items:
        m_scpi = results[item.name].mean()
        m_batch = res_batch[item.name].mean()
        assert abs(m_scpi - m_batch) < 0.02 * abs(m_batch) + 1e-6


def test_full_lot_batch_performance():
    """整 lot 批量路径：5 万 die × 15 项，远低于 3 分钟性能门槛。"""
    plan = TestPlan.load()
    lot, _ = wafer.generate_lot()
    t0 = time.perf_counter()
    results = cp.run_cp(lot, plan)
    dt = time.perf_counter() - t0
    assert len(results) == 50_000
    assert dt < 60, f"批量 CP 耗时 {dt:.1f}s（要求 <3min，实测应秒级）"


def test_results_schema_completeness():
    """结果表字段完整性：W3 STDF writer 需要的字段全部在位，且无缺陷标签泄漏。"""
    plan = TestPlan.load()
    lot, _ = wafer.generate_lot()
    results = cp.run_cp(lot.head(500), plan)
    expected = _schema_expected(plan)
    assert expected <= set(results.columns), \
        f"缺字段: {expected - set(results.columns)}"
    # 标签隔离：测试程序不可见缺陷标签
    assert "defect" not in results.columns
    assert "region" not in results.columns
    # 坐标与 die_id 完整
    assert results["die_id"].is_unique
    assert results[["wafer", "x", "y"]].notna().all().all()


def test_measurement_noise_actually_applied():
    """测量值 ≠ 真值（噪声确实叠加），且差异在噪声模型范围内。"""
    plan = TestPlan.load()
    lot, _ = wafer.generate_lot()
    small = lot.head(2000)
    results = cp.run_cp(small, plan)
    for item in plan.param_items[:3]:
        true_v = small[item.name].to_numpy(dtype=float)
        meas_v = results[item.name].to_numpy(dtype=float)
        assert not np.allclose(true_v, meas_v), f"{item.name} 未叠加测量噪声"
        diff = np.abs(meas_v - true_v)
        sig = plan.sigma_meas(item)
        assert diff.mean() < 5 * sig


def test_test_time_accounting():
    """测试时间：单 die = SETUP+EXECUTE 各项之和；结果表与计划一致。"""
    plan = TestPlan.load()
    lot, _ = wafer.generate_lot()
    results = cp.run_cp(lot.head(100), plan)
    assert np.allclose(results["test_time_s"], plan.die_test_time_s)
    assert plan.die_test_time_s > 0
