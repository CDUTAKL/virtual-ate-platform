"""相关性分析：矩阵性质、冗余对识别、删项复跑零差异（含合成冗余对验证机制）。

诚实口径：本仿真 DUT 参数噪声独立，主 lot 预期无 |r|>0.95 冗余对；
机制正确性用合成冗余对验证（与主 lot 结论分开报告）。
"""

import numpy as np
import pandas as pd
import pytest

from analytics import correlation


def _synthetic_results(n=3000, seed=0):
    """合成结果：VOUT2 = 0.5×VOUT1 完全函数相关（冗余对，fail 集合一致）。"""
    rng = np.random.default_rng(seed)
    vout1 = rng.normal(3.30, 0.016, n)
    vout2 = 0.5 * vout1                        # r = 1.0；>1.68 ⟺ VOUT1>3.36
    iq1 = rng.normal(90, 6, n)
    p1 = (vout1 >= 3.24) & (vout1 <= 3.36)
    p2 = vout2 <= 1.68
    p3 = iq1 <= 120
    any_param_fail = ~(p1 & p2 & p3)
    return pd.DataFrame({
        "wafer": np.repeat([1, 2, 3], n // 3),
        "hard_bin": np.where(any_param_fail, 2, 1),   # 与 pass 标志自洽
        "VOUT1": vout1, "VOUT2": vout2, "IQ1": iq1,
        "VOUT1_pass": p1, "VOUT2_pass": p2, "IQ1_pass": p3,
        "FUNC_REG_pass": True, "FUNC_EN_pass": True, "FUNC_OCP_pass": True,
    })


LIMITS = pd.DataFrame([
    {"test_txt": "VOUT1", "lo_limit": 3.24, "hi_limit": 3.36},
    {"test_txt": "VOUT2", "lo_limit": None, "hi_limit": 1.68},
    {"test_txt": "IQ1", "lo_limit": None, "hi_limit": 120.0},
])
EXEC = ["VOUT1", "VOUT2", "IQ1", "FUNC_REG", "FUNC_EN", "FUNC_OCP"]


def test_corr_matrix_properties():
    corr = correlation.correlation_matrix(_synthetic_results())
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.isclose(corr.loc["VOUT1", "VOUT2"], corr.loc["VOUT2", "VOUT1"])
    assert corr.loc["VOUT1", "VOUT2"] > 0.9       # 合成强相关


def test_redundant_pair_detection():
    corr = correlation.correlation_matrix(_synthetic_results())
    pairs = correlation.redundant_pairs(corr)
    assert len(pairs) == 1
    assert {pairs[0]["item_a"], pairs[0]["item_b"]} == {"VOUT1", "VOUT2"}
    assert pairs[0]["r"] > 0.95


def test_rerun_without_dropped_item_zero_diff():
    """删 VOUT2 后全量复跑：bin 判定与原结果一致（VOUT2 fail 的 die 除外时仍由
    其余项兜住——此处合成数据保证 VOUT2 fail ⊆ VOUT1 fail，故零差异）。"""
    res = _synthetic_results()
    new_bins = correlation.rerun_bins_without(res, ["VOUT2"], LIMITS, EXEC)
    v2_fail = ~res["VOUT2_pass"]
    assert (new_bins[v2_fail] == 2).all()          # VOUT2 超限 die 由重判兜住
    assert (new_bins[~v2_fail] == res["hard_bin"][~v2_fail]).all()


def test_time_report_math():
    rep = correlation.test_time_report(
        {"A": 0.1, "B": 0.2, "C": 0.1}, drop=["B"], setup_time_s=0.05)
    assert rep["baseline_s"] == pytest.approx(0.45)
    assert rep["optimized_s"] == pytest.approx(0.25)
    assert rep["saved_pct"] == pytest.approx(0.2 / 0.45)
    assert rep["dropped_items"] == ["B"]


def test_main_lot_reported_honestly():
    """主 lot 如实口径：max|r| 低于阈值 → 无冗余对；删项复跑（空删）与原
    bin 完全一致（限值直接用 plan 的精确测试限，排除四舍五入伪差异）。"""
    from dut import wafer
    from testexec.plan import TestPlan
    from testexec import cp
    plan = TestPlan.load()
    res = cp.run_cp(wafer.generate_lot()[0], plan)
    corr = correlation.correlation_matrix(res)
    n = len(corr)
    max_r = max(abs(corr.iloc[i, j]) for i in range(n) for j in range(i + 1, n))
    pairs = correlation.redundant_pairs(corr)
    assert max_r < 0.95, f"主 lot 出现意外强相关 {max_r}"
    assert pairs == []
    limits = pd.DataFrame([
        {"test_txt": it.name, "lo_limit": it.test_lo, "hi_limit": it.test_hi}
        for it in plan.param_items
    ])
    new_bins = correlation.rerun_bins_without(res, [], limits,
                                              list(plan.flow.exec_items))
    assert (new_bins == res["hard_bin"]).all(), \
        "空删复跑必须与原 bin 逐 die 一致（同一限值、同一测量值）"
