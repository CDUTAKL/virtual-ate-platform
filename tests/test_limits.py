"""guardband 数学：k=3 时限值确实内缩 3σ_meas；单边/双边规格；余量报告。"""

import numpy as np
import pytest

from dut import pmu300
from testexec.limits import guardband_limits, margin_sigma
from testexec.plan import TestPlan


def test_guardband_shrinks_exactly_k_sigma():
    """k=3 时：test_lo = spec_lo + 3σ_meas，test_hi = spec_hi − 3σ_meas。"""
    lo, hi = guardband_limits(3.234, 3.366, sig_meas=0.001, k=3.0)
    assert np.isclose(lo, 3.234 + 3 * 0.001)
    assert np.isclose(hi, 3.366 - 3 * 0.001)


def test_guardband_one_sided():
    """单边规格只内缩相应一侧。"""
    lo, hi = guardband_limits(None, 120.0, sig_meas=0.05, k=3.0)
    assert lo is None and np.isclose(hi, 120.0 - 0.15)
    lo, hi = guardband_limits(85.0, None, sig_meas=0.05, k=3.0)
    assert np.isclose(lo, 85.0 + 0.15) and hi is None


def test_guardband_k0_equals_spec():
    lo, hi = guardband_limits(1.176, 1.224, sig_meas=0.001, k=0.0)
    assert np.isclose(lo, 1.176) and np.isclose(hi, 1.224)


def test_plan_test_limits_are_inside_spec():
    """计划加载后：全部参数测试限严格在规格限内侧（guardband 生效）。"""
    plan = TestPlan.load()
    assert plan.guardband_k == 3.0
    for item in plan.param_items:
        sig = plan.sigma_meas(item)
        if item.spec_lo is not None:
            assert np.isclose(item.test_lo, item.spec_lo + 3.0 * sig)
            assert item.test_lo > item.spec_lo
        if item.spec_hi is not None:
            assert np.isclose(item.test_hi, item.spec_hi - 3.0 * sig)
            assert item.test_hi < item.spec_hi


def test_sigma_meas_calibration():
    """σ_meas 标定 = hypot(rel×nominal, abs)，与噪声模型实证 σ 一致（<5%）。"""
    plan = TestPlan.load()
    rng = np.random.default_rng(0)
    for name in ("VOUT1", "IQ1", "RIPPLE3"):
        item = plan.item_by_name[name]
        sig_model = plan.sigma_meas(item)
        noise = plan.noise_for(item)
        nominal = abs(pmu300.PARAM_SPECS[name].nominal)
        true_v = np.full(200_000, nominal)
        meas = noise.apply(true_v, rng)
        sig_emp = meas.std()
        assert abs(sig_emp - sig_model) < 0.05 * sig_model, \
            f"{name} 实证 σ {sig_emp:.2e} vs 标定 {sig_model:.2e}"


def test_margin_sigma_symmetric_case():
    """margin_sigma：均值居中、σ=1、限 ±3 时余量 = 3σ/(3σ) = 1。"""
    rng = np.random.default_rng(1)
    v = rng.normal(0.0, 1.0, size=200_000)
    assert abs(margin_sigma(v, -3.0, 3.0) - 1.0) < 0.02
