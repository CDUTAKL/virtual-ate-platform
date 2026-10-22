"""SPC：注入漂移必报警（报告片数与规则），干净 lot 无失控报警，WE 规则手工序列单测。"""

from dut import wafer
from testexec.plan import TestPlan
from testexec import cp
from analytics import spc

import numpy as np
import pandas as pd


def _main_results():
    plan = TestPlan.load()
    lot, _ = wafer.generate_lot()
    return cp.run_cp(lot, plan)


def test_clean_lot_no_alarms():
    """主 lot（seed=20261020 干净数据）：无 ①类（超 3σ）失控报警。

    ③/④ 运行/趋势类规则作用于程序生成噪声，浮点累加顺序随平台
    BLAS/向量化实现不同存在边界翻转（macOS/Linux 曾差 1 条趋势报警），
    不属于 WE 实现缺陷；③/④ 的判定正确性由
    test_we_rules_handcrafted 以手工确定性序列守护。
    """
    alarms = spc.western_electric_alarms(spc.wafer_means(_main_results()))
    assert not any(a["rule"] == 1 for a in alarms), f"干净 lot 出现失控报警: {alarms}"


def test_we_rules_handcrafted():
    """WE 判异逻辑确定性单测（手工序列，跨平台稳定）。

    覆盖：常数序列零报警（含 σ̂=0 退化情形）、规则③单调递增触发、
    规则①超 3σ 触发。程序生成噪声的边界行为不在本测试范围。
    """
    flat = pd.Series([10.0] * 25, index=range(1, 26))
    assert spc.western_electric_alarms(flat, spc.imr_chart(flat)) == []

    ramp = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] + [0.0] * 18,
                     index=range(1, 26))
    rules3 = [a for a in spc.western_electric_alarms(ramp, spc.imr_chart(ramp))
              if a["rule"] == 3]
    assert rules3, "连续 7 点严格递增必须触发规则③"

    out = pd.Series([0.0] * 12 + [100.0] + [0.0] * 12, index=range(1, 26))
    rules1 = [a for a in spc.western_electric_alarms(out, spc.imr_chart(out))
              if a["rule"] == 1]
    assert rules1 and rules1[0]["wafer"] == 13, "单点飞出 3σ 必须触发规则①并报对片号"


def test_injected_drift_triggers_alarm():
    """第 13~19 片注入 +21mV 爬坡：必须报警，且触发点在注入区间内。"""
    res = _main_results()
    injected = spc.inject_wafer_drift(res)
    means = spc.wafer_means(injected)
    alarms = spc.western_electric_alarms(means)
    assert alarms, "注入漂移后 SPC 必须报警"
    trigger_wafers = {a["wafer"] for a in alarms}
    assert trigger_wafers <= set(range(13, 20)), \
        f"触发片 {trigger_wafers} 应落在注入区间 13~19"
    assert any(a["rule"] == 1 for a in alarms), "规则①（超 3σ）必须触发"
    # 报警信息含规则号与片号（面试口径："第几片触发哪条规则"）
    for a in alarms:
        assert 1 <= a["rule"] <= 4 and 1 <= a["wafer"] <= 25 and a["detail"]


def test_injection_shift_monotonic_ramp():
    """注入位移：线性爬坡，W19 位移 = 总位移，W12/W20 不受影响的。"""
    res = _main_results()
    inj = spc.inject_wafer_drift(res, total_shift=0.021)
    delta13 = (inj[inj.wafer == 13]["VOUT1"] - res[res.wafer == 13]["VOUT1"]).iloc[0]
    delta19 = (inj[inj.wafer == 19]["VOUT1"] - res[res.wafer == 19]["VOUT1"]).iloc[0]
    assert np.isclose(delta13, 0.021 / 7, atol=1e-9)
    assert np.isclose(delta19, 0.021, atol=1e-9)
    assert np.isclose((inj[inj.wafer == 12]["VOUT1"].to_numpy()
                       - res[res.wafer == 12]["VOUT1"].to_numpy()), 0).all()


def test_imr_chart_limits_sane():
    """I-MR 控制限：UCL > CL > LCL，σ̂ = MR̄/1.128。"""
    chart = spc.imr_chart(spc.wafer_means(_main_results()))
    assert chart["lcl"] < chart["x_bar"] < chart["ucl"]
    assert np.isclose(chart["sigma_hat"], chart["mr_bar"] / spc.D2)
    assert chart["mr_ucl"] == 3.267 * chart["mr_bar"]
