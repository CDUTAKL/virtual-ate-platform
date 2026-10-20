"""SPC：注入漂移必报警（报告片数与规则），干净 lot 零误报。"""

from dut import wafer
from testexec.plan import TestPlan
from testexec import cp
from analytics import spc

import numpy as np


def _main_results():
    plan = TestPlan.load()
    lot, _ = wafer.generate_lot()
    return cp.run_cp(lot, plan)


def test_clean_lot_no_alarms():
    """主 lot（seed=20261020 干净数据）：WE 四规则零报警。"""
    alarms = spc.western_electric_alarms(spc.wafer_means(_main_results()))
    assert alarms == []


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
