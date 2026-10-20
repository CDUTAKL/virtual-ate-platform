"""参数实测值落在规格内：无缺陷批的主流参数应几乎全部在规格内。

口径：d0=0 生成整批 50,000 die（25°C CP 条件），
每个参数 out-of-spec 比例应 < 0.5%（自然工艺尾部），全项通过率 > 99%。
"""

import numpy as np

from dut import pmu300, wafer


def test_all_params_in_spec_rate():
    dies, _ = wafer.generate_lot(d0=0.0)
    assert len(dies) == 50_000
    for name, spec in pmu300.PARAM_SPECS.items():
        v = dies[name].to_numpy(dtype=float)
        ok = np.ones(len(v), dtype=bool)
        if spec.spec_lo is not None:
            ok &= v >= spec.spec_lo
        if spec.spec_hi is not None:
            ok &= v <= spec.spec_hi
        in_rate = ok.mean()
        assert in_rate > 0.995, f"{name} 规格内比例 {in_rate:.4%} 低于 99.5%"


def test_overall_pass_rate_without_defects():
    dies, _ = wafer.generate_lot(d0=0.0)
    pass_rate = dies["pass"].mean()
    assert pass_rate > 0.99, f"无缺陷批全项通过率 {pass_rate:.4%} 异常"
