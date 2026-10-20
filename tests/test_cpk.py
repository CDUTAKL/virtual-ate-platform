"""过程能力：无缺陷批 12 个参数 Cpk 全部落在 1.0~1.6（文档 §3.1 设计目标）。

文档 §4 预期 Cpk 最低参数为 VOUT3 或 RIPPLE3（1.0~1.3），此处一并验证。
"""

import numpy as np

from dut import pmu300, wafer


def _cpk(values: np.ndarray, spec: pmu300.ParamSpec) -> float:
    mu, sd = values.mean(), values.std()
    lo = spec.spec_lo if spec.spec_lo is not None else -np.inf
    hi = spec.spec_hi if spec.spec_hi is not None else np.inf
    return min(hi - mu, mu - lo) / (3 * sd)


def test_all_params_cpk_in_design_band():
    dies, _ = wafer.generate_lot(d0=0.0)
    cpks = {}
    for name, spec in pmu300.PARAM_SPECS.items():
        cpks[name] = _cpk(dies[name].to_numpy(dtype=float), spec)
        assert 1.0 <= cpks[name] <= 1.6, \
            f"{name} Cpk={cpks[name]:.2f} 不在 1.0~1.6 设计区间"
    lowest = min(cpks, key=cpks.get)
    assert lowest in ("VOUT3", "RIPPLE3"), \
        f"Cpk 最低参数应为 VOUT3 或 RIPPLE3（文档 §4），实际 {lowest}"
    assert cpks[lowest] >= 1.0
