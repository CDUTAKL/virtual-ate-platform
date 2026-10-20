"""波动模型统计合理性：模拟 1 万 die，均值/σ 与模型设定偏差 < 5%。

模型：param = nominal + lot_offset(N(0,σ_lot)) + wafer_gradient(径向)
            + die_noise(N(0,σ_die)) + 温度系数项

本测试对单层分量分别验证：
- die_noise：lot_offset=0、梯度=0 时，1 万样本均值 ≈ nominal、σ ≈ σ_die（±5%）
- lot_offset：4000 个批次的批次偏移 σ ≈ σ_lot（±5%）
- 温度系数：外推 125°C 的相对漂移与 tc1 设定一致
"""

import numpy as np

from dut import config, pmu300

N_DIE = 10_000


def test_die_noise_mean_and_sigma():
    spec = pmu300.PARAM_SPECS["VOUT1"]
    rng = np.random.default_rng(42)
    noise = pmu300.draw_die_noise(spec, N_DIE, rng)
    samples = spec.nominal + noise  # lot_offset = 0、梯度 = 0

    mean_dev = abs(samples.mean() - spec.nominal)
    std_dev = abs(samples.std() - spec.sigma_die)
    assert mean_dev < 0.05 * spec.sigma_die, f"均值偏差 {mean_dev:.2e} 超 5%σ"
    assert std_dev < 0.05 * spec.sigma_die, f"σ 偏差 {std_dev:.2e} 超 5%"


def test_lot_offset_sigma():
    spec = pmu300.PARAM_SPECS["RIPPLE3"]
    rng = np.random.default_rng(7)
    offsets = rng.normal(0.0, spec.sigma_lot, size=4000)
    std_dev = abs(offsets.std() - spec.sigma_lot)
    assert std_dev < 0.05 * spec.sigma_lot, f"σ_lot 偏差 {std_dev:.2e} 超 5%"
    # 文档约定：σ_lot = SIGMA_LOT_RATIO × σ_die
    assert abs(spec.sigma_lot - config.SIGMA_LOT_RATIO * spec.sigma_die) < 1e-12


def test_wafer_gradient_radial_symmetry():
    """径向梯度：同一片内梯度只随 r 变化，与角度无关（同 r 同增量）。"""
    spec = pmu300.PARAM_SPECS["VOUT2"]
    v25 = np.array([spec.nominal, spec.nominal])
    tc1 = np.array([0.0, 0.0])
    u = np.array([0.0, 1.0])  # 圆心 / 边缘
    grad_amp = spec.sigma_grad * 3  # 固定一个较大梯度幅度
    v = v25 + grad_amp * u
    out = pmu300.apply_temperature(v, spec, tc1, config.CP_TEMP_C)
    assert np.allclose(out, v), "25°C 时温度项应为 0"
    out_hot = pmu300.apply_temperature(v, spec, tc1, 125.0)
    assert np.isclose(out_hot[1] - out_hot[0], grad_amp), "径向增量与角度无关"


def test_temperature_coefficient_rel_and_abs():
    """相对系数参数按 ×(1+tc1·ΔT) 变化，绝对系数参数按 +tc1·ΔT 变化。"""
    rng = np.random.default_rng(0)
    v_rel = np.array([3.3])
    s_rel = pmu300.PARAM_SPECS["VOUT1"]
    hot = pmu300.apply_temperature(v_rel, s_rel, s_rel.tc1, 125.0)
    assert np.isclose(hot[0], 3.3 * (1 + s_rel.tc1 * 100), rtol=1e-12)

    v_abs = np.array([18.0])
    s_abs = pmu300.PARAM_SPECS["RIPPLE3"]
    hot = pmu300.apply_temperature(v_abs, s_abs, s_abs.tc1, 125.0)
    assert np.isclose(hot[0], 18.0 + s_abs.tc1 * 100, rtol=1e-12)
