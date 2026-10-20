"""仪器层：pyvisa-sim SCPI 往返、测量噪声统计特性、σ_meas 标定。"""

import numpy as np
import pytest

from instruments.driver import InstrumentDriver, MeasurementNoise


@pytest.fixture(scope="module")
def driver() -> InstrumentDriver:
    return InstrumentDriver()


def test_scpi_identity_and_setup(driver):
    """*IDN? 识别 + CP 条件 SETUP + 编程电压回读（对标上电自检流程）。"""
    idn = driver.idn()
    assert "PMU300-PSU" in idn["PSU"]
    assert "PMU300-DMM" in idn["DMM"]
    driver.setup_cp()
    assert driver.read_programmed_voltage() == pytest.approx(5.0, abs=1e-3)


def test_measure_roundtrip(driver):
    """SCPI 测量往返：真值注入 → READ? 取回 → 叠加噪声后接近真值。"""
    rng = np.random.default_rng(42)
    vals = [3.298217, 1.799512, 0.031255]
    for v in vals:
        m = driver.measure_value(v, rng)
        assert abs(m - v) < 0.01 * abs(v) + 1e-3   # 噪声在 0.05% 读数量级


def test_noise_statistics():
    """噪声统计：零均值、σ = hypot(rel×nominal, abs)（10 万样本，<5% 偏差）。"""
    noise = MeasurementNoise(rel_sigma=5.0e-4, abs_sigma=2.0e-3)
    rng = np.random.default_rng(7)
    true_v = np.full(100_000, 3.3)
    meas = noise.apply(true_v, rng)
    err = meas - true_v
    sig_model = noise.sigma_meas(3.3)
    assert abs(err.mean()) < 0.05 * sig_model, f"噪声有偏: {err.mean():.2e}"
    assert abs(err.std() - sig_model) < 0.05 * sig_model, \
        f"实证 σ {err.std():.2e} vs 模型 {sig_model:.2e}"


def test_noise_proportional_to_reading():
    """读数噪声随读数缩放：2× 读数 → rel 部分 σ 翻倍（绝对项不变）。"""
    noise = MeasurementNoise(rel_sigma=1.0e-3, abs_sigma=0.0)
    rng = np.random.default_rng(8)
    a = noise.apply(np.full(100_000, 1.0), rng)
    b = noise.apply(np.full(100_000, 2.0), rng)
    assert abs(a.std() * 2 - b.std()) < 0.05 * b.std()


def test_sigma_meas_calibration_hypot():
    """σ_meas 标定公式：hypot(rel×nominal, abs)。"""
    noise = MeasurementNoise(rel_sigma=5.0e-4, abs_sigma=1.0e-3)
    assert noise.sigma_meas(0.0) == pytest.approx(1.0e-3)
    assert noise.sigma_meas(3.3) == pytest.approx(
        float(np.hypot(5.0e-4 * 3.3, 1.0e-3)))
