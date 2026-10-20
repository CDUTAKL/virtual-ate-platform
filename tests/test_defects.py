"""缺陷注入后对应参数/功能确实越限（每类缺陷抽 30 颗验证）。

物理合理性的关键断言：
- DEF_OFFSET 只推偏它声明的通道，其他通道输出电压不受影响
- DEF_LEAK 只放大它声明通道的静态电流
- DEF_OPEN 功能测试全部 fail 而参数实测全部正常
- DEF_DRIFT 25°C 正常、125°C 声明通道输出电压越限（潜伏型）
- DEF_TIMING 25°C 正常、高温重载下纹波越限且 FUNC_OCP fail（潜伏型）
"""

import numpy as np
import pytest

from dut import config, pmu300
from dut.defects import make_defect

N_SAMPLES = 30


def _sample(kind: str):
    rng = np.random.default_rng(2026)
    return [pmu300.make_die(rng, make_defect(kind, rng)) for _ in range(N_SAMPLES)]


def test_defect_offset_pushes_only_declared_channel():
    for die in _sample("DEF_OFFSET"):
        ch = die.defect.channel
        # 声明通道越限
        assert not die.in_spec(f"VOUT{ch}"), "DEF_OFFSET 声明通道应在规格外"
        # 其他通道输出电压不受影响（仍在规格内）
        for other in (1, 2, 3):
            if other != ch:
                assert die.in_spec(f"VOUT{other}"), "DEF_OFFSET 不应影响其他通道"
        # 功能测试不受影响
        assert all(die.functional(f) for f in pmu300.FUNCTIONAL_ITEMS)


def test_defect_leak_only_declared_iq():
    for die in _sample("DEF_LEAK"):
        ch = die.defect.channel
        assert ch in (1, 2)
        assert not die.in_spec(f"IQ{ch}"), "DEF_LEAK 声明通道静态电流应越限"
        other = 2 if ch == 1 else 1
        assert die.in_spec(f"IQ{other}"), "DEF_LEAK 不应影响另一通道静态电流"


def test_defect_open_functional_fail_params_ok():
    for die in _sample("DEF_OPEN"):
        assert all(not die.functional(f) for f in pmu300.FUNCTIONAL_ITEMS), \
            "DEF_OPEN 三项功能测试必须全部 fail"
        assert all(die.in_spec(n) for n in pmu300.PARAM_SPECS), \
            "DEF_OPEN 参数实测应全部正常"


def test_defect_drift_latent_at_cp_fails_at_high_temp():
    for die in _sample("DEF_DRIFT"):
        ch = die.defect.channel
        assert die.in_spec(f"VOUT{ch}"), "DEF_DRIFT 在 25°C CP 下应潜伏"
        assert not die.in_spec(f"VOUT{ch}", temp_c=125.0), \
            "DEF_DRIFT 在 125°C 下声明通道应越限"


def test_defect_timing_latent_at_cp_fails_under_stress():
    for die in _sample("DEF_TIMING"):
        assert die.in_spec("RIPPLE3"), "DEF_TIMING 在 25°C 常规负载下应潜伏"
        assert die.functional("FUNC_OCP"), "DEF_TIMING 在 CP 条件下 FUNC_OCP 应通过"
        # 高温 + 重载（文档：低压高温重载下震荡）
        v_hot = die.measure("RIPPLE3", temp_c=125.0, load_a=2.0)
        assert v_hot > pmu300.PARAM_SPECS["RIPPLE3"].spec_hi, "高温重载纹波应越限"
        assert not die.functional("FUNC_OCP", temp_c=125.0, load_a=2.0), \
            "高温重载下 FUNC_OCP 应 fail"


def test_no_defect_die_is_good():
    rng = np.random.default_rng(1)
    for _ in range(N_SAMPLES):
        die = pmu300.make_die(rng)
        assert die.is_good(), "无缺陷 die 应全项通过"


def test_defect_archive_labels_roundtrip():
    """存档标签与行为一致：OFFSET/LEAK/OPEN 在 CP 下即 fail（非潜伏）。"""
    for kind, cp_fail in [("DEF_OFFSET", True), ("DEF_LEAK", True),
                          ("DEF_OPEN", True), ("DEF_DRIFT", False),
                          ("DEF_TIMING", False)]:
        dies = _sample(kind)
        assert all(d.is_good() != cp_fail for d in dies), \
            f"{kind} 的 CP 潜伏性与文档不符"
