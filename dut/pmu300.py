"""PMU300 虚拟芯片行为模型（设计文档 §3.1）。

三通道电源管理芯片（虚构型号）：CH1 LDO 5V→3.3V/300mA、CH2 LDO 5V→1.8V/150mA、
CH3 Buck 5V→1.2V/2A，外加 10bit ADC、温度传感器与全局模拟项。

参数波动模型（蒙特卡洛）：

    param_value(die) = nominal
                     + lot_offset     ~ N(0, σ_lot)     # 批次间漂移
                     + wafer_gradient  径向梯度 A·(r/R)，A ~ N(0, σ_grad)  # 同片相关
                     + die_noise      ~ N(0, σ_die)     # die 间随机
                     + f(温度)         线性温度系数（tc_rel 为相对/绝对两类）

σ 值集中在 dut/config.py（SIGMA_DIE + SIGMA_LOT_RATIO/SIGMA_GRAD_RATIO），
目标：主流参数 Cpk 落在 1.0~1.6，其中 VOUT3 / RIPPLE3 最低（文档 §4）。

本文件包含：
- PARAM_SPECS：12 个参数测试项的规格表（规格值照抄文档 §3.1 表格）
- FUNCTIONAL_ITEMS：3 个功能测试项
- 参数级测量函数（向量化，晶圆生成器使用）
- PMU300Die：单 die 对象，支持任意温度/负载条件下的测量（测试用）
"""

from dataclasses import dataclass

import numpy as np

from . import config
from .defects import Defect


@dataclass(frozen=True)
class ParamSpec:
    """单个参数测试项的规格与统计特性。"""

    name: str
    channel: str                  # 'CH1' / 'CH2' / 'CH3' / 'GLOBAL'
    nominal: float                # 典型值
    spec_lo: float | None         # 规格下限（None = 单边上限）
    spec_hi: float | None         # 规格上限（None = 单边下限）
    unit: str
    sigma_die: float              # die 间随机 σ
    sigma_lot: float              # 批次间漂移 σ
    sigma_grad: float             # 片上径向梯度幅度 σ
    tc1: float                    # 典型一阶温度系数
    tc_rel: bool                  # True=相对系数（×值），False=绝对系数（+值）


def _build_specs() -> dict[str, ParamSpec]:
    """由 config 中的 σ / 温度系数组装规格表。"""
    table = [
        # name, channel, nominal, lo, hi, unit
        ("VOUT1", "CH1", 3.3, 3.234, 3.366, "V"),
        ("IQ1", "CH1", 90.0, None, 120.0, "uA"),
        ("LINE1", "CH1", 0.02, None, 0.05, "%/V"),
        ("VOUT2", "CH2", 1.8, 1.764, 1.836, "V"),
        ("IQ2", "CH2", 60.0, None, 80.0, "uA"),
        ("VOUT3", "CH3", 1.2, 1.176, 1.224, "V"),
        ("EFF3", "CH3", 90.0, 85.0, None, "%"),
        ("RIPPLE3", "CH3", 18.0, None, 30.0, "mVpp"),
        ("ADC_INL", "GLOBAL", 0.0, -4.0, 4.0, "LSB"),
        ("TEMP25", "GLOBAL", 0.0, -3.0, 3.0, "degC"),
        ("ISD", "GLOBAL", 1.1, None, 2.0, "uA"),
        ("POR", "GLOBAL", 1.65, 1.55, 1.75, "V"),
    ]
    specs: dict[str, ParamSpec] = {}
    for name, ch, nom, lo, hi, unit in table:
        tc1, tc_rel = config.TC1[name]
        s_die = config.SIGMA_DIE[name]
        specs[name] = ParamSpec(
            name=name, channel=ch, nominal=nom, spec_lo=lo, spec_hi=hi, unit=unit,
            sigma_die=s_die,
            sigma_lot=s_die * config.SIGMA_LOT_RATIO,
            sigma_grad=s_die * config.SIGMA_GRAD_RATIO,
            tc1=tc1, tc_rel=tc_rel,
        )
    return specs


#: 12 个参数测试项规格表（顺序即测试项顺序）
PARAM_SPECS = _build_specs()

#: 3 个功能测试项
FUNCTIONAL_ITEMS = ("FUNC_REG", "FUNC_EN", "FUNC_OCP")


def in_spec(spec: ParamSpec, value: float) -> bool:
    """参数值是否在规格内。"""
    if spec.spec_lo is not None and value < spec.spec_lo:
        return False
    if spec.spec_hi is not None and value > spec.spec_hi:
        return False
    return True


def draw_die_noise(spec: ParamSpec, n: int, rng) -> np.ndarray:
    """die 间随机分量：N(0, σ_die)，n 个独立样本。"""
    return rng.normal(0.0, spec.sigma_die, size=n)


def apply_temperature(v25: np.ndarray | float, spec: ParamSpec,
                      tc1: np.ndarray | float, temp_c: float) -> np.ndarray | float:
    """按温度系数外推到温度 temp_c（ΔT = temp_c - 25）。

    相对系数参数（VOUT/IQ/ISD）：v(T) = v(25) × (1 + tc1·ΔT)
    绝对系数参数（EFF/RIPPLE/INL/POR 等）：v(T) = v(25) + tc1·ΔT
    tc1 可为逐 die 数组（含 DEF_DRIFT 放大）。
    """
    dt = temp_c - config.CP_TEMP_C
    if dt == 0.0:
        return v25
    if spec.tc_rel:
        return v25 * (1.0 + tc1 * dt)
    return v25 + tc1 * dt


def apply_defect_to_value(name: str, value: float, defect: Defect | None,
                          temp_c: float, load_a: float) -> float:
    """缺陷对单个参数实测值的作用（与文档 §3.1 缺陷表一一对应）。

    - DEF_OFFSET：只推偏它声明通道的输出电压（×1.04~1.06）
    - DEF_LEAK：只放大它声明通道的静态电流（×5~20）
    - DEF_DRIFT：温度系数放大已在 tc1 中体现，此处无需改动
    - DEF_TIMING：Buck 在高温(≥85°C)重载(≥1.5A)下震荡，纹波跳到 60~150 mVpp
    - DEF_OPEN：不影响参数实测值（功能测试才 fail）
    25°C 常规条件下 DEF_DRIFT / DEF_TIMING 完全潜伏。
    """
    if defect is None:
        return value
    if defect.kind == "DEF_OFFSET" and name == f"VOUT{defect.channel}":
        return value * defect.magnitude
    if defect.kind == "DEF_LEAK" and name == f"IQ{defect.channel}":
        return value * defect.magnitude
    if (defect.kind == "DEF_TIMING" and name == "RIPPLE3"
            and temp_c >= config.STRESS_TEMP_C and load_a >= config.STRESS_LOAD_A):
        return defect.magnitude
    return value


class PMU300Die:
    """单颗 PMU300 die：25°C 基准过程值 + 逐 die 温度系数 + 缺陷状态。

    用于测试与 W2+ 的测试执行层；晶圆生成器（wafer.py）用向量化路径
    实现同样的数学，保证两者 25°C 结果一致。
    """

    def __init__(self, base25: dict[str, float], tc1: dict[str, float],
                 defect: Defect | None = None):
        self.base25 = dict(base25)
        self.tc1 = dict(tc1)
        self.defect = defect

    def measure(self, name: str, temp_c: float = config.CP_TEMP_C,
                load_a: float = config.CP_LOAD_A) -> float:
        """测量一个参数项在给定温度/负载下的值。"""
        spec = PARAM_SPECS[name]
        v = apply_temperature(self.base25[name], spec, self.tc1[name], temp_c)
        return apply_defect_to_value(name, float(v), self.defect, temp_c, load_a)

    def in_spec(self, name: str, temp_c: float = config.CP_TEMP_C,
                load_a: float = config.CP_LOAD_A) -> bool:
        return in_spec(PARAM_SPECS[name], self.measure(name, temp_c, load_a))

    def functional(self, name: str, temp_c: float = config.CP_TEMP_C,
                   load_a: float = config.CP_LOAD_A) -> bool:
        """功能测试：寄存器读写 / 使能关断 / 过流保护。

        - DEF_OPEN：引脚开路，三项功能测试全部 fail（参数实测不受影响）
        - DEF_TIMING：高温重载下 Buck 环路震荡，触发过流保护误动作（FUNC_OCP fail）
        - 其余情况全部 PASS
        """
        assert name in FUNCTIONAL_ITEMS
        if self.defect is not None and self.defect.kind == "DEF_OPEN":
            return False
        if (self.defect is not None and self.defect.kind == "DEF_TIMING"
                and name == "FUNC_OCP"
                and temp_c >= config.STRESS_TEMP_C
                and load_a >= config.STRESS_LOAD_A):
            return False
        return True

    def is_good(self, temp_c: float = config.CP_TEMP_C,
                load_a: float = config.CP_LOAD_A) -> bool:
        """全项通过（12 参数 + 3 功能）才算 good die。"""
        return (all(self.in_spec(n, temp_c, load_a) for n in PARAM_SPECS)
                and all(self.functional(n, temp_c, load_a) for n in FUNCTIONAL_ITEMS))


def make_die(rng, defect: Defect | None = None) -> PMU300Die:
    """抽取一颗 die 的过程值（nominal + die_noise + 温漂系数抖动）。

    用于单元测试：不含 lot/wafer 分量，只看单 die 行为。
    DEF_DRIFT 在此把声明通道的 tc1 放大 5~10 倍。
    """
    base: dict[str, float] = {}
    tc1: dict[str, float] = {}
    for name, spec in PARAM_SPECS.items():
        base[name] = float(spec.nominal + rng.normal(0.0, spec.sigma_die))
        t = spec.tc1
        if not (defect is not None and defect.kind == "DEF_DRIFT"
                and name == f"VOUT{defect.channel}"):
            t *= 1.0 + rng.normal(0.0, config.TC1_JITTER)
        else:
            # DEF_DRIFT：缺陷机制主导该通道温度系数，不再叠加 jitter，
            # 保证 5~10 倍放大后在 125°C 必然越限（文档：高温点输出电压失效）
            t *= defect.magnitude
        tc1[name] = t
    return PMU300Die(base, tc1, defect)
