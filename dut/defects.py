"""PMU300 五类缺陷模型定义（设计文档 §3.1 缺陷表）。

每类缺陷的作用机制：

| 缺陷 ID      | 作用机制                                   | 受影响测试项            | 目标检出手段        |
|-------------|-------------------------------------------|------------------------|--------------------|
| DEF_OFFSET  | 某通道输出电压偏移 +4~6%（落在规格外）      | CH1/2/3 输出电压（PTR） | CP 参数测试         |
| DEF_DRIFT   | 温度系数放大 5~10 倍                       | 高温点输出电压          | CP 多温测试 / Shmoo |
| DEF_LEAK    | 静态电流 ×5~20                             | 静态电流（PTR）         | CP 参数测试         |
| DEF_OPEN    | 某引脚功能失效                             | 功能测试（FTR）         | CP/FT 功能测试      |
| DEF_TIMING  | Buck 在低压高温重载下震荡                   | 纹波超上限 / 功能 fail  | FT 系统级测试       |

注入比例由 dut/config.py 的 D0 与 DEFECT_WEIGHTS 控制。
"""

from dataclasses import dataclass

import numpy as np

from . import config

#: 全部缺陷类型
DEFECT_KINDS = tuple(config.DEFECT_WEIGHTS.keys())

#: 各类缺陷可作用的通道（DEF_OFFSET / DEF_DRIFT），DEF_LEAK 只影响 LDO 通道
OFFSET_CHANNELS = (1, 2, 3)
DRIFT_CHANNELS = (1, 2, 3)
LEAK_CHANNELS = (1, 2)

#: DEF_OPEN 的开路引脚候选（开路后芯片无法通信/使能）
OPEN_PINS = ("SDA", "SCL", "EN", "FB3")


@dataclass
class Defect:
    """单颗 die 上的一道缺陷（标签仅存档，测试程序不可见）。"""

    kind: str                      # DEFECT_KINDS 之一
    channel: int = 0               # 作用通道 1/2/3（全局类缺陷为 0）
    magnitude: float = 0.0         # 幅度：偏移倍数 / 温漂倍数 / 漏电倍数 / 热纹波值
    pin: str = ""                  # DEF_OPEN 的引脚名
    detail: str = ""               # 人类可读描述（存档用）

    def __str__(self) -> str:  # pragma: no cover - 展示用途
        return self.detail or self.kind


def make_defect(kind: str, rng) -> Defect:
    """按缺陷类型抽取具体实现参数（通道、幅度），rng 保证可复现。"""
    if kind == "DEF_OFFSET":
        ch = int(rng.choice(OFFSET_CHANNELS))
        mag = 1.04 + 0.02 * rng.random()   # 输出偏移 +4~6%
        return Defect(kind, ch, mag, detail=f"CH{ch} 输出偏移 +{(mag - 1) * 100:.1f}%")
    if kind == "DEF_DRIFT":
        ch = int(rng.choice(DRIFT_CHANNELS))
        mag = 5.0 + 5.0 * rng.random()     # 温度系数放大 5~10 倍
        return Defect(kind, ch, mag, detail=f"CH{ch} 温度系数 ×{mag:.1f}")
    if kind == "DEF_LEAK":
        ch = int(rng.choice(LEAK_CHANNELS))
        mag = 5.0 + 15.0 * rng.random()    # 静态电流 ×5~20
        return Defect(kind, ch, mag, detail=f"CH{ch} 静态电流 ×{mag:.1f}")
    if kind == "DEF_OPEN":
        pin = str(rng.choice(OPEN_PINS))
        return Defect(kind, 0, 0.0, pin=pin, detail=f"引脚 {pin} 开路")
    if kind == "DEF_TIMING":
        mag = 60.0 + 90.0 * rng.random()   # 震荡时纹波 60~150 mVpp（限值 30）
        return Defect(kind, 3, mag, detail=f"Buck 环路不稳（热纹波 {mag:.0f} mVpp）")
    raise ValueError(f"未知缺陷类型: {kind}")


def choose_kind(rng) -> str:
    """按 config.DEFECT_WEIGHTS 的权重抽取一种缺陷类型。"""
    kinds = tuple(config.DEFECT_WEIGHTS.keys())
    weights = np.array([config.DEFECT_WEIGHTS[k] for k in kinds], dtype=float)
    weights /= weights.sum()
    return str(rng.choice(kinds, p=weights))
