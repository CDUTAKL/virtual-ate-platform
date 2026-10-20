"""统一仪器驱动（设计文档 §3.2 仪器层）。

架构与对标关系：
- 仪器经 PyVISA 资源字符串连接（TCPIP0::1::inst0::INSTR 等），后端为 pyvisa-sim；
  SCPI 命令集与真实仪器一致（*IDN? / APPLy / CONFigure / READ? / MEASure?），
  将来接真实仪器只改资源字符串（及删去 SIM:INPut 桥接），这就是仪器自动化能力的可迁移证明。
- 测量噪声模型（文档 §3.2）：读数噪声 ±0.05%（σ）+ 偏移 σ（零均值校准残余，
  确定性偏置应由校准消除，guardband 覆盖的是残余不确定度）。
- σ_meas 标定：MeasurementNoise.sigma_meas(nominal) 供 testexec/limits 计算 guardband。

两条测量路径：
- SCPI 单 die 路径（measure_value）：真值 → SIM:INPut 写入 → READ? 取回（真实
  SCPI 往返）→ 叠加噪声。用于小 lot 端到端验证与驱动级测试。
- 批量路径（apply_batch）：同一噪声模型的 numpy 向量化实现，用于 5 万 die CP
  全流程（75 万次测量，SCPI 逐条往返会在真实机台上由硬件并行完成，软件仿真用
  批量路径等效）。
"""

from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np
import pyvisa
import yaml

_PKG_DIR = Path(__file__).parent
PSU_YAML = _PKG_DIR / "power_supply.yaml"
DMM_YAML = _PKG_DIR / "dmm.yaml"

# 默认资源字符串（对标真实机台的资源寻址；换真机只改这里）
DEFAULT_PSU_RESOURCE = "TCPIP0::1::inst0::INSTR"
DEFAULT_DMM_RESOURCE = "TCPIP0::2::inst0::INSTR"

# 每 die 的 SETUP 动作（与 configs/cp_plan.yaml 的 SETUP 阶段对应）
CP_APPLY_VOLT = 5.0
CP_APPLY_CURR = 0.5
DMM_CONFIG_VOLT = "CONFigure:VOLTage:DC 10,0.001"
DMM_CONFIG_CURR = "CONFigure:CURRent:DC 1,0.001"


@dataclass(frozen=True)
class MeasurementNoise:
    """测量噪声模型：measured = true×(1+N(0,rel_sigma)) + N(0,abs_sigma)。"""

    rel_sigma: float     # 读数噪声 σ（±0.05%）
    abs_sigma: float     # 偏移 σ（零均值校准残余，按量纲）

    def sigma_meas(self, nominal: float) -> float:
        """标定测量不确定度 σ_meas（guardband 用）：hypot(rel×nominal, abs)。"""
        return float(np.hypot(self.rel_sigma * abs(nominal), self.abs_sigma))

    def apply(self, true_value: np.ndarray | float, rng) -> np.ndarray | float:
        """对真值批量叠加测量噪声（向量化路径与 SCPI 路径共用同一模型）。"""
        v = np.asarray(true_value, dtype=float)
        noisy = v + v * rng.normal(0.0, self.rel_sigma, size=v.shape)
        noisy = noisy + rng.normal(0.0, self.abs_sigma, size=v.shape)
        if isinstance(true_value, np.ndarray):
            return noisy
        return float(noisy)


def merge_device_yaml(psu_yaml: Path = PSU_YAML, dmm_yaml: Path = DMM_YAML) -> dict:
    """合并 PSU/DMM 两份设备定义为一份 pyvisa-sim 配置（设备 + 资源表）。"""
    merged: dict = {"spec": "1.0", "devices": {}, "resources": {}}
    for path in (psu_yaml, dmm_yaml):
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for name, dev in cfg["devices"].items():
            if name in merged["devices"]:
                raise ValueError(f"设备名冲突: {name}")
            merged["devices"][name] = dev
        for res, ref in cfg["resources"].items():
            if res in merged["resources"]:
                raise ValueError(f"资源地址冲突: {res}")
            merged["resources"][res] = ref
    return merged


class InstrumentDriver:
    """PSU + DMM 统一驱动（PyVISA/pyvisa-sim 后端）。

    真实机台对应：换真实仪器时把 backend 参数换成 "@ivi" 等资源管理器、
    改资源字符串即可，SCPI 层与本类接口保持不变。
    """

    def __init__(self, backend: str | None = None,
                 psu_resource: str = DEFAULT_PSU_RESOURCE,
                 dmm_resource: str = DEFAULT_DMM_RESOURCE,
                 noise: MeasurementNoise | None = None,
                 workdir: Path | None = None):
        """backend=None → 使用内置 pyvisa-sim 设备（合并 instruments/*.yaml）。

        noise 缺省为纯 0.05% 读数噪声模型（abs_sigma=0），具体 abs_sigma
        由 TestPlan 按量纲注入（见 testexec/plan.py）。
        workdir 指定合并后仿真配置的落盘目录（默认系统临时目录）。
        """
        self.noise = noise or MeasurementNoise(rel_sigma=5.0e-4, abs_sigma=0.0)
        workdir = workdir or Path(tempfile.mkdtemp(prefix="pm300_sim_"))
        if backend is None:
            cfg = merge_device_yaml()
            sim_yaml = workdir / "_sim_devices.yaml"   # 合并后的仿真后端配置
            with open(sim_yaml, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True)
            backend = f"{sim_yaml}@sim"
        self._rm = pyvisa.ResourceManager(backend)
        self.psu = self._rm.open_resource(psu_resource)
        self.dmm = self._rm.open_resource(dmm_resource)
        for inst in (self.psu, self.dmm):
            inst.write_termination = "\n"
            inst.read_termination = "\n"
            inst.timeout = 5000

    # ---- 基础 SCPI ----
    def idn(self) -> dict[str, str]:
        """*IDN? 识别两台仪器（对标仪器自动化流程的连通性检查）。"""
        return {"PSU": self.psu.query("*IDN?").strip(),
                "DMM": self.dmm.query("*IDN?").strip()}

    def setup_cp(self) -> None:
        """CP 条件 SETUP：PSU 加电 5V/0.5A，DMM 配置电压/电流量程。"""
        self.psu.write("APPLy 5.0,0.5")
        self.psu.write("OUTPut ON")
        self.apply_voltage(CP_APPLY_VOLT)     # 编程值写入属性（同时完成类型初始化）
        self.apply_current(CP_APPLY_CURR)
        self.dmm.write(DMM_CONFIG_VOLT)
        self.dmm.write(DMM_CONFIG_CURR)

    def apply_voltage(self, volts: float) -> None:
        """VOLTage 编程（标准 SCPI 属性写）。"""
        self.psu.write(f"VOLTage {volts:.6f}")

    def apply_current(self, amps: float) -> None:
        """CURRent 编程（标准 SCPI 属性写）。"""
        self.psu.write(f"CURRent {amps:.6f}")

    def read_programmed_voltage(self) -> float:
        """回读 PSU 编程电压（验证激励加电成功）。"""
        return float(self.psu.query("MEASure:VOLTage:DC?"))

    # ---- 测量（SCPI 单 die 路径）----
    def measure_value(self, true_value: float, rng) -> float:
        """DMM 单值测量：SCPI 往返（SIM:INPut → READ?）+ 测量噪声。

        接真实仪器时，真值参数不再需要（仪器直接测 DUT），接口保持
        measure_value(pin, rng) 形态即可，本签名中的 true_value 对应
        "探针已把 DUT 输出送达仪器前端"的物理事实。
        """
        self.dmm.write(f"SIM:INPut {true_value:.6f}")
        raw = float(self.dmm.query("READ?"))
        return self.noise.apply(raw, rng)

    def measure_die(self, true_values: dict[str, float], rng) -> dict[str, float]:
        """按测试项名批量测一颗 die 的全部参数项（仍逐条 SCPI 往返）。

        供小 lot 端到端验证；整 lot 生产路径用 apply_batch 向量化。
        """
        return {name: self.measure_value(v, rng) for name, v in true_values.items()}
