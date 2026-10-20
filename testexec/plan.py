"""TestPlan：测试计划（YAML 加载，设计文档 §3.3）。

加载 configs/cp_plan.yaml：
- 组装 TestItem 列表（规格限来自 YAML——测试程序的限值外置；加载时与 DUT
  规格表 dut/pmu300.py 校验一致性，防止计划与芯片规格脱节）；
- 按量纲注入测量噪声模型，标定 σ_meas；
- 应用 guardband：test_limit = spec_limit ∓ k·σ_meas（limits.py）；
- 加载 bin 定义表（configs/bins.yaml）。

对标关系：本类对应机台测试程序中的 Test Plan 对象（STS8200 / V93000 概念），
Flow/Item/Binning 分别对应其 Test Flow / Test Method / Binning Table。
"""

from pathlib import Path

import numpy as np
import yaml

from dut import pmu300
from instruments.driver import MeasurementNoise
from .binning import BinningTable
from .flow import TestFlow
from .item import TestItem
from .limits import guardband_limits

DEFAULT_PLAN_PATH = Path(__file__).parent.parent / "configs" / "cp_plan.yaml"


class TestPlan:
    """一份 CP 测试计划：流程 + 测试项（含测试限）+ bin 表 + guardband 参数。"""

    __test__ = False  # 防止 pytest 误收集（类名以 Test 开头）

    def __init__(self, lot_id: str, program: str, flow: TestFlow,
                 items: list[TestItem], bins: BinningTable,
                 guardband_k: float, setup_actions: list[dict],
                 noise_rel_sigma: float, noise_abs_by_unit: dict[str, float]):
        self.lot_id = lot_id
        self.program = program
        self.flow = flow
        self.items = items                      # 按 EXECUTE 顺序
        self.bins = bins
        self.guardband_k = guardband_k
        self.setup_actions = setup_actions
        self.noise_rel_sigma = noise_rel_sigma
        self.noise_abs_by_unit = noise_abs_by_unit

    # ---- 派生视图 ----
    @property
    def item_by_name(self) -> dict[str, TestItem]:
        return {it.name: it for it in self.items}

    @property
    def param_items(self) -> list[TestItem]:
        return [it for it in self.items if it.kind == "PARM"]

    @property
    def func_items(self) -> list[TestItem]:
        return [it for it in self.items if it.kind == "FUNC"]

    def noise_for(self, item: TestItem) -> MeasurementNoise:
        """按测试项量纲取测量噪声模型（abs_sigma 按量纲配置）。"""
        abs_sig = float(self.noise_abs_by_unit.get(item.unit or "", 0.0))
        return MeasurementNoise(rel_sigma=self.noise_rel_sigma, abs_sigma=abs_sig)

    def sigma_meas(self, item: TestItem) -> float:
        """标定 σ_meas：σ_meas = hypot(rel×nominal, abs)（guardband 输入）。"""
        nominal = abs(pmu300.PARAM_SPECS[item.name].nominal)
        return self.noise_for(item).sigma_meas(nominal)

    @property
    def die_test_time_s(self) -> float:
        """单 die 理论测试时间 = SETUP + EXECUTE 各项 meas_time 之和。"""
        setup_t = sum(a.get("meas_time_s", 0.0) for a in self.setup_actions)
        return setup_t + sum(it.meas_time_s for it in self.items)

    # ---- 加载 ----
    @classmethod
    def load(cls, path: str | Path = DEFAULT_PLAN_PATH) -> "TestPlan":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        plan_cfg = cfg["plan"]

        flow = TestFlow.from_yaml(plan_cfg["flow"])
        gb = cfg.get("guardband", {})
        k = float(gb.get("k", 3.0))
        meas = cfg.get("measurement", {})
        rel_sigma = float(meas.get("rel_sigma", 5.0e-4))
        abs_by_unit = {str(u): float(s) for u, s in meas.get("abs_sigma_by_unit", {}).items()}

        items: list[TestItem] = []
        for row in cfg["items"]:
            name = row["name"]
            if name not in flow.exec_items:
                raise ValueError(f"测试项 {name} 未在 EXECUTE 阶段入列")
            item = TestItem(
                test_num=int(row["test_num"]), name=name, kind=row["kind"],
                unit=row.get("unit"), spec_lo=row.get("lo"), spec_hi=row.get("hi"),
                instrument=row.get("instrument"), pin=row.get("pin"),
                meas_time_s=float(row.get("meas_time_s", 0.0)),
            )
            items.append(item)

        # 与 DUT 规格表一致性校验（YAML 限值照抄文档，防止两处漂移）
        for item in items:
            if item.kind != "PARM":
                continue
            spec = pmu300.PARAM_SPECS[item.name]
            for yaml_lim, dut_lim, tag in ((item.spec_lo, spec.spec_lo, "下限"),
                                           (item.spec_hi, spec.spec_hi, "上限")):
                if (yaml_lim is None) != (dut_lim is None) or (
                        yaml_lim is not None and abs(yaml_lim - dut_lim) > 1e-9):
                    raise ValueError(
                        f"{item.name} 规格{tag}不一致: 计划 {yaml_lim} vs DUT {dut_lim}")

        # guardband：σ_meas 标定 + 测试限内缩
        finalized: list[TestItem] = []
        for item in items:
            if item.kind != "PARM":
                finalized.append(item)
                continue
            sig = cls._sigma_meas_for(item, rel_sigma, abs_by_unit)
            t_lo, t_hi = guardband_limits(item.spec_lo, item.spec_hi, sig, k)
            if t_lo is not None and t_hi is not None and t_lo >= t_hi:
                raise ValueError(f"{item.name} guardband 后测试限倒挂 "
                                 f"({t_lo} >= {t_hi})，请检查 k 与 σ_meas")
            finalized.append(item.with_test_limits(t_lo, t_hi))

        # bin 表（相对 plan 文件解析路径）
        bins_path = path.parent / Path(flow.binning_path).name
        bins = BinningTable.load(str(bins_path), [it.name for it in finalized
                                                  if it.kind == "FUNC"])

        return cls(
            lot_id=plan_cfg["lot_id"], program=plan_cfg["program"],
            flow=flow, items=finalized, bins=bins, guardband_k=k,
            setup_actions=list(cfg.get("setup", [])),
            noise_rel_sigma=rel_sigma, noise_abs_by_unit=abs_by_unit,
        )

    @staticmethod
    def _sigma_meas_for(item: TestItem, rel_sigma: float,
                        abs_by_unit: dict[str, float]) -> float:
        nominal = abs(pmu300.PARAM_SPECS[item.name].nominal)
        return float(np.hypot(rel_sigma * nominal, abs_by_unit.get(item.unit or "", 0.0)))
