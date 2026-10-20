"""TestFlow：测试流程（设计文档 §3.3，结构对标机台测试程序）。

流程固定为 SETUP → EXECUTE → BINNING 三阶段：
- SETUP：加电/量程配置等激励动作（psu_apply_5v, dmm_config）；
- EXECUTE：按序执行的测试项（顺序即 soft bin 的 fail 优先序）；
- BINNING：查 bin 定义表判定分档。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowPhase:
    name: str            # SETUP / EXECUTE / BINNING
    items: list[str]     # 测试项名（BINNING 阶段为表路径占位，见 TestFlow.binning_path）


class TestFlow:
    """SETUP → EXECUTE → BINNING 流程容器。"""

    __test__ = False  # 防止 pytest 误收集（类名以 Test 开头）

    REQUIRED_ORDER = ("SETUP", "EXECUTE", "BINNING")

    def __init__(self, phases: list[FlowPhase]):
        names = tuple(p.name for p in phases)
        if names != self.REQUIRED_ORDER:
            raise ValueError(f"流程阶段必须为 {self.REQUIRED_ORDER}，实际 {names}")
        self.phases = phases

    @property
    def setup_items(self) -> list[str]:
        return self.phases[0].items

    @property
    def exec_items(self) -> list[str]:
        """EXECUTE 阶段测试项（按执行顺序）。"""
        return self.phases[1].items

    @property
    def binning_path(self) -> str:
        """BINNING 阶段引用的 bin 表路径（plan yaml 内）。"""
        return self.phases[2].items[0]

    @classmethod
    def from_yaml(cls, flow_cfg: list[dict]) -> "TestFlow":
        phases = []
        for p in flow_cfg:
            items = p.get("items")
            if items is None:               # BINNING 阶段：table 路径
                items = [p["table"]]
            phases.append(FlowPhase(name=p["name"], items=list(items)))
        return cls(phases)
