"""binning：soft/hard bin 定义与判定（设计文档 §3.3）。

规则（照文档）：
- 每个测试项 fail → 映射 soft bin（如 VOUT1 fail → SBin 11）；
  一颗 die 多项 fail 时，soft bin 取 EXECUTE 顺序中首个 fail 项。
- hard bin 汇总：全过 → HBin 1；任一参数 fail → HBin 2；功能 fail → HBin 3。
  功能 fail 优先于参数 fail（功能失效更接近封装级问题，单独归 bin 便于 FA）。
"""

from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class SoftBin:
    sbin: int
    name: str
    item: str | None          # 对应测试项名；None=PASS；'FUNC_ANY'=任一功能项


@dataclass(frozen=True)
class HardBin:
    hbin: int
    name: str


class BinningTable:
    """soft/hard bin 表：加载 bins.yaml 并提供判定接口。"""

    def __init__(self, soft_bins: list[SoftBin], hard_bins: list[HardBin],
                 func_items: list[str]):
        self.soft_bins = soft_bins
        self.hard_bins = hard_bins
        # 测试项名 → soft bin（FUNC_ANY 展开为各功能项）
        self._sbin_by_item: dict[str, SoftBin] = {}
        for sb in soft_bins:
            if sb.item is None:
                continue
            if sb.item == "FUNC_ANY":
                for f in func_items:
                    self._sbin_by_item[f] = sb
            else:
                self._sbin_by_item[sb.item] = sb
        self._pass_bin = next(sb for sb in soft_bins if sb.item is None)
        self._hard = {hb.hbin: hb for hb in hard_bins}

    @classmethod
    def load(cls, path: str, func_items: list[str]) -> "BinningTable":
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        soft = [SoftBin(**row) for row in cfg["soft_bins"]]
        hard = [HardBin(**row) for row in cfg["hard_bins"]]
        return cls(soft, hard, func_items)

    def soft_bin(self, first_fail_item: str | None) -> SoftBin:
        """首个 fail 测试项 → soft bin；全过 → PASS bin。"""
        if first_fail_item is None:
            return self._pass_bin
        return self._sbin_by_item[first_fail_item]

    def hard_bin(self, param_fail: bool, func_fail: bool) -> HardBin:
        """hard bin 汇总：功能 fail 优先，其次参数 fail，全过 HBin1。"""
        if func_fail:
            return self._hard[3]
        if param_fail:
            return self._hard[2]
        return self._hard[1]
