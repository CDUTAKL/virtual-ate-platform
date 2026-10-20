"""从 configs/cp_plan.yaml 构建 CpMeta（stdf/writer.py 的元数据）。

只读取配置（不修改），复用 testexec.plan.TestPlan 的加载与 guardband
计算，保证 STDF 里的测试限与 CP 执行判定完全一致。
"""

from __future__ import annotations

from .writer import CpMeta


def build_cp_meta(plan_path: str = "configs/cp_plan.yaml") -> CpMeta:
    from testexec.plan import TestPlan          # 只读复用，保持单一事实来源
    plan = TestPlan.load(plan_path)
    items = {}
    pins = []
    for pmr_indx, it in enumerate(plan.items, start=1):
        items[it.name] = {
            "test_num": it.test_num, "unit": it.unit or "",
            "test_lo": it.test_lo, "test_hi": it.test_hi,
            "spec_lo": it.spec_lo, "spec_hi": it.spec_hi,
            "meas_time_s": it.meas_time_s, "kind": it.kind,
        }
        pins.append((pmr_indx, it.pin or it.name, it.name))
    return CpMeta(
        lot_id=plan.lot_id, program=plan.program,
        pins=pins, items=items, exec_order=list(plan.flow.exec_items),
    )
