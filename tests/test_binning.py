"""binning 判定正确性：soft bin 映射、hard bin 汇总与优先级。"""

from testexec.binning import BinningTable
from testexec.plan import TestPlan


def _table() -> BinningTable:
    plan = TestPlan.load()
    return plan.bins


def test_soft_bin_param_mapping():
    bins = _table()
    assert bins.soft_bin(None).sbin == 1            # 全过 → SBin1
    assert bins.soft_bin("VOUT1").sbin == 11        # 文档示例：VOUT1 fail → SBin 11
    assert bins.soft_bin("POR").sbin == 22
    for f in ("FUNC_REG", "FUNC_EN", "FUNC_OCP"):
        assert bins.soft_bin(f).sbin == 3           # 功能 fail → SBin3


def test_hard_bin_precedence():
    bins = _table()
    assert bins.hard_bin(param_fail=False, func_fail=False).hbin == 1
    assert bins.hard_bin(param_fail=True, func_fail=False).hbin == 2
    assert bins.hard_bin(param_fail=False, func_fail=True).hbin == 3
    # 参数+功能都 fail：功能优先 → HBin3
    assert bins.hard_bin(param_fail=True, func_fail=True).hbin == 3


def test_binning_table_covers_all_exec_items():
    plan = TestPlan.load()
    for item in plan.items:
        sb = plan.bins.soft_bin(item.name)
        assert sb is not None and sb.item is not None, f"{item.name} 未映射 soft bin"


def test_all_pass_gives_bin1():
    """全过 die → soft=1, hard=1（无缺陷小批走完整判定链验证一致性）。"""
    from dut import wafer
    from testexec import cp
    plan = TestPlan.load()
    clean, _ = wafer.generate_lot(d0=0.0)
    clean = clean.head(2000)
    results = cp.run_cp(clean, plan)
    # 无缺陷批通过率 >99%，HBin1 占绝对主体
    assert (results["hard_bin"] == 1).mean() > 0.99
    # soft/hard 一致性：hard=1 ⟺ soft=1
    assert ((results["hard_bin"] == 1) == (results["soft_bin"] == 1)).all()
