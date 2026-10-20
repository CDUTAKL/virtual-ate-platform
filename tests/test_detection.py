"""检出率统计（面试关键口径）：
- CP 可检出缺陷（DEF_OFFSET/DEF_LEAK/DEF_OPEN）检出率 ≥99%；
- 潜伏缺陷（DEF_DRIFT/DEF_TIMING）逃逸数如实统计（W1 已知边界，W4 素材）。
"""

from dut import wafer
from testexec import cp
from testexec.plan import TestPlan


def _full_run():
    plan = TestPlan.load()
    lot, _ = wafer.generate_lot()
    results = cp.run_cp(lot, plan)
    return plan, lot, results


def test_detectable_defects_caught_99pct():
    """CP 可检出缺陷检出率 ≥99%（漏检只应来自 guardband 边界的极少数）。"""
    _, lot, results = _full_run()
    rep = cp.detection_report(results, lot)
    assert rep["detection_rate"] >= 0.99, \
        f"检出率 {rep['detection_rate']:.2%} 低于 99%"


def test_escapes_are_latent_only():
    """逃逸 = 仅潜伏型缺陷：DRIFT/TIMING；可检出类型零逃逸。"""
    _, lot, results = _full_run()
    fail = results["hard_bin"].to_numpy() != 1
    labels = lot["defect"].to_numpy()
    for kind in cp.DETECTABLE_LABELS:
        escapes = int(((labels == kind) & ~fail).sum())
        assert escapes == 0, f"{kind} 不应有逃逸，实际 {escapes}"
    rep = cp.detection_report(results, lot)
    assert rep["latent_escapes"] > 0
    assert rep["latent_escapes"] <= rep["latent_total"]


def test_overkill_is_small():
    """guardband 过杀率：无缺陷 die 被误杀比例应 <2%（报告实际值）。"""
    _, lot, results = _full_run()
    rep = cp.detection_report(results, lot)
    assert rep["overkill_rate"] < 0.02, \
        f"过杀率 {rep['overkill_rate']:.2%} 异常高，检查 guardband k 与 σ_meas"


def test_detection_report_numbers_consistent():
    """检出率报告数字自洽：检出+逃逸+过杀分区不重叠、总数对齐。"""
    _, lot, results = _full_run()
    rep = cp.detection_report(results, lot)
    labels = lot["defect"].to_numpy()
    n_labelled = int((labels != "").sum())
    assert rep["detectable_caught"] + rep["latent_escapes"] <= n_labelled
    # 6 颗潜伏 die 被自然波动顺带杀掉（W1 已知：15 颗自然尾部 fail）
    assert rep["latent_total"] - rep["latent_escapes"] <= 20
