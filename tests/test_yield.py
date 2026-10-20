"""良率分析：yield/FPY/DPPM 数学 + bin 帕累托累计。"""

import importlib

import numpy as np
import pandas as pd

ayield = importlib.import_module("analytics.yield")


def _synthetic() -> pd.DataFrame:
    """1000 die 合成结果：800 过 / 150 参数 fail(SBin11) / 50 功能 fail(SBin3)。"""
    return pd.DataFrame({
        "hard_bin": [1] * 800 + [2] * 150 + [3] * 50,
        "soft_bin": [1] * 800 + [11] * 150 + [3] * 50,
        "soft_bin_name": ["PASS"] * 800 + ["VOUT1_FAIL"] * 150 + ["FUNC_FAIL"] * 50,
    })


def test_yield_math():
    s = ayield.summary(_synthetic())
    assert s["total"] == 1000 and s["good"] == 800 and s["bad"] == 200
    assert s["yield"] == 0.8 and s["fpy"] == 0.8
    assert s["dppm"] == 200_000


def test_bin_pareto_cumulative():
    _, soft = ayield.bin_pareto(_synthetic())
    assert np.isclose(soft["cum_pct"].iloc[-1], 1.0)
    top = soft.iloc[0]
    assert top["soft_bin"] == 1 and top["count"] == 800


def test_main_lot_yield_matches_w2():
    """主 lot 良率与 W2 CP 摘要一致（90.35%，STDF round-trip 后重算）。"""
    from dut import wafer
    from testexec.plan import TestPlan
    from testexec import cp
    from stdf.cp_meta import build_cp_meta
    from stdf.writer import write_cp_stdf
    from stdf.reader import read_stdf
    import tempfile, pathlib
    plan = TestPlan.load()
    lot, _ = wafer.generate_lot()
    results = cp.run_cp(lot.head(3000), plan)
    path = pathlib.Path(tempfile.mkdtemp()) / "y.stdf"
    write_cp_stdf(results, build_cp_meta(), path, now=1760000000)
    frame = read_stdf(path).to_results_frame()
    s = ayield.summary(frame)
    assert s["total"] == 3000
    assert 0.80 < s["yield"] < 0.98
    assert s["dppm"] == (s["bad"] / s["total"]) * 1e6
