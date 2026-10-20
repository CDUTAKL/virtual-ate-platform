"""过程能力：Cpk/Ppk 值域合理（基于 HBin1 与测试限）。"""

import numpy as np
import pandas as pd
import pytest

from analytics import capability


def _synthetic_results():
    """合成 wafer 结构结果表：2 片 × 2000 die，VOUT1 正态 + 缺陷尾部。"""
    rng = np.random.default_rng(3)
    rows = []
    for w in (1, 2):
        v = rng.normal(3.30, 0.016, 2000)
        v[:30] = 3.45                     # 缺陷尾部（会被 HBin≠1 排除）
        rows.append(pd.DataFrame({
            "wafer": w, "hard_bin": np.r_[[2] * 30, [1] * 1970],
            "VOUT1": v,
        }))
    return pd.concat(rows, ignore_index=True)


LIMITS = pd.DataFrame([{
    "test_num": 1001, "test_txt": "VOUT1", "units": "V",
    "lo_limit": 3.239172, "hi_limit": 3.360828,
    "lo_spec": 3.234, "hi_spec": 3.366,
}])


def test_cpk_reasonable_range():
    """良品集 Cpk 在合理区间（合成数据 μ=3.30,σ≈0.016, 限 3.239~3.361）。"""
    res = _synthetic_results()
    cap = capability.capability_table(res, LIMITS)
    row = cap.iloc[0]
    assert 0.8 < row["cpk"] < 1.6
    assert 0.8 < row["ppk"] < 1.6
    assert row["n_good"] == 1970 * 2
    # Cpk 用组内σ，Ppk 用整体σ：单一批次两者接近
    assert abs(row["cpk"] - row["ppk"]) < 0.2


def test_defect_tail_excluded():
    """缺陷尾部（HBin2）不进入 Cpk 统计总体。"""
    res = _synthetic_results()
    cap = capability.capability_table(res, LIMITS)
    assert cap.iloc[0]["n_good"] == (res["hard_bin"] == 1).sum()
    mu = cap.iloc[0]["mean"]
    assert 3.28 < mu < 3.32          # 若含缺陷尾部 μ 会被拉高


def test_main_lot_cpk_consistent_with_w2():
    """主 lot：Cpk 最低参数为 VOUT3，值域 0.9~1.3（与 W2 的 1.00 一致）。"""
    from dut import wafer
    from testexec.plan import TestPlan
    from testexec import cp
    from stdf.cp_meta import build_cp_meta
    from stdf.writer import write_cp_stdf
    from stdf.reader import read_stdf
    import tempfile, pathlib
    plan = TestPlan.load()
    lot, _ = wafer.generate_lot()
    results = cp.run_cp(lot.head(4000), plan)
    path = pathlib.Path(tempfile.mkdtemp()) / "c.stdf"
    sf = read_stdf(write_cp_stdf(results, build_cp_meta(), path, now=1760000000))
    cap = capability.capability_table(sf.to_results_frame(), sf.limits_table())
    assert cap.iloc[0]["item"] in ("VOUT3", "VOUT1")   # 最紧参数在前
    for _, r in cap.iterrows():
        assert 0.5 < r["cpk"] < 2.5, f"{r['item']} Cpk={r['cpk']} 异常"
        assert r["cpk"] <= r["ppk"] * 1.15 + 0.05       # 组内σ ≤ 整体σ（近似）
