"""参数分布图：统计与 parquet 一致、测试限在规格限内侧、产出文件齐全。"""

import json

import numpy as np
import pytest

from char import distribution
from testexec.plan import TestPlan


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("dist")
    return distribution.run(out_dir=out_dir)


def test_stats_match_parquet(result):
    """分布统计必须与 parquet 原始数据一致（重算比对）。"""
    cp = distribution.load_data()
    good = cp[cp["hard_bin"] == 1]
    for name, st in result["stats"].items():
        v = good[name].to_numpy(dtype=float)
        assert st["n_good"] == len(v)
        assert np.isclose(st["mean"], v.mean(), rtol=0, atol=1e-9)
        assert np.isclose(st["std"], v.std(), rtol=0, atol=1e-9)


def test_test_limits_inside_spec():
    """guardband 后测试限必须严格内缩在规格限内侧（W2 口径复核）。"""
    plan = TestPlan.load()
    for name in distribution.PARAMS_TO_PLOT:
        item = plan.item_by_name[name]
        if item.spec_lo is not None:
            assert item.test_lo is not None
            assert item.spec_lo < item.test_lo
        if item.spec_hi is not None:
            assert item.test_hi is not None
            assert item.test_hi < item.spec_hi


def test_stats_have_all_limit_fields(result):
    for name, st in result["stats"].items():
        for field in ("spec_lo", "spec_hi", "test_lo", "test_hi",
                      "margin_to_test_limit_sigma", "n_out_of_spec"):
            assert field in st, f"{name} 缺字段 {field}"
        assert st["margin_to_test_limit_sigma"] > 0, \
            f"{name} 测试限余量必须为正（guardband 不应吞掉全部余量）"


def test_plots_written(result):
    for name in distribution.PARAMS_TO_PLOT:
        path = result["paths"][name]
        assert path.exists() and path.stat().st_size > 0
    summary = json.loads(
        (result["paths"]["VOUT1"].parent / "dist_summary.json").read_text())
    assert set(summary) == set(distribution.PARAMS_TO_PLOT)
