"""Shmoo 扫描正确性：已知缺陷 die 在其失效角必须 fail、网格形态、产出文件。"""

import json
from pathlib import Path

import numpy as np
import pytest

from char import shmoo
from dut import config
from dut.pmu300 import make_die, PARAM_SPECS
from dut.defects import Defect, make_defect


@pytest.fixture(scope="module")
def dies():
    return shmoo.pick_shmoo_dies()


def test_nominal_die_passes_all_corners(dies):
    """典型好 die：100 个角全 pass。"""
    for t in shmoo.GRID_TEMP:
        for l in shmoo.GRID_LOAD:
            ok, _v, _r = shmoo.corner_pass(dies["nominal"], t, l)
            assert ok, f"nominal die 在 T={t}°C, ILOAD={l}A 不应 fail"


def test_timing_die_fails_at_stress_corner(dies):
    """DEF_TIMING die：高温重载角（125 °C / 2 A）必须 fail。"""
    for vin in shmoo.GRID_VIN:  # V 轴全范围（模型中 VIN 与触发解耦）
        ok, v, ripple = shmoo.corner_pass(dies["DEF_TIMING"], 125.0, 2.0)
        assert not ok
        # 物理口径：震荡不移动 DC 工作点，失效承载在纹波/OCP 上
        assert shmoo.VOUT3_WINDOW[0] <= v <= shmoo.VOUT3_WINDOW[1]
        assert ripple > shmoo.RIPPLE3_LIMIT


def test_timing_die_passes_at_room(dies):
    """DEF_TIMING die：常温常规负载（25 °C / 1 A）必须 pass（潜伏）。"""
    ok, _v, _r = shmoo.corner_pass(dies["DEF_TIMING"], 25.0, 1.0)
    assert ok


def test_drift_die_fails_at_temp_extreme(dies):
    """DEF_DRIFT(CH3) die：高温角（125 °C）输出电压必须越限。"""
    ok, v, _r = shmoo.corner_pass(dies["DEF_DRIFT"], 125.0, 2.0)
    assert not ok
    assert not (shmoo.VOUT3_WINDOW[0] <= v <= shmoo.VOUT3_WINDOW[1])


def test_corner_pass_judgement_rule():
    """判定规则本身：VOUT3 越窗/纹波超限/OCP fail 各自独立导致 fail。"""
    rng = np.random.default_rng(config.SEED)
    # 构造 VOUT3 基值越出窗口的 die（直接抬高 base25）
    die = make_die(rng, None)
    die.base25["VOUT3"] = 1.25  # > 1.224
    ok, _v, _r = shmoo.corner_pass(die, 25.0, 1.0)
    assert not ok
    # 纹波超限
    die2 = make_die(rng, None)
    die2.base25["RIPPLE3"] = 60.0
    ok, _v, _r = shmoo.corner_pass(die2, 25.0, 1.0)
    assert not ok


def test_scan_grid_shape_and_binary(dies):
    grids = shmoo.scan_shmoo(dies["nominal"])
    assert grids["vt"].shape == (len(shmoo.GRID_TEMP), len(shmoo.GRID_VIN))
    assert grids["vl"].shape == (len(shmoo.GRID_LOAD), len(shmoo.GRID_VIN))
    for key in ("vt", "vl"):
        assert set(np.unique(grids[key])).issubset({0, 1})


def test_run_writes_outputs(tmp_path):
    out = shmoo.run(tmp_path)
    assert (tmp_path / "shmoo_vt.png").exists()
    assert (tmp_path / "shmoo_vl.png").exists()
    summary = json.loads((tmp_path / "shmoo_summary.json").read_text())
    assert "DEF_TIMING" in summary["conclusion"]
    # 网格与结论自洽：timing die 在 V×T 图上确有 fail 角
    assert summary["dies"]["DEF_TIMING"]["fail_corners_vt"], \
        "DEF_TIMING die 必须在高温重载角 fail（结论的依据）"
    assert out["results"]["DEF_TIMING"]["vt"].sum() < \
        out["results"]["nominal"]["vt"].sum()
