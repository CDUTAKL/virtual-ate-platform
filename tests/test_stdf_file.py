"""STDF 文件级测试：从 CP 结果写文件 → reader 解析 → 结构/数值一致性 + 性能。"""

import struct
import time

import numpy as np
import pandas as pd
import pytest

from dut import wafer
from testexec.plan import TestPlan
from testexec import cp
from stdf.cp_meta import build_cp_meta
from stdf.writer import write_cp_stdf
from stdf.reader import read_stdf


@pytest.fixture(scope="module")
def small_env(tmp_path_factory):
    """3 片 wafer 的小结果集 + 已写好的 STDF 文件。"""
    plan = TestPlan.load()
    meta = build_cp_meta()
    lot, _ = wafer.generate_lot()
    small_lot = lot[lot["wafer"] <= 3].reset_index(drop=True)
    results = cp.run_cp(small_lot, plan)
    path = tmp_path_factory.mktemp("stdf") / "small.stdf"
    write_cp_stdf(results, meta, path, now=1760000000)
    return results, path


def test_file_header_valid(small_env):
    """文件头合法性：首记录 FAR(0,10)，REC_LEN/版本正确（标准：REC_LEN 只计数据体）。"""
    _, path = small_env
    head = path.read_bytes()[:6]
    rec_len = struct.unpack(">H", head[:2])[0]
    assert rec_len == 2                       # FAR 数据体 = CPU_TYPE + STDF_VER
    assert head[2] == 0 and head[3] == 10     # FAR (0,10)
    assert path.read_bytes()[:6] == bytes.fromhex("00 02 00 0A 02 04")
    sf = read_stdf(path)
    assert sf.first("FAR")["STDF_VER"] == 4


def test_record_counts_match_dies(small_env):
    """记录计数：PIR=PRR=die 数，PTR=12×die，FTR=3×die，WIR=WRR=wafer 数。"""
    results, path = small_env
    sf = read_stdf(path)
    n = len(results)
    assert sf.count("PIR") == n
    assert sf.count("PRR") == n
    assert sf.count("PTR") == 12 * n
    assert sf.count("FTR") == 3 * n
    assert sf.count("WIR") == results["wafer"].nunique()
    assert sf.count("WRR") == results["wafer"].nunique()
    assert sf.count("TSR") == 15
    assert sf.count("PMR") == 15


def test_roundtrip_values_and_bins(small_env):
    """round-trip：坐标/bin/测量值与源结果表一致（R4 精度内）。"""
    results, path = small_env
    sf = read_stdf(path)
    frame = sf.to_results_frame()
    assert len(frame) == len(results)
    assert (frame["wafer"].to_numpy() == results["wafer"].to_numpy()).all()
    assert (frame["x"].to_numpy() == results["x"].to_numpy()).all()
    assert (frame["y"].to_numpy() == results["y"].to_numpy()).all()
    assert (frame["hard_bin"].to_numpy() == results["hard_bin"].to_numpy()).all()
    assert (frame["soft_bin"].to_numpy() == results["soft_bin"].to_numpy()).all()
    for p in ("VOUT1", "VOUT3", "IQ1", "RIPPLE3"):
        a = frame[p].to_numpy(dtype=float)
        b = results[p].to_numpy(dtype=float)
        assert np.allclose(a, b, rtol=1e-5, atol=1e-6), f"{p} round-trip 超差"
    # 功能项 PASS 标志从 FTR TEST_FLG 还原
    for f in ("FUNC_REG", "FUNC_EN", "FUNC_OCP"):
        assert (frame[f].to_numpy() == results[f"{f}_pass"].to_numpy()).all()


def test_limits_from_ptr(small_env):
    """PTR 携带测试限/规格限：分析层 Cpk 的唯一限值来源。"""
    results, path = small_env
    sf = read_stdf(path)
    lim = sf.limits_table().set_index("test_txt")
    assert np.isclose(lim.loc["VOUT1", "lo_limit"], 3.239172, atol=1e-4)
    assert np.isclose(lim.loc["VOUT1", "hi_limit"], 3.360828, atol=1e-4)
    assert np.isclose(lim.loc["VOUT1", "lo_spec"], 3.234, atol=1e-4)
    assert lim.loc["IQ1", "lo_limit"] is None or np.isnan(lim.loc["IQ1", "lo_limit"])
    assert np.isclose(lim.loc["IQ1", "hi_limit"], 119.852, atol=1e-2)


def test_mir_sdr_tsr_content(small_env):
    results, path = small_env
    sf = read_stdf(path)
    mir = sf.first("MIR")
    assert mir["LOT_ID"] == "LOT20261020A"
    assert mir["PART_TYP"] == "PMU300"
    assert mir["TEST_COD"] == "CP"
    sdr = sf.first("SDR")
    assert sdr["SITE_NUM"] == [1]
    tsr = {r["TEST_NAM"]: r for r in sf.by_type(10, 30)}
    assert set(tsr) == {"VOUT1", "IQ1", "LINE1", "VOUT2", "IQ2", "VOUT3",
                        "EFF3", "RIPPLE3", "ADC_INL", "TEMP25", "ISD", "POR",
                        "FUNC_REG", "FUNC_EN", "FUNC_OCP"}
    vout1_tsr = tsr["VOUT1"]
    assert vout1_tsr["EXEC_CNT"] == len(results)
    assert vout1_tsr["FAIL_CNT"] == int((~results["VOUT1_pass"]).sum())


def test_full_lot_write_read_performance(tmp_path):
    """整 lot（5 万 die）STDF 写出 + 读回性能：< 90s（预算 <2min 的一半）。"""
    plan = TestPlan.load()
    meta = build_cp_meta()
    lot, _ = wafer.generate_lot()
    results = cp.run_cp(lot, plan)
    path = tmp_path / "lot.stdf"
    t0 = time.perf_counter()
    write_cp_stdf(results, meta, path, now=1760000000)
    t_write = time.perf_counter() - t0
    t0 = time.perf_counter()
    sf = read_stdf(path)
    t_read = time.perf_counter() - t0
    size_mb = path.stat().st_size / 1e6
    # 结构性记录数（每 die 固定条数，与平台/数据无关，必须精确）
    assert sf.count("PRR") == 50_000
    assert sf.count("PTR") == 600_000
    assert sf.count("FTR") == 150_000
    assert sf.count("PIR") == 50_000
    # bin 表（HBR/SBR）按实际出现的 bin 组合发射：guardband 边界 die 的
    # 浮点判定随平台 BLAS 累加顺序可能翻转，bin 表条数与总记录数随之
    # 差 ±1（macOS 实测 850,098 / Linux CI 850,099）。故总数只守宽松上界，
    # 精确值记入下方打印输出。
    header = len(sf.records) - (50_000 + 600_000 + 150_000 + 50_000)
    assert header <= 160, f"头/尾派生记录数 {header} 超出预期上界"
    assert t_write < 60 and t_read < 60, \
        f"STDF 写出 {t_write:.1f}s / 读回 {t_read:.1f}s 超预算"
    # 文件大小报告（供 README 实录）
    print(f"\nSTDF 全 lot: {size_mb:.1f} MB, records {len(sf.records)}, "
          f"write {t_write:.1f}s, read {t_read:.1f}s")
