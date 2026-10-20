"""HTML 测试报告（文档 §3.8 报告层）：文件生成、KPI 数字一致性、base64 内联。"""

import json

import pandas as pd
import pytest

from dut import wafer
from report import html_report
from stdf.cp_meta import build_cp_meta
from stdf.writer import write_cp_stdf
from testexec import cp
from testexec.plan import TestPlan

#: 1×1 像素 PNG（图表占位，base64 内联验证用）
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000d49444154789c626001000000ffff030000060005"
    "57bfabd40000000049454e44ae426082")


@pytest.fixture()
def report_env(tmp_path):
    """小型一致 fixture：800 die 结果 + 小 STDF + 各 JSON/CSV + 7 张占位图。"""
    plan = TestPlan.load()
    lot, _ = wafer.generate_lot()
    small_lot = lot.head(800).reset_index(drop=True)
    results = cp.run_cp(small_lot, plan)
    stdf_path = tmp_path / "lot.stdf"
    write_cp_stdf(results, build_cp_meta(), stdf_path, now=1760000000)

    n_hard = results["hard_bin"].nunique()
    cpk_table = pd.DataFrame([
        {"item": "VOUT3", "units": "V", "cpk": 1.02, "ppk": 1.00, "judgement": "偏低"},
        {"item": "IQ1", "units": "uA", "cpk": 1.56, "ppk": 1.53, "judgement": "合格"},
    ])
    fa_report = {"yield": 0.9, "diagonal_hit_rate": 0.9648,
                 "method": {"silhouette_score": 0.5056}}
    shmoo_summary = {"dies": {"DEF_TIMING": {"fail_corners_vt": [[85.0, 3.6], [125.0, 5.0]]}},
                     "fixed": {"vt_load_a": 2.0}}
    time_report = {"baseline_s": 0.46, "optimized_s": 0.46, "saved_pct": 0.0}
    images = {}
    for i, title in enumerate(("wafermap", "pareto", "spc", "corr",
                               "shmoo_vt", "shmoo_vl", "dist")):
        p = tmp_path / f"{title}.png"
        p.write_bytes(_TINY_PNG)
        images[title] = p
    out = html_report.generate_report(
        tmp_path / "report.html", results=results, lot=small_lot,
        cpk_table=cpk_table, fa_report=fa_report, shmoo_summary=shmoo_summary,
        time_report=time_report, stdf_path=stdf_path, images=images,
        gen_date="2026-10-21")
    return {"out": out, "results": results, "lot": small_lot,
            "stdf_path": stdf_path, "cpk_table": cpk_table}


def test_report_file_generated(report_env):
    """文件生成：存在、非空、含标题与闭合标签。"""
    out = report_env["out"]
    text = out.read_text(encoding="utf-8")
    assert out.exists() and len(text) > 5000
    assert "PMU300 CP 测试报告" in text
    assert "</html>" in text


def test_kpi_numbers_match_inputs(report_env):
    """KPI 数字与输入一致：良率双口径、检出率、Cpk 最低参数、STDF 大小。"""
    text = report_env["out"].read_text(encoding="utf-8")
    res, lot = report_env["results"], report_env["lot"]
    y_test = (res["hard_bin"] == 1).mean()
    y_spec = lot["pass"].mean()
    assert f"{y_test * 100:.2f}%" in text, "测试限良率 KPI 不符"
    assert f"{y_spec * 100:.2f}%" in text, "规格限良率 KPI 不符"
    assert "VOUT3" in text and "1.02" in text, "最低 Cpk 参数 KPI 不符"
    assert "96.48%" in text, "FA 对角线 KPI 不符"
    stdf_mb = report_env["stdf_path"].stat().st_size / 1e6
    assert f"{stdf_mb:.1f} MB" in text, "STDF 大小 KPI 不符"
    assert "T≥85°C" in text, "Shmoo 失效边界 KPI 不符"
    assert "460 ms" in text, "单 die 测试时间 KPI 不符"


def test_nan_defect_labels_equivalent(report_env):
    """CSV 读回时空标签为 NaN：归一化后与内存空串结果完全一致（KPI 不回退）。"""
    import numpy as np
    base = report_env["out"].read_text(encoding="utf-8")
    parent = report_env["out"].parent
    kw = dict(
        results=report_env["results"], cpk_table=report_env["cpk_table"],
        fa_report={"yield": 0.9, "diagonal_hit_rate": 0.9648,
                   "method": {"silhouette_score": 0.5056}},
        shmoo_summary={"dies": {"DEF_TIMING": {"fail_corners_vt": [[85.0, 3.6]]}},
                       "fixed": {"vt_load_a": 2.0}},
        time_report={"baseline_s": 0.46, "optimized_s": 0.46, "saved_pct": 0.0},
        stdf_path=report_env["stdf_path"],
        images={k: parent / f"{k}.png"
                for k in ("wafermap", "pareto", "spc", "corr",
                          "shmoo_vt", "shmoo_vl", "dist")},
        gen_date="2026-10-21")
    lot_nan = report_env["lot"].copy()
    lot_nan["defect"] = lot_nan["defect"].replace("", np.nan)
    out_nan = html_report.generate_report(parent / "r_nan.html", lot=lot_nan, **kw)
    assert out_nan.read_text(encoding="utf-8") == base


def test_images_base64_inlined(report_env):
    """7 张图全部 base64 内联（无外部引用）。"""
    text = report_env["out"].read_text(encoding="utf-8")
    assert text.count("data:image/png;base64,") == 7
    assert 'src="http' not in text and 'src="./' not in text


def test_stdf_record_count_matches_parse(report_env):
    """记录数快速统计与全量解析一致（含 PIR/PRR/PTR/FTR）。"""
    from stdf.reader import count_stdf_records, read_stdf
    path = report_env["stdf_path"]
    assert count_stdf_records(path) == len(read_stdf(path).records)
    n = count_stdf_records(path)
    assert n > 800          # 800 die 的 PIR/PRR 之外还有头/尾记录
