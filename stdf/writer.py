"""STDF V4 writer：从 CP 结果表生成完整 CP 数据文件（设计文档 §3.4）。

文件布局（整 lot 单文件）：
    FAR ATR MIR SDR PMR×N PCR HBR×N SBR×N
    （每 wafer：WIR → [每 die：PIR PRR PTR×12 FTR×3] → WRR）
    TSR×15 MRR

writer 不接触缺陷标签（测试程序视角），只消费 testexec 产出的结果表 + 元数据。
自检：stdf/reader.py 读回逐字段比对（round-trip 测试覆盖）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .records import Record


@dataclass
class CpMeta:
    """写 STDF 所需的批次/程序/引脚/限值元数据（由调用方构建，见 cp_meta.py）。"""

    lot_id: str
    program: str
    part_type: str = "PMU300"
    node_name: str = "VIRTUAL_FAB"
    tester_type: str = "VIRTUAL-ATE"
    job_name: str = "PMU300_CP"
    operator: str = "virtual-te"
    test_temp: str = "25C"
    exec_type: str = "PY-TESTEXEC"
    exec_ver: str = "1.2"
    # 引脚映射 PMR：[(pmr_indx, phys, log)]
    pins: list[tuple[int, str, str]] = field(default_factory=list)
    # 测试项定义 name → dict(test_num, unit, test_lo, test_hi, spec_lo, spec_hi,
    #                         meas_time_s, kind)
    items: dict[str, dict] = field(default_factory=dict)
    # EXECUTE 顺序（soft bin 首个 fail 的判定顺序也用它）
    exec_order: list[str] = field(default_factory=list)


def write_cp_stdf(results: pd.DataFrame, meta: CpMeta, path: str | Path,
                  now: int | None = None) -> Path:
    """从 CP 结果表（testexec.cp.run_cp 输出）写完整 STDF V4 文件。

    now : 固定时间戳（epoch s）便于可复现；None 用当前时间。
    """
    t0 = now if now is not None else int(time.time())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    param_items = [n for n in meta.exec_order if meta.items[n]["kind"] == "PARM"]
    func_items = [n for n in meta.exec_order if meta.items[n]["kind"] == "FUNC"]
    hard = results["hard_bin"].to_numpy()
    soft = results["soft_bin"].to_numpy()

    out = bytearray()

    def emit(rec: Record) -> None:
        out.extend(rec.to_bytes())

    # ---- 文件头与批次信息 ----
    emit(Record(0, 10, {"CPU_TYPE": 2, "STDF_VER": 4}))      # FAR：大端（SUN），V4
    emit(Record(0, 20, {"MOD_TIM": t0, "CMD_LINE": f"python {meta.program}"}))
    emit(Record(1, 10, {
        "SETUP_T": t0, "START_T": t0, "STAT_NUM": 1,
        "LOT_ID": meta.lot_id, "PART_TYP": meta.part_type,
        "NODE_NAM": meta.node_name, "TSTR_TYP": meta.tester_type,
        "JOB_NAM": meta.job_name, "OPER_NAM": meta.operator,
        "EXEC_TYP": meta.exec_type, "EXEC_VER": meta.exec_ver,
        "TEST_COD": "CP", "TST_TEMP": meta.test_temp,
        "PKG_TYP": "QFN24",
    }))
    emit(Record(1, 80, {"HEAD_NUM": 1, "SITE_GRP": 1, "SITE_NUM": [1],
                        "HAND_TYP": "VIRTUAL_PROBER"}))
    for pmr_indx, phys, log in meta.pins:
        emit(Record(1, 60, {"PMR_INDX": pmr_indx, "CHAN_TYP": 0,
                            "CHAN_NAM": "", "PHY_NAM": phys,
                            "LOG_NAM": log, "HEAD_NUM": 1, "SITE_NUM": 1}))
    good_total = int((hard == 1).sum())
    emit(Record(1, 30, {"HEAD_NUM": 1, "SITE_NUM": 1, "PART_CNT": len(results),
                        "RTST_CNT": 0, "ABRT_CNT": 0, "GOOD_CNT": good_total,
                        "FUNC_CNT": good_total}))

    # ---- HBR / SBR bin 表 ----
    for hbin, cnt in sorted(pd.Series(hard).value_counts().items()):
        name = {1: "GOOD", 2: "PARAM_FAIL", 3: "FUNC_FAIL"}.get(int(hbin), f"H{hbin}")
        emit(Record(1, 40, {"HEAD_NUM": 1, "SITE_NUM": 1, "HBIN_NUM": int(hbin),
                            "HBIN_CNT": int(cnt), "HBIN_PF": "P" if hbin == 1 else "F",
                            "HBIN_NAM": name}))
    soft_counts = results.groupby(["soft_bin", "soft_bin_name"]).size()
    for (sbin, sname), cnt in sorted(soft_counts.items()):
        emit(Record(1, 50, {"HEAD_NUM": 1, "SITE_NUM": 1, "SBIN_NUM": int(sbin),
                            "SBIN_CNT": int(cnt),
                            "SBIN_PF": "P" if sbin == 1 else "F",
                            "SBIN_NAM": str(sname)}))

    # ---- 逐 wafer：WIR → die 流 → WRR ----
    meas_cols = {n: results[n].to_numpy(dtype=float) for n in param_items}
    func_cols = {n: results[f"{n}_pass"].to_numpy(dtype=bool) for n in func_items}
    test_ms = int(round(float(results["test_time_s"].iloc[0]) * 1000))
    die_seq = 0
    for wafer, grp in results.groupby("wafer", sort=True):
        idx = grp.index.to_numpy()
        emit(Record(2, 10, {"HEAD_NUM": 1, "SITE_GRP": 1, "START_T": t0,
                            "WAFER_ID": f"W{int(wafer):02d}"}))
        for row_i in idx:
            die_seq += 1
            hbin = int(hard[row_i])
            sbin = int(soft[row_i])
            fail = hbin != 1
            emit(Record(5, 10, {"HEAD_NUM": 1, "SITE_NUM": 1}))
            # PTR × 12
            for name in param_items:
                it = meta.items[name]
                v = float(meas_cols[name][row_i])
                passed = bool(results[f"{name}_pass"].iloc[row_i])
                emit(Record(15, 10, {
                    "TEST_NUM": it["test_num"], "HEAD_NUM": 1, "SITE_NUM": 1,
                    "TEST_FLG": 0x01 if not passed else 0x00,
                    "PARM_FLG": 0x00,
                    "RESULT": v, "TEST_TXT": name, "ALARM_ID": "",
                    "RES_SCAL": 0, "LLM_SCAL": 0, "HLM_SCAL": 0,
                    "LO_LIMIT": it["test_lo"], "HI_LIMIT": it["test_hi"],
                    "UNITS": it["unit"], "C_RESFMT": "%.4f",
                    "C_LLMFMT": "%.4f", "C_HLMFMT": "%.4f",
                    "LO_SPEC": it["spec_lo"], "HI_SPEC": it["spec_hi"],
                }))
            # FTR × 3
            for name in func_items:
                it = meta.items[name]
                passed = bool(func_cols[name][row_i])
                emit(Record(15, 20, {
                    "TEST_NUM": it["test_num"], "HEAD_NUM": 1, "SITE_NUM": 1,
                    "TEST_FLG": 0x01 if not passed else 0x00,
                    "OPT_FLAG": 0x00, "CYCL_CNT": 0, "REL_VADR": 0,
                    "REPT_CNT": 0, "NUM_FAIL": 0, "XFAIL_ADR": 0, "YFAIL_ADR": 0,
                    "VECT_OFF": 0, "RTN_INDX": [], "RTN_STAT": [],
                    "PGM_INDX": [], "PGM_STAT": [], "FAIL_PIN": [],
                    "VECT_NAM": "", "TIME_SET": "", "OP_CODE": "",
                    "TEST_TXT": name, "ALARM_ID": "", "PROG_TXT": "",
                    "RSLT_TXT": "", "PATG_NUM": 0, "SPIN_MAP": [],
                }))
            w = int(results["wafer"].iloc[row_i])
            x = int(results["x"].iloc[row_i])
            y = int(results["y"].iloc[row_i])
            emit(Record(5, 20, {
                "HEAD_NUM": 1, "SITE_NUM": 1,
                "PART_FLG": 0x10 if fail else 0x00,
                "NUM_TEST": len(meta.exec_order),
                "HARD_BIN": hbin, "SOFT_BIN": sbin,
                "X_COORD": x, "Y_COORD": y,
                "TEST_T": test_ms, "PART_ID": f"W{w:02d}-D{die_seq:05d}",
                "PART_TXT": "", "PART_FIX": [],
            }))
        g = grp
        emit(Record(2, 20, {
            "HEAD_NUM": 1, "SITE_GRP": 1, "FINISH_T": t0, "PART_CNT": len(g),
            "RTST_CNT": 0, "ABRT_CNT": 0,
            "GOOD_CNT": int((g["hard_bin"] == 1).sum()),
            "FUNC_CNT": int((g["hard_bin"] == 1).sum()),
            "WAFER_ID": f"W{int(wafer):02d}",
        }))

    # ---- TSR × 15：测试项汇总（执行数/失败数/时间/min/max/平方和）----
    for name in meta.exec_order:
        it = meta.items[name]
        if it["kind"] == "PARM":
            vals = meas_cols[name]
            fail_cnt = int((~results[f"{name}_pass"]).sum())
            vmin, vmax = float(vals.min()), float(vals.max())
            sums = float(vals.sum())
            sqrs = float((vals.astype(np.float64) ** 2).sum())
        else:
            fail_cnt = int((~func_cols[name]).sum())
            vmin = vmax = sums = sqrs = 0.0
        emit(Record(10, 30, {
            "HEAD_NUM": 1, "SITE_NUM": 1,
            "TEST_TYP": "P" if it["kind"] == "PARM" else "F",
            "TEST_NUM": it["test_num"], "EXEC_CNT": len(results),
            "FAIL_CNT": fail_cnt, "ALRM_CNT": 0, "TEST_NAM": name,
            "SEQ_NAME": "", "TEST_LBL": "", "OPT_FLAG": 0x13,  # 保留位 bit3/bit6-7 规范要求为 1
            "TEST_TIM": float(it["meas_time_s"]), "TEST_MIN": vmin,
            "TEST_MAX": vmax, "TST_SUMS": sums, "TST_SQRS": sqrs,
        }))

    emit(Record(1, 20, {"FINISH_T": t0, "DISP_COD": " ",
                        "USR_DESC": "virtual ATE platform W3 STDF writer",
                        "EXC_DESC": ""}))
    path.write_bytes(bytes(out))
    return path
