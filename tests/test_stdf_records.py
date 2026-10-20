"""STDF round-trip：16 种记录类型逐字段比对（writer → reader）。

覆盖：FAR ATR MIR SDR HBR SBR PMR PCR WIR WRR PIR PRR PTR FTR TSR MRR。
"""

import numpy as np
import pytest

from stdf.records import RECORD_NAMES, Record

REC_VERSION = {"CPU_TYPE": 2, "STDF_VER": 4}


def _samples() -> dict[tuple[int, int], dict]:
    """每种记录的代表性字段样例。"""
    return {
        (0, 10): dict(REC_VERSION),
        (0, 20): {"MOD_TIM": 1760000000, "CMD_LINE": "python test"},
        (1, 10): {"SETUP_T": 1760000000, "START_T": 1760000001, "STAT_NUM": 1,
                  "MODE_COD": "A", "RTST_COD": " ", "PROT_COD": " ",
                  "BURN_TIM": 0, "CMOD_COD": " ",
                  "LOT_ID": "LOT20261020A", "PART_TYP": "PMU300",
                  "NODE_NAM": "VIRTUAL_FAB", "TSTR_TYP": "VIRTUAL-ATE",
                  "JOB_NAM": "PMU300_CP", "JOB_REV": "r1",
                  "SBLOT_ID": "S1", "OPER_NAM": "te", "EXEC_TYP": "PY",
                  "EXEC_VER": "1.2", "TEST_COD": "CP", "TEST_TEMP": "25C",
                  "USER_TXT": "w3", "AUX_FILE": "", "PKG_TYP": "QFN24",
                  "PKG_ID": "", "HAND_ID": "", "HAND_TYP": "VIRTUAL_PROBER",
                  "LASR_TYP": "", "LASR_ID": "", "EXTR_TYP": "", "EXTR_ID": ""},
        (1, 20): {"FINISH_T": 1760000100, "DISP_COD": " ", "USR_DESC": "test",
                  "EXC_DESC": ""},
        (1, 30): {"HEAD_NUM": 1, "SITE_GRP": 1, "SITE_NUM": [1, 2],
                  "HAND_TYP": "P", "HAND_ID": "h1", "CARD_TYP": "", "CARD_ID": "",
                  "LOAD_TYP": "", "LOAD_ID": "", "DIB_TYP": "", "DIB_ID": "",
                  "CABL_TYP": "", "CABL_ID": "", "CONT_TYP": "", "CONT_ID": "",
                  "LASR_TYP": "", "LASR_ID": "", "EXTR_TYP": "", "EXTR_ID": ""},
        (1, 40): {"HEAD_NUM": 1, "SITE_NUM": 1, "HBIN_NUM": 2, "HBIN_CNT": 100,
                  "HBIN_PF": "F", "HBIN_NAM": "PARAM_FAIL"},
        (1, 50): {"HEAD_NUM": 1, "SITE_NUM": 1, "SBIN_NUM": 11, "SBIN_CNT": 42,
                  "SBIN_PF": "F", "SBIN_NAM": "VOUT1_FAIL"},
        (1, 60): {"HEAD_NUM": 1, "SITE_NUM": 1, "PART_CNT": 50000, "RTST_CNT": 0,
                  "ABRT_CNT": 0, "GOOD_CNT": 45176, "FUNC_CNT": 45176},
        (1, 62): {"PMR_INDX": 3, "CHANN_TYP": 0, "CHANN_NUM": 3,
                  "PHYS_NAM": "VOUT1", "LOG_NAM": "VOUT1", "HEAD_NUM": 1,
                  "SITE_NUM": 1},
        (2, 10): {"HEAD_NUM": 1, "SITE_GRP": 1, "START_T": 1760000000,
                  "WAFER_ID": "W07"},
        (2, 20): {"HEAD_NUM": 1, "SITE_GRP": 1, "FINISH_T": 1760000060,
                  "PART_CNT": 2000, "RTST_CNT": 0, "ABRT_CNT": 10,
                  "GOOD_CNT": 1800, "FUNC_CNT": 1800, "WAFER_ID": "W07",
                  "FABWF_ID": "", "FRAME_ID": "", "MASK_ID": "",
                  "USR_DESC": "", "EXC_DESC": ""},
        (5, 10): {"HEAD_NUM": 1, "SITE_NUM": 1},
        (5, 20): {"HEAD_NUM": 1, "SITE_NUM": 1, "PART_FLG": 0x80, "NUM_TEST": 15,
                  "HARD_BIN": 2, "SOFT_BIN": 11, "X_COORD": -17, "Y_COORD": 23,
                  "TEST_T": 460, "PART_ID": "W01-D00042", "PART_TXT": "",
                  "PART_FIX": [1, 0, 1, 1]},
        (10, 10): {"TEST_NUM": 1001, "HEAD_NUM": 1, "SITE_NUM": 1,
                   "TEST_FLG": 0x80, "PARM_FLG": 0x00, "RESULT": 3.301691,
                   "TEST_TXT": "VOUT1", "ALARM_ID": "", "RES_SCAL": 0,
                   "LLM_SCAL": 0, "HLM_SCAL": 0, "LO_LIMIT": 3.239172,
                   "HI_LIMIT": 3.360828, "UNITS": "V", "C_RESFMT": "%.4f",
                   "C_LLMFMT": "%.4f", "C_HLMFMT": "%.4f",
                   "LO_SPEC": 3.234, "HI_SPEC": 3.366},
        (10, 30): {"HEAD_NUM": 1, "SITE_NUM": 1, "TEST_TYP": "P",
                   "TEST_NUM": 1001, "EXEC_CNT": 50000, "FAIL_CNT": 700,
                   "ALRM_CNT": 0, "TEST_NAM": "VOUT1", "ALRM_ID": "",
                   "OPT_FLAG": 0x00, "TEST_TIM": 0.02, "TEST_MIN": 3.22,
                   "TEST_MAX": 3.45, "TST_SUMS": 164480.5, "TST_SQRS": 543210.25},
        (15, 20): {"TEST_NUM": 2001, "HEAD_NUM": 1, "SITE_NUM": 1,
                   "TEST_FLG": 0x00, "OPT_FLAG": 0x00, "CYCL_CNT": 128,
                   "REL_VADR": 0, "REPT_CNT": 1, "NUM_FAIL": 0, "XFAIL_ADR": 0,
                   "YFAIL_ADR": 0, "VECT_OFF": 0, "RTN_ICNT": 2,
                   "PGM_ICNT": 2, "RTN_INDX": [3, 5], "RTN_STAT": [1, 0],
                   "PGM_INDX": [7, 9], "PGM_STAT": [2, 1], "FAIL_PIN": [1, 0, 1],
                   "VECT_NAM": "reg_wr", "TIME_SET": "", "OP_CODE": "",
                   "TEST_TXT": "FUNC_REG", "ALARM_ID": "", "PROG_TXT": "",
                   "RSLT_TXT": "", "PATG_NUM": 0, "SPIN_MAP": []},
    }


def _assert_fields_equal(name: str, original: dict, back: dict) -> None:
    for k, v in original.items():
        assert k in back, f"{name}: 读回缺字段 {k}"
        got = back[k]
        if isinstance(v, float):
            assert got is not None and abs(got - v) < max(1e-6, abs(v) * 1e-6), \
                f"{name}.{k}: {got} != {v}"
        elif isinstance(v, list):
            assert list(got) == list(v), f"{name}.{k}: {got} != {v}"
        else:
            assert got == v, f"{name}.{k}: {got!r} != {v!r}"


@pytest.mark.parametrize("key", sorted(RECORD_NAMES.keys()),
                         ids=[RECORD_NAMES[k] for k in sorted(RECORD_NAMES.keys())])
def test_record_roundtrip(key):
    """每种记录类型：编码 → 解码 → 逐字段一致。"""
    fields = _samples()[key]
    rec = Record(key[0], key[1], fields)
    blob = rec.to_bytes()
    # 文件头合法性：U16 长度 + U8/U8 类型子类型
    import struct
    rec_len = struct.unpack(">H", blob[:2])[0]
    assert rec_len == len(blob) - 2
    assert blob[2] == key[0] and blob[3] == key[1]
    back = Record.from_payload(key[0], key[1], blob[4:])
    _assert_fields_equal(RECORD_NAMES[key], fields, back.fields)


def test_far_header_constants():
    """FAR 版本常量：CPU_TYPE=2（大端），STDF_VER=4。"""
    rec = Record(0, 10, dict(REC_VERSION))
    back = Record.from_payload(0, 10, rec.to_bytes()[4:])
    assert back["STDF_VER"] == 4
    assert back["CPU_TYPE"] == 2


def test_ptr_missing_limits_roundtrip():
    """单边规格（无下限）：OPT_FLAG 位 advertise，读回为 None。"""
    f = _samples()[(10, 10)].copy()
    f.update({"LO_LIMIT": None, "LO_SPEC": None, "RESULT": 89.5, "TEST_FLG": 0})
    rec = Record(10, 10, f)
    back = Record.from_payload(10, 10, rec.to_bytes()[4:])
    assert back.fields["LO_LIMIT"] is None
    assert back.fields["LO_SPEC"] is None
    assert back.fields["HI_LIMIT"] == pytest.approx(f["HI_LIMIT"], abs=1e-6)
    assert back.fields["OPT_FLAG"] & 0x40 and back.fields["OPT_FLAG"] & 0x10
