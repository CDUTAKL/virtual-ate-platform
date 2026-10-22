"""STDF round-trip：16 种记录类型逐字段比对（writer → reader）。

覆盖：FAR ATR MIR SDR HBR SBR PMR PCR WIR WRR PIR PRR PTR FTR TSR MRR。
"""

import struct

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
                  "EXEC_VER": "1.2", "TEST_COD": "CP", "TST_TEMP": "25C",
                  "USER_TXT": "w3", "AUX_FILE": "", "PKG_TYP": "QFN24",
                  "FAMLY_ID": "", "DATE_COD": "", "FACIL_ID": "",
                  "FLOOR_ID": "", "PROC_ID": "", "OPER_FRQ": "",
                  "SPEC_NAM": "", "SPEC_VER": "", "FLOW_ID": "CP_FLOW",
                  "SETUP_ID": "", "DSGN_REV": "", "ENG_ID": "",
                  "ROM_COD": "", "SERL_NUM": "", "SUPR_NAM": "lead-te"},
        (1, 20): {"FINISH_T": 1760000100, "DISP_COD": " ", "USR_DESC": "test",
                  "EXC_DESC": ""},
        (1, 80): {"HEAD_NUM": 1, "SITE_GRP": 1, "SITE_NUM": [1, 2],
                  "HAND_TYP": "P", "HAND_ID": "h1", "CARD_TYP": "", "CARD_ID": "",
                  "LOAD_TYP": "", "LOAD_ID": "", "DIB_TYP": "", "DIB_ID": "",
                  "CABL_TYP": "", "CABL_ID": "", "CONT_TYP": "", "CONT_ID": "",
                  "LASR_TYP": "", "LASR_ID": "", "EXTR_TYP": "", "EXTR_ID": ""},
        (1, 40): {"HEAD_NUM": 1, "SITE_NUM": 1, "HBIN_NUM": 2, "HBIN_CNT": 100,
                  "HBIN_PF": "F", "HBIN_NAM": "PARAM_FAIL"},
        (1, 50): {"HEAD_NUM": 1, "SITE_NUM": 1, "SBIN_NUM": 11, "SBIN_CNT": 42,
                  "SBIN_PF": "F", "SBIN_NAM": "VOUT1_FAIL"},
        (1, 30): {"HEAD_NUM": 1, "SITE_NUM": 1, "PART_CNT": 50000, "RTST_CNT": 0,
                  "ABRT_CNT": 0, "GOOD_CNT": 45176, "FUNC_CNT": 45176},
        (1, 60): {"PMR_INDX": 3, "CHAN_TYP": 0, "CHAN_NAM": "",
                  "PHY_NAM": "VOUT1", "LOG_NAM": "VOUT1", "HEAD_NUM": 1,
                  "SITE_NUM": 1},
        (2, 10): {"HEAD_NUM": 1, "SITE_GRP": 1, "START_T": 1760000000,
                  "WAFER_ID": "W07"},
        (2, 20): {"HEAD_NUM": 1, "SITE_GRP": 1, "FINISH_T": 1760000060,
                  "PART_CNT": 2000, "RTST_CNT": 0, "ABRT_CNT": 10,
                  "GOOD_CNT": 1800, "FUNC_CNT": 1800, "WAFER_ID": "W07",
                  "FABWF_ID": "", "FRAME_ID": "", "MASK_ID": "",
                  "USR_DESC": "", "EXC_DESC": ""},
        (5, 10): {"HEAD_NUM": 1, "SITE_NUM": 1},
        (5, 20): {"HEAD_NUM": 1, "SITE_NUM": 1, "PART_FLG": 0x10, "NUM_TEST": 15,
                  "HARD_BIN": 2, "SOFT_BIN": 11, "X_COORD": -17, "Y_COORD": 23,
                  "TEST_T": 460, "PART_ID": "W01-D00042", "PART_TXT": "",
                  "PART_FIX": [1, 0, 1, 1, 0, 0, 1, 1]},
        (15, 10): {"TEST_NUM": 1001, "HEAD_NUM": 1, "SITE_NUM": 1,
                   "TEST_FLG": 0x01, "PARM_FLG": 0x00, "RESULT": 3.301691,
                   "TEST_TXT": "VOUT1", "ALARM_ID": "", "RES_SCAL": 0,
                   "LLM_SCAL": 0, "HLM_SCAL": 0, "LO_LIMIT": 3.239172,
                   "HI_LIMIT": 3.360828, "UNITS": "V", "C_RESFMT": "%.4f",
                   "C_LLMFMT": "%.4f", "C_HLMFMT": "%.4f",
                   "LO_SPEC": 3.234, "HI_SPEC": 3.366},
        (10, 30): {"HEAD_NUM": 1, "SITE_NUM": 1, "TEST_TYP": "P",
                   "TEST_NUM": 1001, "EXEC_CNT": 50000, "FAIL_CNT": 700,
                   "ALRM_CNT": 0, "TEST_NAM": "VOUT1", "SEQ_NAME": "cp_flow",
                   "TEST_LBL": "", "OPT_FLAG": 0x13, "TEST_TIM": 0.02,
                   "TEST_MIN": 3.22, "TEST_MAX": 3.45, "TST_SUMS": 164480.5,
                   "TST_SQRS": 543210.25},
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
    # 文件头合法性（SEMI STDF V4）：REC_LEN 只计数据体（不含 4 字节头）
    rec_len = struct.unpack(">H", blob[:2])[0]
    assert rec_len == len(blob) - 4
    assert blob[2] == key[0] and blob[3] == key[1]
    back = Record.from_payload(key[0], key[1], blob[4:])
    _assert_fields_equal(RECORD_NAMES[key], fields, back.fields)


def test_far_golden_bytes():
    """FAR 逐字节对照真实标准样本：00 02 00 0A 02 04。

    REC_LEN=0x0002（仅数据体 CPU_TYPE+STDF_VER），REC_TYP=0，REC_SUB=10，
    数据体 = 02（CPU_TYPE 大端/SUN）04（STDF_VER V4）。共 6 字节。
    """
    rec = Record(0, 10, dict(REC_VERSION))
    assert rec.to_bytes() == bytes.fromhex("00 02 00 0A 02 04")


def test_cn_uses_u8_length_prefix():
    """C*n 字符串为 U8 长度前缀（标准），非 U16。

    WIR(2,10) W07：REC_LEN=10，数据体 = HEAD_NUM + SITE_GRP + START_T(U32)
    + C*n("W07")（1 字节长度 + 3 字节 ASCII）。
    """
    rec = Record(2, 10, {"HEAD_NUM": 1, "SITE_GRP": 1, "START_T": 1760000000,
                         "WAFER_ID": "W07"})
    blob = rec.to_bytes()
    assert blob[:2] == struct.pack(">H", 10)
    assert blob[2:4] == bytes([2, 10])
    assert blob[-4] == 3                      # C*n U8 长度 = 3
    assert blob[-3:] == b"W07"


def test_far_header_constants():
    """FAR 版本常量：CPU_TYPE=2（大端），STDF_VER=4。"""
    rec = Record(0, 10, dict(REC_VERSION))
    back = Record.from_payload(0, 10, rec.to_bytes()[4:])
    assert back["STDF_VER"] == 4
    assert back["CPU_TYPE"] == 2


def test_read_handcrafted_standard_hex(tmp_path):
    """reader 读手工构造的标准 hex 样本（按 SEMI STDF V4 规范逐字节手写）。

    FAR(0,10) + PMR(1,60) + PTR(15,10) + PRR(5,20) + TSR(10,30)，
    与 stdf.records 的编码逻辑完全独立（含 B*n LSB-first、TSR SEQ_NAME/TEST_LBL）。
    """
    from stdf.reader import read_stdf
    std = bytes.fromhex(
        # FAR: REC_LEN=2, TYP=0, SUB=10, CPU_TYPE=2, STDF_VER=4
        "00 02 00 0A 02 04"
        # PMR(1,60): REC_LEN=19，PMR_INDX=1 CHAN_TYP=0 CHAN_NAM=""
        # PHY_NAM=LOG_NAM="VOUT1" HEAD=1 SITE=1
        "00 13 01 3C 00 01 00 00 00 05 56 4F 55 54 31 05 56 4F 55 54 31 01 01"
        # PTR(15,10): REC_LEN=56，数据体见下方字段
        "00 38 0F 0A"
        "00 00 03 E9"                          # TEST_NUM = 1001
        "01 01 00 00"                          # HEAD/SITE/TEST_FLG/PARM_FLG
        "40 60 00 00"                          # RESULT = 3.5 (R4)
        "05 56 4F 55 54 31"                    # TEST_TXT C*n = "VOUT1"
        "00"                                   # ALARM_ID C*n = ""
        "40 00 00 00"                          # OPT_FLAG(保留位bit1=1) + RES/LLM/HLM_SCAL
        "40 4C CC CD 40 66 66 66"              # LO_LIMIT=3.2, HI_LIMIT=3.6
        "01 56"                                # UNITS = "V"
        "04 25 2E 34 66"                       # C_RESFMT = "%.4f"
        "04 25 2E 34 66"                       # C_LLMFMT = "%.4f"
        "04 25 2E 34 66"                       # C_HLMFMT = "%.4f"
        "40 46 66 66 40 6C CC CD"              # LO_SPEC=3.1, HI_SPEC=3.7
        # PRR(5,20): REC_LEN=31，PART_FLG=0x10(fail) NUM_TEST=15 HARD_BIN=2
        # SOFT_BIN=11 X=-17 Y=23 TEST_T=460 PART_ID="W01-D00042"
        # PART_FIX=B*n：1 字节 LSB-first 0xCD = [1,0,1,1,0,0,1,1]
        "00 1F 05 14 01 01 10 00 0F 00 02 00 0B FF EF 00 17 00 00 01 CC"
        "0A 57 30 31 2D 44 30 30 30 34 32 00 01 CD"
        # TSR(10,30): REC_LEN=48，TEST_NAM="VOUT1" SEQ_NAME="" TEST_LBL=""
        # OPT_FLAG=0x13(保留位) TEST_TIM=0.02 MIN=3.22 MAX=3.45 SUMS=13000 SQRS=42100
        "00 30 0A 1E 01 01 50 00 00 03 E9 00 00 0F A0 00 00 01 47 00 00 00 00"
        "05 56 4F 55 54 31 00 00 13 3C A3 D7 0A 40 4E 14 7B 40 5C CC CD"
        "46 4B 20 00 47 24 74 00")
    path = tmp_path / "handcrafted.stdf"
    path.write_bytes(std)
    sf = read_stdf(path)
    assert sf.names == ["FAR", "PMR", "PTR", "PRR", "TSR"]
    assert sf.first("FAR")["CPU_TYPE"] == 2
    pmr = sf.first("PMR")
    assert pmr.fields["PMR_INDX"] == 1
    assert pmr.fields["PHY_NAM"] == "VOUT1" and pmr.fields["LOG_NAM"] == "VOUT1"
    assert pmr.fields["CHAN_NAM"] == ""
    ptr = sf.first("PTR")
    assert ptr.fields["TEST_NUM"] == 1001
    assert ptr.fields["RESULT"] == pytest.approx(3.5, abs=1e-6)
    assert ptr.fields["TEST_TXT"] == "VOUT1"
    assert ptr.fields["UNITS"] == "V"
    assert ptr.fields["LO_LIMIT"] == pytest.approx(3.2, abs=1e-6)
    assert ptr.fields["HI_LIMIT"] == pytest.approx(3.6, abs=1e-6)
    prr = sf.first("PRR")
    assert prr.fields["PART_FLG"] == 0x10
    assert prr.fields["HARD_BIN"] == 2 and prr.fields["SOFT_BIN"] == 11
    assert prr.fields["X_COORD"] == -17 and prr.fields["Y_COORD"] == 23
    assert prr.fields["PART_ID"] == "W01-D00042"
    assert prr.fields["PART_FIX"] == [1, 0, 1, 1, 0, 0, 1, 1]
    tsr = sf.first("TSR")
    assert tsr.fields["TEST_NAM"] == "VOUT1"
    assert tsr.fields["SEQ_NAME"] == "" and tsr.fields["TEST_LBL"] == ""
    assert tsr.fields["EXEC_CNT"] == 4000 and tsr.fields["FAIL_CNT"] == 327


def test_ptr_missing_limits_roundtrip():
    """单边规格（无下限）：OPT_FLAG 规范位 advertise，读回为 None。

    规范 OPT_FLAG：bit6(0x02)=无下限 bit4(0x08)=下限无效 bit2(0x20)=无下规格。
    """
    f = _samples()[(15, 10)].copy()
    f.update({"LO_LIMIT": None, "LO_SPEC": None, "RESULT": 89.5, "TEST_FLG": 0})
    rec = Record(15, 10, f)
    back = Record.from_payload(15, 10, rec.to_bytes()[4:])
    assert back.fields["LO_LIMIT"] is None
    assert back.fields["LO_SPEC"] is None
    assert back.fields["HI_LIMIT"] == pytest.approx(f["HI_LIMIT"], abs=1e-6)
    assert back.fields["OPT_FLAG"] & 0x02 and back.fields["OPT_FLAG"] & 0x20
