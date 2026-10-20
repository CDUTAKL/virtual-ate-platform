"""STDF V4 记录定义与二进制编解码（设计文档 §3.4，自实现）。

格式（以 STDF V4 规范为准）：
    每条记录 = U16 REC_LEN + U8 REC_TYP + U8 REC_SUB + 字段流
    REC_LEN = REC_TYP+REC_SUB+字段流的字节数（不含 REC_LEN 本身）。
    整数字段大端（big-endian）；C*n = U16 字节数 + ASCII 字节（长度不含 U16）；
    B*n 位数组 = U16 位数 + 紧缩字节；N*1 半字节数组 = U16 个数 + 每字节 2 个。
    可选字段：记录提前结束（剩余字段全部缺省）即缺失，读回为 None。

记录子集（类型/子类型编号照文档 §3.4 表，实现时逐条核对）：
    FAR(0,10) ATR(0,20) MIR(1,10) SDR(1,30) HBR(1,40) SBR(1,50)
    PMR(1,62) PCR(1,60) WIR(2,10) WRR(2,20) PIR(5,10) PRR(5,20)
    PTR(10,10) FTR(15,20) TSR(10,30) MRR(1,20)

位编号：STDF 位字段 bit0 = 字节最高位（0x80）。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# ---- 记录注册表：REC_TYP/REC_SUB → 名称 ----
RECORD_NAMES = {
    (0, 10): "FAR", (0, 20): "ATR",
    (1, 10): "MIR", (1, 20): "MRR", (1, 30): "SDR",
    (1, 40): "HBR", (1, 50): "SBR", (1, 60): "PCR", (1, 62): "PMR",
    (2, 10): "WIR", (2, 20): "WRR",
    (5, 10): "PIR", (5, 20): "PRR",
    (10, 10): "PTR", (10, 30): "TSR",
    (15, 20): "FTR",
}


# =====================================================================
# 字节级原语
# =====================================================================

class Buf:
    """字段流写入缓冲（大端）。"""

    def __init__(self) -> None:
        self.parts: list[bytes] = []

    def u8(self, v: int) -> "Buf":
        self.parts.append(struct.pack(">B", v & 0xFF))
        return self

    def u16(self, v: int) -> "Buf":
        self.parts.append(struct.pack(">H", v & 0xFFFF))
        return self

    def u32(self, v: int) -> "Buf":
        self.parts.append(struct.pack(">I", v & 0xFFFFFFFF))
        return self

    def i8(self, v: int) -> "Buf":
        self.parts.append(struct.pack(">b", v))
        return self

    def i16(self, v: int) -> "Buf":
        self.parts.append(struct.pack(">h", v))
        return self

    def r4(self, v: float) -> "Buf":
        self.parts.append(struct.pack(">f", float(v)))
        return self

    def c1(self, s: str | None) -> "Buf":
        """C*1：单字符；None 按缺省字符写。"""
        s = (s or " ")[:1]
        self.parts.append(s.encode("ascii"))
        return self

    def cn(self, s: str | None) -> "Buf":
        """C*n：U16 长度 + ASCII 字节。"""
        b = (s or "").encode("ascii", errors="replace")
        self.parts.append(struct.pack(">H", len(b)))
        self.parts.append(b)
        return self

    def b1(self, flags: int) -> "Buf":
        """B*1：单字节标志位。"""
        return self.u8(flags)

    def u8_items(self, count: int, vals) -> "Buf":
        """k×U8：个数由外部字段给定，无长度前缀。"""
        vals = list(vals or [])[:count]
        assert len(vals) == count, f"k×U8 数量不符: {len(vals)} != {count}"
        for v in vals:
            self.u8(int(v))
        return self

    def n1_items(self, count: int, vals) -> "Buf":
        """k×N*1：半字节数组，每字节 2 个，个数由外部字段给定。"""
        vals = [int(v) & 0xF for v in (vals or [])][:count]
        assert len(vals) == count
        for i in range(0, count, 2):
            hi = vals[i]
            lo = vals[i + 1] if i + 1 < count else 0
            self.u8((hi << 4) | lo)
        return self

    def bn(self, bits) -> "Buf":
        """B*n 位数组：U16 位数 + 紧缩字节。"""
        bits = [int(b) & 1 for b in (bits or [])]
        self.u16(len(bits))
        for i in range(0, len(bits), 8):
            byte = 0
            for j, b in enumerate(bits[i:i + 8]):
                byte |= b << (7 - j)      # bit0 = 最高位
            self.u8(byte)
        return self

    def bytes(self) -> bytes:
        return b"".join(self.parts)


class Cursor:
    """字段流读取游标；记录提前结束则剩余字段保持 None。"""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    @property
    def remaining(self) -> int:
        return len(self.data) - self.pos

    def _take(self, n: int) -> bytes:
        if self.remaining < n:
            raise ValueError(f"记录字段截断：需要 {n} 字节，剩余 {self.remaining}")
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def u8(self) -> int:
        return struct.unpack(">B", self._take(1))[0]

    def u16(self) -> int:
        return struct.unpack(">H", self._take(2))[0]

    def u32(self) -> int:
        return struct.unpack(">I", self._take(4))[0]

    def i8(self) -> int:
        return struct.unpack(">b", self._take(1))[0]

    def i16(self) -> int:
        return struct.unpack(">h", self._take(2))[0]

    def r4(self) -> float:
        return struct.unpack(">f", self._take(4))[0]

    def c1(self) -> str:
        return self._take(1).decode("ascii", errors="replace")

    def cn(self) -> str:
        n = self.u16()
        return self._take(n).decode("ascii", errors="replace")

    def b1(self) -> int:
        return self.u8()

    def u8_items(self, count: int) -> list[int]:
        return [self.u8() for _ in range(count)]

    def n1_items(self, count: int) -> list[int]:
        vals: list[int] = []
        for _ in range((count + 1) // 2):
            byte = self.u8()
            vals.append((byte >> 4) & 0xF)
            if len(vals) < count:
                vals.append(byte & 0xF)
        return vals

    def bn(self) -> list[int]:
        count = self.u16()
        bits: list[int] = []
        for _ in range((count + 7) // 8):
            byte = self.u8()
            for j in range(8):
                if len(bits) < count:
                    bits.append((byte >> (7 - j)) & 1)
        return bits


# =====================================================================
# 记录对象
# =====================================================================

@dataclass
class Record:
    """一条 STDF 记录：类型/子类型 + 字段字典。"""

    typ: int
    sub: int
    fields: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return RECORD_NAMES.get((self.typ, self.sub), f"REC({self.typ},{self.sub})")

    def to_payload(self) -> bytes:
        """按记录类型编码字段流（不含 REC_LEN/TYP/SUB 头）。"""
        return _ENCODERS[(self.typ, self.sub)](self.fields)

    def to_bytes(self) -> bytes:
        payload = self.to_payload()
        head = bytes([self.typ & 0xFF, self.sub & 0xFF])
        rec_len = len(head) + len(payload)
        return struct.pack(">H", rec_len) + head + payload

    @classmethod
    def from_payload(cls, typ: int, sub: int, payload: bytes) -> "Record":
        cur = Cursor(payload)
        fields = _DECODERS[(typ, sub)](cur)
        return cls(typ, sub, fields)

    def __getitem__(self, key: str):
        return self.fields[key]


# =====================================================================
# 各记录字段编解码（字段顺序以 STDF V4 规范为准）
# =====================================================================

# ---- FAR (0,10) ----
def _enc_far(b: Buf, f: dict) -> None:
    b.u8(f["CPU_TYPE"]).u8(f["STDF_VER"])

def _dec_far(c: Cursor) -> dict:
    return {"CPU_TYPE": c.u8(), "STDF_VER": c.u8()}

# ---- ATR (0,20) ----
def _enc_atr(b: Buf, f: dict) -> None:
    b.u32(f["MOD_TIM"]).cn(f["CMD_LINE"])

def _dec_atr(c: Cursor) -> dict:
    return {"MOD_TIM": c.u32(), "CMD_LINE": c.cn()}

# ---- MIR (1,10)：批次信息 ----
def _enc_mir(b: Buf, f: dict) -> None:
    (b.u32(f["SETUP_T"]).u32(f["START_T"]).u8(f["STAT_NUM"])
     .c1(f.get("MODE_COD")).c1(f.get("RTST_COD")).c1(f.get("PROT_COD"))
     .u16(f.get("BURN_TIM", 0)).c1(f.get("CMOD_COD"))
     .cn(f["LOT_ID"]).cn(f.get("PART_TYP")).cn(f.get("NODE_NAM"))
     .cn(f.get("TSTR_TYP")).cn(f.get("JOB_NAM")).cn(f.get("JOB_REV"))
     .cn(f.get("SBLOT_ID")).cn(f.get("OPER_NAM")).cn(f.get("EXEC_TYP"))
     .cn(f.get("EXEC_VER")).cn(f.get("TEST_COD")).cn(f.get("TEST_TEMP"))
     .cn(f.get("USER_TXT")).cn(f.get("AUX_FILE")).cn(f.get("PKG_TYP"))
     .cn(f.get("PKG_ID")).cn(f.get("HAND_ID")).cn(f.get("HAND_TYP"))
     .cn(f.get("LASR_TYP")).cn(f.get("LASR_ID")).cn(f.get("EXTR_TYP"))
     .cn(f.get("EXTR_ID")))

def _dec_mir(c: Cursor) -> dict:
    f: dict = {"SETUP_T": c.u32(), "START_T": c.u32(), "STAT_NUM": c.u8(),
               "MODE_COD": c.c1(), "RTST_COD": c.c1(), "PROT_COD": c.c1(),
               "BURN_TIM": c.u16(), "CMOD_COD": c.c1(), "LOT_ID": c.cn(),
               "PART_TYP": c.cn(), "NODE_NAM": c.cn(), "TSTR_TYP": c.cn(),
               "JOB_NAM": c.cn(), "JOB_REV": c.cn(), "SBLOT_ID": c.cn(),
               "OPER_NAM": c.cn(), "EXEC_TYP": c.cn(), "EXEC_VER": c.cn(),
               "TEST_COD": c.cn(), "TEST_TEMP": c.cn(), "USER_TXT": c.cn(),
               "AUX_FILE": c.cn(), "PKG_TYP": c.cn(), "PKG_ID": c.cn(),
               "HAND_ID": c.cn(), "HAND_TYP": c.cn(), "LASR_TYP": c.cn(),
               "LASR_ID": c.cn(), "EXTR_TYP": c.cn(), "EXTR_ID": c.cn()}
    return f

# ---- SDR (1,30)：测试头/站点 ----
def _enc_sdr(b: Buf, f: dict) -> None:
    sites = list(f["SITE_NUM"])
    (b.u8(f["HEAD_NUM"]).u8(f["SITE_GRP"]).u8(len(sites)).u8_items(len(sites), sites)
     .cn(f.get("HAND_TYP")).cn(f.get("HAND_ID")).cn(f.get("CARD_TYP"))
     .cn(f.get("CARD_ID")).cn(f.get("LOAD_TYP")).cn(f.get("LOAD_ID"))
     .cn(f.get("DIB_TYP")).cn(f.get("DIB_ID")).cn(f.get("CABL_TYP"))
     .cn(f.get("CABL_ID")).cn(f.get("CONT_TYP")).cn(f.get("CONT_ID"))
     .cn(f.get("LASR_TYP")).cn(f.get("LASR_ID")).cn(f.get("EXTR_TYP"))
     .cn(f.get("EXTR_ID")))

def _dec_sdr(c: Cursor) -> dict:
    head = c.u8(); grp = c.u8(); cnt = c.u8()
    sites = c.u8_items(cnt)
    keys = ["HAND_TYP", "HAND_ID", "CARD_TYP", "CARD_ID", "LOAD_TYP", "LOAD_ID",
            "DIB_TYP", "DIB_ID", "CABL_TYP", "CABL_ID", "CONT_TYP", "CONT_ID",
            "LASR_TYP", "LASR_ID", "EXTR_TYP", "EXTR_ID"]
    f: dict = {"HEAD_NUM": head, "SITE_GRP": grp, "SITE_CNT": cnt, "SITE_NUM": sites}
    f.update({k: c.cn() for k in keys})
    return f

# ---- HBR (1,40) / SBR (1,50) ----
def _enc_hbr(b: Buf, f: dict) -> None:
    (b.u8(f["HEAD_NUM"]).u8(f["SITE_NUM"]).u8(f["HBIN_NUM"])
     .u32(f["HBIN_CNT"]).c1(f.get("HBIN_PF")).cn(f.get("HBIN_NAM")))

def _dec_hbr(c: Cursor) -> dict:
    return {"HEAD_NUM": c.u8(), "SITE_NUM": c.u8(), "HBIN_NUM": c.u8(),
            "HBIN_CNT": c.u32(), "HBIN_PF": c.c1(), "HBIN_NAM": c.cn()}

def _enc_sbr(b: Buf, f: dict) -> None:
    (b.u8(f["HEAD_NUM"]).u8(f["SITE_NUM"]).u8(f["SBIN_NUM"])
     .u32(f["SBIN_CNT"]).c1(f.get("SBIN_PF")).cn(f.get("SBIN_NAM")))

def _dec_sbr(c: Cursor) -> dict:
    return {"HEAD_NUM": c.u8(), "SITE_NUM": c.u8(), "SBIN_NUM": c.u8(),
            "SBIN_CNT": c.u32(), "SBIN_PF": c.c1(), "SBIN_NAM": c.cn()}

# ---- PCR (1,60)：各 site 计数 ----
def _enc_pcr(b: Buf, f: dict) -> None:
    (b.u8(f["HEAD_NUM"]).u8(f["SITE_NUM"]).u32(f["PART_CNT"])
     .u32(f.get("RTST_CNT", 0)).u32(f.get("ABRT_CNT", 0))
     .u32(f.get("GOOD_CNT", 0)).u32(f.get("FUNC_CNT", 0)))

def _dec_pcr(c: Cursor) -> dict:
    return {"HEAD_NUM": c.u8(), "SITE_NUM": c.u8(), "PART_CNT": c.u32(),
            "RTST_CNT": c.u32(), "ABRT_CNT": c.u32(),
            "GOOD_CNT": c.u32(), "FUNC_CNT": c.u32()}

# ---- PMR (1,62)：引脚映射 ----
def _enc_pmr(b: Buf, f: dict) -> None:
    (b.u16(f["PMR_INDX"]).u16(f.get("CHANN_TYP", 0))
     .u16(f.get("CHANN_NUM", 0)).cn(f.get("PHYS_NAM")).cn(f.get("LOG_NAM"))
     .u8(f.get("HEAD_NUM", 1)).u8(f.get("SITE_NUM", 1)))

def _dec_pmr(c: Cursor) -> dict:
    return {"PMR_INDX": c.u16(), "CHANN_TYP": c.u16(), "CHANN_NUM": c.u16(),
            "PHYS_NAM": c.cn(), "LOG_NAM": c.cn(),
            "HEAD_NUM": c.u8(), "SITE_NUM": c.u8()}

# ---- WIR (2,10) / WRR (2,20) ----
def _enc_wir(b: Buf, f: dict) -> None:
    b.u8(f["HEAD_NUM"]).u8(f.get("SITE_GRP", 1)).u32(f["START_T"]).cn(f.get("WAFER_ID"))

def _dec_wir(c: Cursor) -> dict:
    return {"HEAD_NUM": c.u8(), "SITE_GRP": c.u8(), "START_T": c.u32(),
            "WAFER_ID": c.cn()}

def _enc_wrr(b: Buf, f: dict) -> None:
    (b.u8(f["HEAD_NUM"]).u8(f.get("SITE_GRP", 1)).u32(f["FINISH_T"])
     .u32(f["PART_CNT"]).u32(f.get("RTST_CNT", 0)).u32(f.get("ABRT_CNT", 0))
     .u32(f.get("GOOD_CNT", 0)).u32(f.get("FUNC_CNT", 0))
     .cn(f.get("WAFER_ID")).cn(f.get("FABWF_ID")).cn(f.get("FRAME_ID"))
     .cn(f.get("MASK_ID")).cn(f.get("USR_DESC")).cn(f.get("EXC_DESC")))

def _dec_wrr(c: Cursor) -> dict:
    return {"HEAD_NUM": c.u8(), "SITE_GRP": c.u8(), "FINISH_T": c.u32(),
            "PART_CNT": c.u32(), "RTST_CNT": c.u32(), "ABRT_CNT": c.u32(),
            "GOOD_CNT": c.u32(), "FUNC_CNT": c.u32(), "WAFER_ID": c.cn(),
            "FABWF_ID": c.cn(), "FRAME_ID": c.cn(), "MASK_ID": c.cn(),
            "USR_DESC": c.cn(), "EXC_DESC": c.cn()}

# ---- PIR (5,10) / PRR (5,20) ----
def _enc_pir(b: Buf, f: dict) -> None:
    b.u8(f["HEAD_NUM"]).u8(f["SITE_NUM"])

def _dec_pir(c: Cursor) -> dict:
    return {"HEAD_NUM": c.u8(), "SITE_NUM": c.u8()}

# PART_FLG（以规范为准）：bit0(0x80)=fail bit1(0x40)=无 pass/fail
# bit2(0x20)=abort bit3(0x10)=无 bin 信息 bit4(0x08)=reject
def _enc_prr(b: Buf, f: dict) -> None:
    (b.u8(f["HEAD_NUM"]).u8(f["SITE_NUM"]).b1(f.get("PART_FLG", 0))
     .u16(f["NUM_TEST"]).u16(f["HARD_BIN"]).u16(f["SOFT_BIN"])
     .i16(f.get("X_COORD", -32768)).i16(f.get("Y_COORD", -32768))
     .u32(f.get("TEST_T", 0)).cn(f.get("PART_ID")).cn(f.get("PART_TXT"))
     .bn(f.get("PART_FIX")))

def _dec_prr(c: Cursor) -> dict:
    f: dict = {"HEAD_NUM": c.u8(), "SITE_NUM": c.u8(), "PART_FLG": c.b1(),
               "NUM_TEST": c.u16(), "HARD_BIN": c.u16(), "SOFT_BIN": c.u16(),
               "X_COORD": c.i16(), "Y_COORD": c.i16(), "TEST_T": c.u32(),
               "PART_ID": c.cn(), "PART_TXT": c.cn(), "PART_FIX": c.bn()}
    return f

# ---- PTR (10,10)：参数测试结果 ----
# TEST_FLG：bit0(0x80)=fail bit1(0x40)=无结果 bit2(0x20)=不可靠 bit3(0x10)=超时
#           bit4(0x08)=未执行 bit5(0x04)=abort
# OPT_FLAG：bit0(0x80)=无结果 bit1(0x40)=无下限 bit2(0x20)=无上限
#           bit3(0x10)=无下规格 bit4(0x08)=无上规格
def _enc_ptr(b: Buf, f: dict) -> None:
    has_res = f.get("RESULT") is not None
    has_lo = f.get("LO_LIMIT") is not None
    has_hi = f.get("HI_LIMIT") is not None
    has_lspec = f.get("LO_SPEC") is not None
    has_hspec = f.get("HI_SPEC") is not None
    opt = ((0x80 if not has_res else 0) | (0x40 if not has_lo else 0)
           | (0x20 if not has_hi else 0) | (0x10 if not has_lspec else 0)
           | (0x08 if not has_hspec else 0))
    # 限值/规格字段固定全写（缺省侧写 0.0），OPT_FLAG 位 advertise 有效性——
    # 保持字段定长，任何规范一致的分析器都能正确读取（缺失仅截断记录尾）。
    (b.u32(f["TEST_NUM"]).u8(f["HEAD_NUM"]).u8(f["SITE_NUM"])
     .b1(f.get("TEST_FLG", 0)).b1(f.get("PARM_FLG", 0))
     .r4(f["RESULT"] if has_res else 0.0)
     .cn(f.get("TEST_TXT")).cn(f.get("ALARM_ID"))
     .b1(opt)
     .i8(f.get("RES_SCAL", 0)).i8(f.get("LLM_SCAL", 0)).i8(f.get("HLM_SCAL", 0))
     .r4(f.get("LO_LIMIT") or 0.0).r4(f.get("HI_LIMIT") or 0.0)
     .cn(f.get("UNITS")).cn(f.get("C_RESFMT")).cn(f.get("C_LLMFMT")).cn(f.get("C_HLMFMT"))
     .r4(f.get("LO_SPEC") or 0.0).r4(f.get("HI_SPEC") or 0.0))

def _dec_ptr(c: Cursor) -> dict:
    # 字段定长全读（writer 固定写全部限值/规格字段，缺省侧 0.0），
    # 有效性由 OPT_FLAG 位 advertise。
    f: dict = {"TEST_NUM": c.u32(), "HEAD_NUM": c.u8(), "SITE_NUM": c.u8(),
               "TEST_FLG": c.b1(), "PARM_FLG": c.b1(), "RESULT": c.r4(),
               "TEST_TXT": c.cn(), "ALARM_ID": c.cn(), "OPT_FLAG": c.b1(),
               "RES_SCAL": c.i8(), "LLM_SCAL": c.i8(), "HLM_SCAL": c.i8(),
               "LO_LIMIT": c.r4(), "HI_LIMIT": c.r4(),
               "UNITS": c.cn(), "C_RESFMT": c.cn(), "C_LLMFMT": c.cn(),
               "C_HLMFMT": c.cn(), "LO_SPEC": c.r4(), "HI_SPEC": c.r4()}
    opt = f["OPT_FLAG"]
    if opt & 0x80:
        f["RESULT"] = None
    if opt & 0x40:
        f["LO_LIMIT"] = None
    if opt & 0x20:
        f["HI_LIMIT"] = None
    if opt & 0x10:
        f["LO_SPEC"] = None
    if opt & 0x08:
        f["HI_SPEC"] = None
    return f

# ---- FTR (15,20)：功能测试结果 ----
def _enc_ftr(b: Buf, f: dict) -> None:
    rtn = list(f.get("RTN_STAT") or [])
    pgm = list(f.get("PGM_STAT") or [])
    rtn_i = list(f.get("RTN_INDX") or [])
    pgm_i = list(f.get("PGM_INDX") or [])
    assert len(rtn) == len(rtn_i) and len(pgm) == len(pgm_i)
    (b.u32(f["TEST_NUM"]).u8(f["HEAD_NUM"]).u8(f["SITE_NUM"])
     .b1(f.get("TEST_FLG", 0)).b1(f.get("OPT_FLAG", 0))
     .u32(f.get("CYCL_CNT", 0)).u32(f.get("REL_VADR", 0))
     .u32(f.get("REPT_CNT", 0)).u32(f.get("NUM_FAIL", 0))
     .u32(f.get("XFAIL_ADR", 0)).u32(f.get("YFAIL_ADR", 0))
     .u16(f.get("VECT_OFF", 0)).u16(len(rtn)).u16(len(pgm))
     .u8_items(len(rtn_i), rtn_i).n1_items(len(rtn), rtn)
     .u8_items(len(pgm_i), pgm_i).n1_items(len(pgm), pgm)
     .bn(f.get("FAIL_PIN"))
     .cn(f.get("VECT_NAM")).cn(f.get("TIME_SET")).cn(f.get("OP_CODE"))
     .cn(f.get("TEST_TXT")).cn(f.get("ALARM_ID")).cn(f.get("PROG_TXT"))
     .cn(f.get("RSLT_TXT")).u8(f.get("PATG_NUM", 0)).bn(f.get("SPIN_MAP")))

def _dec_ftr(c: Cursor) -> dict:
    f: dict = {"TEST_NUM": c.u32(), "HEAD_NUM": c.u8(), "SITE_NUM": c.u8(),
               "TEST_FLG": c.b1(), "OPT_FLAG": c.b1(), "CYCL_CNT": c.u32(),
               "REL_VADR": c.u32(), "REPT_CNT": c.u32(), "NUM_FAIL": c.u32(),
               "XFAIL_ADR": c.u32(), "YFAIL_ADR": c.u32(), "VECT_OFF": c.u16(),
               "RTN_ICNT": c.u16(), "PGM_ICNT": c.u16()}
    f["RTN_INDX"] = c.u8_items(f["RTN_ICNT"])
    f["RTN_STAT"] = c.n1_items(f["RTN_ICNT"])
    f["PGM_INDX"] = c.u8_items(f["PGM_ICNT"])
    f["PGM_STAT"] = c.n1_items(f["PGM_ICNT"])
    f["FAIL_PIN"] = c.bn()
    for k in ("VECT_NAM", "TIME_SET", "OP_CODE", "TEST_TXT", "ALARM_ID",
              "PROG_TXT", "RSLT_TXT"):
        f[k] = c.cn()
    f["PATG_NUM"] = c.u8()
    f["SPIN_MAP"] = c.bn()
    return f

# ---- TSR (10,30)：测试项汇总 ----
def _enc_tsr(b: Buf, f: dict) -> None:
    (b.u8(f["HEAD_NUM"]).u8(f["SITE_NUM"]).c1(f.get("TEST_TYP", "P"))
     .u32(f["TEST_NUM"]).u32(f["EXEC_CNT"]).u32(f["FAIL_CNT"])
     .u32(f.get("ALRM_CNT", 0)).cn(f.get("TEST_NAM")).cn(f.get("ALRM_ID"))
     .b1(f.get("OPT_FLAG", 0))
     .r4(f.get("TEST_TIM", 0.0)).r4(f.get("TEST_MIN", 0.0))
     .r4(f.get("TEST_MAX", 0.0)).r4(f.get("TST_SUMS", 0.0))
     .r4(f.get("TST_SQRS", 0.0)))

def _dec_tsr(c: Cursor) -> dict:
    f: dict = {"HEAD_NUM": c.u8(), "SITE_NUM": c.u8(), "TEST_TYP": c.c1(),
               "TEST_NUM": c.u32(), "EXEC_CNT": c.u32(), "FAIL_CNT": c.u32(),
               "ALRM_CNT": c.u32(), "TEST_NAM": c.cn(), "ALRM_ID": c.cn(),
               "OPT_FLAG": c.b1(), "TEST_TIM": c.r4(), "TEST_MIN": c.r4(),
               "TEST_MAX": c.r4(), "TST_SUMS": c.r4(), "TST_SQRS": c.r4()}
    return f

# ---- MRR (1,20) ----
def _enc_mrr(b: Buf, f: dict) -> None:
    b.u32(f["FINISH_T"]).c1(f.get("DISP_COD")).cn(f.get("USR_DESC")).cn(f.get("EXC_DESC"))


def _dec_mrr(c: Cursor) -> dict:
    return {"FINISH_T": c.u32(), "DISP_COD": c.c1(),
            "USR_DESC": c.cn(), "EXC_DESC": c.cn()}


def _pack(enc_fn, f: dict) -> bytes:
    b = Buf()
    enc_fn(b, f)
    return b.bytes()


_ENCODERS = {
    (0, 10): lambda f: _pack(_enc_far, f),
    (0, 20): lambda f: _pack(_enc_atr, f),
    (1, 10): lambda f: _pack(_enc_mir, f),
    (1, 20): lambda f: _pack(_enc_mrr, f),
    (1, 30): lambda f: _pack(_enc_sdr, f),
    (1, 40): lambda f: _pack(_enc_hbr, f),
    (1, 50): lambda f: _pack(_enc_sbr, f),
    (1, 60): lambda f: _pack(_enc_pcr, f),
    (1, 62): lambda f: _pack(_enc_pmr, f),
    (2, 10): lambda f: _pack(_enc_wir, f),
    (2, 20): lambda f: _pack(_enc_wrr, f),
    (5, 10): lambda f: _pack(_enc_pir, f),
    (5, 20): lambda f: _pack(_enc_prr, f),
    (10, 10): lambda f: _pack(_enc_ptr, f),
    (10, 30): lambda f: _pack(_enc_tsr, f),
    (15, 20): lambda f: _pack(_enc_ftr, f),
}

_DECODERS = {
    (0, 10): _dec_far, (0, 20): _dec_atr,
    (1, 10): _dec_mir, (1, 20): _dec_mrr, (1, 30): _dec_sdr,
    (1, 40): _dec_hbr, (1, 50): _dec_sbr, (1, 60): _dec_pcr, (1, 62): _dec_pmr,
    (2, 10): _dec_wir, (2, 20): _dec_wrr,
    (5, 10): _dec_pir, (5, 20): _dec_prr,
    (10, 10): _dec_ptr, (10, 30): _dec_tsr,
    (15, 20): _dec_ftr,
}
