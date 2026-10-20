"""STDF V4 reader：二进制解析回结构化记录对象（设计文档 §3.4）。

自检机制（文档要求）：writer 写完用 reader 读回逐字段比对（round-trip，
pytest 覆盖每种记录类型）。
"""

from __future__ import annotations

import struct
from pathlib import Path

import pandas as pd

from .records import RECORD_NAMES, Record


class StdfFile:
    """一份解析后的 STDF 文件：记录列表 + 按类型检索。"""

    def __init__(self, records: list[Record]) -> None:
        self.records = records

    def by_type(self, typ: int, sub: int | None = None) -> list[Record]:
        return [r for r in self.records
                if r.typ == typ and (sub is None or r.sub == sub)]

    @property
    def names(self) -> list[str]:
        return [r.name for r in self.records]

    def count(self, name: str) -> int:
        return sum(1 for r in self.records if r.name == name)

    def first(self, name: str) -> Record:
        for r in self.records:
            if r.name == name:
                return r
        raise KeyError(name)

    def to_results_frame(self) -> pd.DataFrame:
        """把 STDF 记录流重建为 die 级结果表（分析层唯一入口）。

        按记录顺序走：WIR 记当前 wafer；PIR 开新 die；PTR/FTR 按 EXECUTE
        顺序归位；PRR 收尾（坐标/bin/测试时间）。分析层只消费本方法的输出，
        不接触 DUT 模型与测试程序内部对象（文档 §2.1 架构边界）。
        """
        rows: list[dict] = []
        cur: dict | None = None
        wafer = 0
        ptr_seq: list[str] = []
        for r in self.records:
            if r.name == "WIR":
                wid = r.fields.get("WAFER_ID", "")
                wafer = int("".join(ch for ch in wid if ch.isdigit()) or 0)
            elif r.name == "PIR":
                cur = {"wafer": wafer}
            elif r.name == "PTR" and cur is not None:
                cur[f"P::{r.fields['TEST_TXT']}"] = r.fields["RESULT"]
                if r.fields["TEST_TXT"] not in ptr_seq:
                    ptr_seq.append(r.fields["TEST_TXT"])
            elif r.name == "FTR" and cur is not None:
                cur[f"F::{r.fields['TEST_TXT']}"] = not (r.fields["TEST_FLG"] & 0x80)
            elif r.name == "PRR" and cur is not None:
                f = r.fields
                cur.update({"x": f["X_COORD"], "y": f["Y_COORD"],
                            "soft_bin": f["SOFT_BIN"], "hard_bin": f["HARD_BIN"],
                            "test_time_s": f["TEST_T"] / 1000.0,
                            "part_id": f["PART_ID"]})
                rows.append(cur)
                cur = None
        df = pd.DataFrame(rows)
        df = df.rename(columns={f"P::{c}": c for c in ptr_seq})
        df = df.rename(columns={c: c.replace("F::", "") for c in df.columns})
        # soft bin 名称从 SBR 记录还原（STDF 自带 bin 定义表）
        sbr = {r.fields["SBIN_NUM"]: r.fields["SBIN_NAM"]
               for r in self.by_type(1, 50)}
        if len(df) and "soft_bin" in df.columns:
            df["soft_bin_name"] = df["soft_bin"].map(lambda s: sbr.get(int(s), f"S{s}"))
        return df

    def ptr_table(self) -> pd.DataFrame:
        """全部 PTR 记录 → DataFrame（逐 die 逐参数一行）。"""
        rows = []
        for r in self.by_type(10, 10):
            f = r.fields
            rows.append({
                "test_num": f["TEST_NUM"], "test_txt": f["TEST_TXT"],
                "units": f["UNITS"], "result": f["RESULT"],
                "lo_limit": f["LO_LIMIT"], "hi_limit": f["HI_LIMIT"],
                "lo_spec": f["LO_SPEC"], "hi_spec": f["HI_SPEC"],
                "test_flg": f["TEST_FLG"],
            })
        return pd.DataFrame(rows)

    def limits_table(self) -> pd.DataFrame:
        """测试项测试限/规格限表：每种 TEST_NUM 取首条 PTR（每 die 重复）。"""
        ptr = self.ptr_table()
        cols = ["test_num", "test_txt", "units", "lo_limit", "hi_limit",
                "lo_spec", "hi_spec"]
        return ptr[cols].drop_duplicates(subset="test_num").reset_index(drop=True)

    def prr_table(self) -> pd.DataFrame:
        """全部 PRR 记录 → DataFrame（坐标/bin/测试时间）。"""
        rows = []
        for r in self.by_type(5, 20):
            f = r.fields
            rows.append({
                "part_id": f["PART_ID"], "head_num": f["HEAD_NUM"],
                "site_num": f["SITE_NUM"], "part_flg": f["PART_FLG"],
                "num_test": f["NUM_TEST"], "hard_bin": f["HARD_BIN"],
                "soft_bin": f["SOFT_BIN"], "x": f["X_COORD"], "y": f["Y_COORD"],
                "test_t": f["TEST_T"],
            })
        return pd.DataFrame(rows)


def read_stdf(path: str | Path) -> StdfFile:
    """解析 STDF 文件 → StdfFile。只认本 records.py 定义的记录子集。"""
    data = Path(path).read_bytes()
    pos = 0
    records: list[Record] = []
    n = len(data)
    while pos < n:
        if pos + 4 > n:
            raise ValueError(f"文件尾部残留 {n - pos} 字节，不是完整记录头")
        rec_len = struct.unpack(">H", data[pos:pos + 2])[0]
        typ = data[pos + 2]
        sub = data[pos + 3]
        payload = data[pos + 4: pos + 2 + rec_len]
        if len(payload) != rec_len - 2:
            raise ValueError(f"记录({typ},{sub}) 截断：声明 {rec_len - 2} 字节，"
                             f"实际 {len(payload)}")
        if (typ, sub) not in RECORD_NAMES:
            raise ValueError(f"未支持的记录类型 ({typ},{sub})")
        records.append(Record.from_payload(typ, sub, payload))
        pos += 2 + rec_len
    return StdfFile(records)
