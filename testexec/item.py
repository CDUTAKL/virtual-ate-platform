"""测试项抽象（设计文档 §3.3 TestItem）。

对标机台测试程序（STS8200 / V93000 概念结构）中的测试方法节点：
字段 = test_num、名称、测量方法（调用哪个仪器）、spec 上下限、guardband 后
的测试限、单位、引脚关联；判定即 run 的语义（测量值 vs 测试限）。

本类只承载数据与判定逻辑；测量执行在 testexec/cp.py（批量路径）或
instruments/driver.py（SCPI 单 die 路径）。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TestItem:
    """单个测试项（参数项或功能项）。"""

    __test__ = False  # 防止 pytest 误收集（类名以 Test 开头）

    test_num: int                 # 测试编号（STDF PTR/FTR 的 TEST_NUM）
    name: str                     # 测试项名
    kind: str                     # 'PARM'（参数，走仪器测量）/ 'FUNC'（功能）
    unit: str | None              # 单位（功能项无）
    spec_lo: float | None         # 规格下限（设计规格，真实判定依据）
    spec_hi: float | None         # 规格上限
    test_lo: float | None = None  # 测试限（guardband 内缩后，limits 模块填充）
    test_hi: float | None = None
    instrument: str | None = None # 测量仪器（DMM / None）
    pin: str | None = None        # 关联引脚
    meas_time_s: float = 0.0      # 单项测试时间（W4 优化基数）

    def judge(self, value: float | bool) -> bool:
        """判定：参数项对测试限比较，功能项直接取布尔结果。"""
        if self.kind == "FUNC":
            return bool(value)
        v = float(value)
        if self.test_lo is not None and v < self.test_lo:
            return False
        if self.test_hi is not None and v > self.test_hi:
            return False
        return True

    def with_test_limits(self, test_lo: float | None, test_hi: float | None) -> "TestItem":
        """返回填充测试限后的新实例（frozen dataclass 语义）。"""
        return TestItem(
            test_num=self.test_num, name=self.name, kind=self.kind, unit=self.unit,
            spec_lo=self.spec_lo, spec_hi=self.spec_hi,
            test_lo=test_lo, test_hi=test_hi,
            instrument=self.instrument, pin=self.pin, meas_time_s=self.meas_time_s,
        )
