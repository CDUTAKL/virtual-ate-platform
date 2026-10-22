"""跨平台数值指纹：定位 macOS/Linux 数据流分叉点（临时诊断工具）。

打印三级指纹：
1. 原始 RNG 流（default_rng 前几个值，验证 bit generator 跨平台一致性）
2. lot 生成结果（各缺陷类型计数，验证生成阶段）
3. CP 测量结果（首 5 个 VOUT1 测量值，验证测量/噪声阶段）
"""
import numpy as np

from dut import wafer
from testexec.plan import TestPlan
from testexec import cp
from dut import config

print(f"SEED={config.SEED}")
rng = np.random.default_rng(config.SEED)
print("rng.random(5):", [f"{v:.17g}" for v in rng.random(5)])
rng2 = np.random.default_rng(config.SEED)
print("rng.normal(5):", [f"{v:.17g}" for v in rng2.normal(size=5)])

lot, dies = wafer.generate_lot()
print("lot defect histogram:")
print(lot["defect"].astype(str).value_counts().to_string())
print("lot first5 VOUT1:", [f"{v:.17g}" for v in lot["VOUT1"].iloc[:5]])

plan = TestPlan.load()
res = cp.run_cp(lot, plan)
print("cp first5 VOUT1 measured:", [f"{v:.17g}" for v in res["VOUT1"].iloc[:5]])
print("cp pass counts:", res["VOUT1_pass"].value_counts().to_dict())
print("hard_bin counts:", res["hard_bin"].value_counts().to_dict())
