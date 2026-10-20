"""全局配置：随机种子、晶圆几何、缺陷注入参数、各参数波动 σ 与温度系数。

所有统计参数集中在本文件，便于调参。固定 SEED 保证全流程可复现。
"""

# ---- 随机种子与批次 ----
SEED = 20261020
LOT_ID = "LOT20261020A"

# ---- 晶圆几何 ----
N_WAFERS = 25                    # 片数
DIE_PER_WAFER = 2000             # 每片 die 数（25 × 2000 = 50,000）
GRID_N = 50                      # 50×50 候选网格，取距圆心最近的 2000 个 die
EDGE_RING_WIDTH = 2.2            # 边缘环宽度（die 单位），覆盖最外 1~2 圈
EDGE_FACTOR = 3.0                # 边缘环缺陷概率倍率（文档 §3.1）

# ---- 成簇缺陷 ----
CLUSTER_MIN = 0                  # 每片簇数下限
CLUSTER_MAX = 3                  # 每片簇数上限
CLUSTER_R_MIN = 5.0              # 簇半径下限（die 单位）
CLUSTER_R_MAX = 30.0             # 簇半径上限
CLUSTER_FACTOR = 10.0            # 簇内缺陷概率倍率

# ---- 随机缺陷密度 ----
# D0 = 权重 1 的 die 的基准缺陷概率；边缘/簇内 die 按倍率放大。
# 缺陷 die 有相当一部分（DEF_DRIFT/DEF_TIMING）在 25°C CP 下是潜伏的，
# 因此整体良率由 D0 与各类型权重共同决定，目标区间 85%~97%。
D0 = 0.022

# ---- 缺陷类型权重（缺陷 die 的类型分布）----
DEFECT_WEIGHTS = {
    "DEF_OFFSET": 0.28,   # 基准失调：某通道输出电压偏移 +4~6%
    "DEF_DRIFT": 0.20,    # 温漂异常：温度系数放大 5~10 倍（25°C 潜伏）
    "DEF_LEAK": 0.28,     # 漏电流超标：静态电流 ×5~20
    "DEF_OPEN": 0.12,     # 引脚开路：功能测试 fail
    "DEF_TIMING": 0.12,   # 环路不稳定：低压高温重载下震荡（25°C 潜伏）
}

# ---- 波动模型比例（文档 §3.1：σ_lot 取 σ_die 的经验比）----
SIGMA_LOT_RATIO = 0.35   # σ_lot = 0.35 × σ_die（批次间漂移）
SIGMA_GRAD_RATIO = 0.40  # σ_grad = 0.40 × σ_die（片上径向梯度幅度）

# ---- 各参数 die 间随机 σ（绝对值，单位与参数一致）----
# 按目标 Cpk 1.0~1.6 反推（σ_tot ≈ σ_die × 1.12，含梯度贡献）。
# 文档 §4 预期 Cpk 最低参数为 VOUT3 / RIPPLE3（1.0~1.3），此处对应收紧。
SIGMA_DIE = {
    "VOUT1": 0.0160,    # ±0.066 V 半宽 → Cpk ≈ 1.1~1.3
    "IQ1": 6.3,         # ≤120 µA，nominal 90 → Cpk ≈ 1.4~1.6
    "LINE1": 0.0072,    # ≤0.05 %/V，nominal 0.02 → Cpk ≈ 1.4~1.6
    "VOUT2": 0.0080,    # ±0.036 V 半宽 → Cpk ≈ 1.2~1.4
    "IQ2": 4.2,         # ≤80 µA，nominal 60 → Cpk ≈ 1.3~1.5
    "VOUT3": 0.0068,    # ±0.024 V 半宽 → Cpk ≈ 1.0~1.3（全参数最低，文档 §4）
    "EFF3": 1.06,       # ≥85 %，nominal 90 → Cpk ≈ 1.4~1.6
    "RIPPLE3": 3.05,    # ≤30 mVpp，nominal 18 → Cpk ≈ 1.1~1.4（次低）
    "ADC_INL": 0.80,    # ±4 LSB → Cpk ≈ 1.3~1.5
    "TEMP25": 0.58,     # ±3 °C → Cpk ≈ 1.4~1.6
    "ISD": 0.215,       # ≤2 µA，nominal 1.1 → Cpk ≈ 1.4~1.6
    "POR": 0.0215,      # 1.55~1.75 V，半宽 0.10 → Cpk ≈ 1.4~1.6
}

# ---- 典型一阶温度系数 ----
# tc_rel=True 的参数：v(T) = v(25) × (1 + tc1 × ΔT)
# tc_rel=False 的参数：v(T) = v(25) + tc1 × ΔT（绝对量）
TC1 = {
    "VOUT1": (-8.0e-5, True),    # -0.8%/100°C，正常 die 全温区内不出规格
    "VOUT2": (+8.0e-5, True),
    "VOUT3": (-1.0e-4, True),    # -1%/100°C
    "IQ1": (0.0198, True),       # 约每 35°C 翻倍（高温下不套用 25°C 限值）
    "IQ2": (0.0198, True),
    "ISD": (0.0277, True),       # 约每 25°C 翻倍
    "EFF3": (-0.02, False),      # -2 百分点/100°C
    "RIPPLE3": (+0.02, False),   # +2 mVpp/100°C
    "ADC_INL": (+0.008, False),  # +0.8 LSB/100°C
    "TEMP25": (0.0, False),      # 规格定义在 25°C，无温度外推
    "POR": (-5.0e-4, False),     # -0.5 mV/°C
    "LINE1": (0.0, False),
}

# 每 die 温度系数相对典型值的抖动（标准差比例，模拟 die 间温漂差异）
TC1_JITTER = 0.30

# ---- CP 测量条件（W1 晶圆生成器的测量口径）----
CP_TEMP_C = 25.0
CP_VIN_V = 5.0
CP_LOAD_A = 1.0

# DEF_TIMING 的震荡触发条件（Buck 低压高温重载）
STRESS_TEMP_C = 85.0
STRESS_LOAD_A = 1.5
