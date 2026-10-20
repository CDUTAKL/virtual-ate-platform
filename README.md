# 虚拟 ATE 测试开发平台（Virtual ATE Platform）

以纯软件方式复现"量产测试开发 → 特性化 → 量产数据分析 → 失效定位"完整闭环的**演练平台**。
W1~W4 全部交付：虚拟芯片模型 + 晶圆生成器 → 仪器层 + 测试程序框架 + CP 全流程 → STDF V4 数据层 +
量产分析 → Shmoo 特性化 + 失效聚类。

> **诚实声明**
> - PMU300 为虚构芯片（行为模型，非晶体管级模型）；所有数据来自固定随机种子的蒙特卡洛仿真，
>   **不是真实产线数据**，但缺陷机制、波动结构、测试流程与分析方法对标真实量产测试场景。
> - 仪器层为 pyvisa-sim 仿真（SCPI 命令集与真实仪器一致），**未接触真实 ATE 机台**；
>   测试程序框架为概念对标 STS8200 / V93000 结构，不声称使用真实机台 API。
> - 缺陷标签（`defect` 列，仅 lot_dies.csv.gz 存档）**测试程序不可见**；
>   CP 结果表与 STDF 文件均不含标签——检出率/混淆矩阵对照是事后 join 存档的验证动作。
> - 分析层只消费 STDF 文件（round-trip 解析后的结果表），不接触 DUT 模型（文档 §2.1 架构边界）。

## 环境准备

```bash
uv venv && uv pip install numpy pandas matplotlib pyyaml pytest pyvisa pyvisa-sim pyarrow scipy scikit-learn jinja2
# 或使用 python3 -m venv .venv && .venv/bin/pip install ...
```

## 运行方法

```bash
.venv/bin/python main.py                 # 一键全流程（W1 晶圆 → W2 CP → W3 STDF+分析 → W4 特性化+FA → 报告）
.venv/bin/python -m pytest tests/ -q     # 单元测试（118 个用例，W1 24 + W2 24 + W3 39 + W4 26 + 报告层 5）
```

全流程末尾自动生成 **`outputs/report.html`**：自包含 HTML 测试报告（KPI 卡 +
图表 base64 内联，约 1.5 MB 单文件，无外部依赖）。

实测运行环境：Apple Silicon Mac，Python 3.14.5（uv venv）；`python main.py` 全流程 **约 26 s**
（W1+W2 约 2.4 s，W3 约 21 s——其中 STDF 写出 12.3 s + 读回 8.3 s，W4 约 1.2 s）。

## 数据口径（全部实测，seed=20261020 可复现；2026-10-21 回填）

| 数字 | 实测值 | 口径/出处 |
|---|---|---|
| 模拟 die 总数 | **50,000**（25 wafer × 2,000） | 生成器直出 |
| 测试项 | **12 参数 + 3 功能** | configs/cp_plan.yaml |
| 整体良率 | **90.59% / 90.35%** | 规格限 / guardband 测试限（差 0.24% 即过杀） |
| hard bin | H1=45176 / H2=4034 / H3=790 | CP 结果 |
| DPPM | 96,480 | 测试限口径 |
| **CP 可检出缺陷检出率** | **100.00%（4654/4654）** | 要求 ≥99% |
| guardband 过杀率 | **0.38%（164/43224）** | 无缺陷 die 被误杀 |
| 潜伏缺陷逃逸 | **2,116 / 2,122** | DRIFT/TIMING 需多温/FT/Shmoo |
| Cpk 最低参数 | **VOUT3 Cpk=1.02 / Ppk=1.00** | 测试限口径，HBin1；其余 11 项 ≥1.33 合格 |
| guardband 余量最小 Top3 | VOUT3 1.00σ / VOUT1 1.01σ / VOUT2 1.15σ | 漂移风险最高 |
| **STDF 文件** | **52.5 MB，850,098 条记录**（写出 12.3 s + 读回 8.3 s） | V4 二进制自实现，round-trip 逐字段比对 |
| SPC 演示 | 注入 +21 mV 爬坡（W13~19）→ **第 19 片触发规则①**；干净 lot 零报警 | I-MR + Western Electric |
| **FA 混淆矩阵** | **对角线命中率 96.48%（目标 ≥90%），轮廓系数 0.5056** | K-Means K=5，5 失效子模式 |
| Shmoo 失效边界 | **T≥85°C × 2 A**（DEF_TIMING 高温重载角）；DEF_DRIFT 温度两端越限 | V×T / V×Load 双图 |
| 测试时间优化 | 主 lot **无 \|r\|>0.95 冗余对（max \|r\|=0.038）如实报告 0%**；识别→删项→5 万 die 复跑零差异机制由合成对验证 | 基线 0.460 s/die |
| 单元测试 | **118 个全绿** | pytest |

### 检出率口径与已知边界（两层报告，面试必问）

- **CP 可检出缺陷**（DEF_OFFSET 输出失调 / DEF_LEAK 漏电 / DEF_OPEN 开路）：
  检出率 **100.00%**。漏检只可能来自 guardband 边界翻转——本批为 0。
- **逃逸清单**（已知边界）：DEF_DRIFT（温漂 ×5~10，1,313 颗）与 DEF_TIMING
  （环路不稳，809 颗）在 25°C CP 下物理潜伏（对应真实产线需多温 CP/FT/Shmoo
  抓出的失效），逃逸 2,116 颗。char/shmoo.py 演示了二者在应力角的暴露；
  FA 报告中其混淆矩阵行为近零是"检出手段"问题而非聚类失败。

## 模型与机制要点

- 参数波动：`nominal + lot_offset(N(0,σ_lot)) + wafer_gradient(径向) + die_noise(N(0,σ_die)) + 温度系数`
- 测量模型（文档 §3.2）：`measured = true×(1+N(0,0.05%)) + N(0,σ_offset)`，
  σ_meas = hypot(0.05%×读数, σ_offset)，guardband = 规格限内缩 k·σ_meas（k=3.0）。
- 五类缺陷：OFFSET/LEAK 在 CP 越限检出；OPEN 功能 fail 检出；
  DRIFT/TIMING 25°C 潜伏（≥85°C 且 ≥1.5A 暴露）。
- 空间缺陷模式：边缘环 ×3、成簇 ×10（半径 5~30 die）、随机 D0——wafer map 肉眼可分。

## CI

`.github/workflows/ci.yml`：push 后自动 setup uv → 安装依赖 → `pytest -q` →
`python main.py` 全流程 → upload-artifact 留存 outputs/。
徽章占位（推送到远程后替换 `<user>/<repo>`）：

```markdown
![CI](https://github.com/<user>/<repo>/actions/workflows/ci.yml/badge.svg)
```

## 目录结构（W1~W4 完整）

```
virtual-ate-platform/
├── dut/                  # 虚拟芯片与晶圆（W1）
│   ├── config.py         # 全部可调参数（σ、缺陷率、D0、seed）
│   ├── pmu300.py         # PMU300 行为模型 + 单 die 测量 API
│   ├── defects.py        # 5 类缺陷定义
│   └── wafer.py          # 晶圆/批次生成器（边缘环/成簇/随机）
├── instruments/          # 仪器层（W2，文档 §3.2）
│   ├── power_supply.yaml # pyvisa-sim PSU（*IDN?/APPLy/VOLTage/MEASure?）
│   ├── dmm.yaml          # pyvisa-sim DMM（*IDN?/CONFigure/READ?）
│   └── driver.py         # 统一驱动：SCPI 往返 + 测量噪声模型 + σ_meas 标定
├── testexec/             # 测试程序框架（W2，文档 §3.3）
│   ├── plan.py           # TestPlan（YAML 加载、规格一致性校验、guardband）
│   ├── flow.py           # TestFlow（SETUP→EXECUTE→BINNING）
│   ├── item.py           # TestItem 抽象
│   ├── binning.py        # soft/hard bin 判定（功能优先 HBin3）
│   ├── limits.py         # guardband = 规格限 ∓ k·σ_meas
│   └── cp.py             # CP 执行器（批量/SCPI 双路径 + 检出率事后分析）
├── stdf/                 # STDF V4 数据层（W3，文档 §3.4）
│   ├── records.py        # 16 种记录二进制编解码（自实现）
│   ├── writer.py         # 完整 CP 文件 writer（FAR→MRR，85 万记录）
│   ├── reader.py         # 解析回结构化对象 + to_results_frame（分析层唯一入口）
│   └── cp_meta.py        # CP 元数据构建（限值与 CP 判定单一来源）
├── analytics/            # 量产分析（W3，文档 §3.6；只消费 STDF）
│   ├── yield.py          # 良率/FPY/DPPM/bin 帕累托
│   ├── capability.py     # Cpk（组内σ）/Ppk（整体σ）
│   ├── spc.py            # I-MR 控制图 + Western Electric 四规则 + 注入漂移演示
│   ├── correlation.py    # 相关矩阵/冗余对/删项复跑零差异/测试时间优化
│   └── run_all.py        # W3 分析编排（一键产物到 outputs/analysis/）
├── char/                 # 特性化（W4，文档 §3.5）
│   ├── shmoo.py          # CH3 Buck Shmoo 扫描（V×T / V×Load 双图 + 失效边界解读）
│   └── distribution.py   # 参数分布直方图（正态拟合 + 规格限 + 测试限三线）
├── fa/                   # 失效分析（W4，文档 §3.7）
│   ├── signature.py      # 失效签名提取（fail 项/偏离方向/坐标/空间簇）
│   └── cluster.py        # K-Means K=5 + 根因反推 + 混淆矩阵（对角线 ≥90%）
├── configs/
│   ├── cp_plan.yaml      # CP 测试计划（限值照抄文档 §3.1）
│   └── bins.yaml         # bin 定义表
├── report/
│   ├── wafermap.py       # wafer map 可视化
│   └── html_report.py    # 一键 HTML 测试报告（KPI 卡 + 图表 base64 内联，§3.8）
├── tests/                # 118 个 pytest 用例
├── outputs/              # 全部产物（gitignore，CI artifact 留存）
└── main.py               # 一键入口（W1+W2+W3+W4 全流程）
```
