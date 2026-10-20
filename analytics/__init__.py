"""analytics 包：量产分析（只消费 STDF 解析结果，不接触 DUT——文档 §2.1 架构边界）。

- yield       良率 / FPY / DPPM / bin 帕累托
- capability  Cpk / Ppk（测试限口径，良品集）
- spc         I-MR 控制图 + Western Electric 四规则 + 注入漂移演示
- correlation 参数相关矩阵 / 冗余对识别 / 删项复跑零差异 / 测试时间优化
"""
