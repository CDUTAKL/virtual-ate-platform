"""stdf 包：STDF V4 数据层（writer / reader / 记录定义 / CP 元数据构建）。

设计文档 §3.4：SEMI 标准二进制格式，ATE 行业通用数据语言。writer + reader
均自实现，round-trip 逐字段比对（pytest 覆盖）。
"""
