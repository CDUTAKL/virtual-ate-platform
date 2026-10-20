"""过程能力分析：Cpk / Ppk（设计文档 §3.6）。

公式（文档照抄）：Cpk = min(USL−μ, μ−LSL)/(3σ)。
口径说明：
- 限值用 guardband 后测试限（分析层从 STDF PTR 记录取，不接触 DUT）。
- 统计总体取 HBin1（良品）die —— 工程惯例：缺陷 die 的极端值（如漏电 ×20）
  会污染 μ/σ，使"过程能力"失真；guardband 截断使结果略偏乐观，与 W1
  无缺陷批规格限 Cpk 对照使用。
- Cpk 的 σ 用组内σ（按 wafer 分组的合并标准差，pooled within-wafer）；
  Ppk 的 σ 用整体σ（全部良品）。组间差异（片间梯度幅度波动）进 Ppk 不进 Cpk。
判定线：Cpk ≥ 1.33 合格、≥ 1.67 优秀（文档 §3.6）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def _one_sided_margin(v: np.ndarray, lo: float | None, hi: float | None) -> float:
    mu = v.mean()
    dists = []
    if not _is_missing(hi):
        dists.append(hi - mu)
    if not _is_missing(lo):
        dists.append(mu - lo)
    return float(min(dists))


def _within_wafer_sigma(df: pd.DataFrame, col: str) -> float:
    """组内σ：按 wafer 分组的分合并标准差（ddof=1）。"""
    vars_, ns = [], []
    for _, g in df.groupby("wafer"):
        v = g[col].to_numpy(dtype=float)
        if len(v) >= 2:
            vars_.append(v.var(ddof=1))
            ns.append(len(v))
    vars_ = np.asarray(vars_)
    ns = np.asarray(ns)
    return float(np.sqrt((vars_ * (ns - 1)).sum() / (ns - 1).sum()))


def capability_table(results: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
    """逐参数 Cpk/Ppk 表。limits 需含列：test_txt, lo_limit, hi_limit。

    results：分析层结果表（stdf.reader.to_results_frame 输出，含 wafer/hard_bin）。
    """
    good = results[results["hard_bin"] == 1]
    rows = []
    for lim in limits.itertuples():
        name = lim.test_txt
        if name not in good.columns:
            continue
        v = good[name].to_numpy(dtype=float)
        mu = float(v.mean())
        s_overall = float(v.std(ddof=1))
        s_within = _within_wafer_sigma(good, name)
        margin = _one_sided_margin(v, lim.lo_limit, lim.hi_limit)
        cpk = margin / (3 * s_within) if s_within > 0 else np.nan
        ppk = margin / (3 * s_overall) if s_overall > 0 else np.nan
        rows.append({
            "item": name, "units": lim.units, "n_good": len(v),
            "mean": mu, "sigma_within": s_within, "sigma_overall": s_overall,
            "test_lo": lim.lo_limit, "test_hi": lim.hi_limit,
            "cpk": cpk, "ppk": ppk,
            "judgement": ("优秀" if cpk >= 1.67 else "合格" if cpk >= 1.33
                          else "偏低" if cpk >= 1.0 else "不足"),
        })
    df = pd.DataFrame(rows).sort_values("cpk").reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def save_capability_csv(table: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False, float_format="%.4f")
    return out_path
