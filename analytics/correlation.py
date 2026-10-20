"""测试项相关性与测试时间优化（设计文档 §3.6 压轴深度点）。

方法（文档照抄）：
1. 算全部参数测试项的相关矩阵（Pearson，良品口径）；
2. 找 |r| > 0.95 的冗余对（同工艺强相关的测试项，如文档预期的 VOUT1/VOUT2）；
3. 删除冗余项后用全部 die 复跑 bin 判定，证明 bin 结果零差异——
   用数据证明而不是假设；
4. 输出测试时间基线 → 优化后（节省百分比）。

诚实口径：本仿真 DUT 模型的参数噪声独立（无共享工艺分量），主 lot 的
交叉相关来自抽样噪声与缺陷结构，预期无 |r|>0.95 对——实测如实报告；
冗余识别与"删项复跑零差异"的完整机制由 tests/test_correlation.py 用合成
冗余对验证。真实产线中 VOUT1/VOUT2 共享工艺 corners，相关对确实存在。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REDUNDANT_R_THRESHOLD = 0.95


def param_names_from(results: pd.DataFrame) -> list[str]:
    """结果表中的参数列（排除元数据/bin/功能布尔列）。"""
    skip = {"wafer", "x", "y", "soft_bin", "soft_bin_name", "hard_bin",
            "test_time_s", "part_id", "lot", "program", "head_num",
            "site_num", "die_id"}
    names = []
    for c in results.columns:
        if c in skip or c.endswith("_pass") or c == "hard_bin_name":
            continue
        if pd.api.types.is_float_dtype(results[c]):
            names.append(c)
    return names


def correlation_matrix(results: pd.DataFrame, good_only: bool = True) -> pd.DataFrame:
    """参数 Pearson 相关矩阵（良品口径，pairwise 完整）。"""
    df = results[results["hard_bin"] == 1] if good_only else results
    names = param_names_from(results)
    return df[names].corr(method="pearson")


def redundant_pairs(corr: pd.DataFrame,
                    threshold: float = REDUNDANT_R_THRESHOLD) -> list[dict]:
    """|r| > 阈值的冗余对列表（去重：每对只报一次）。"""
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                pairs.append({"item_a": cols[i], "item_b": cols[j], "r": float(r)})
    return sorted(pairs, key=lambda p: -abs(p["r"]))


def rerun_bins_without(results: pd.DataFrame, drop: list[str],
                       limits: pd.DataFrame,
                       exec_order: list[str]) -> pd.Series:
    """删除指定参数项后，用全部 die 复跑 hard bin 判定（零差异证明用）。

    规则与 testexec.binning 一致：功能 fail → HBin3（优先），参数 fail → HBin2，
    全过 → HBin1。功能项不经参数项删除影响（drop 只含参数项）。
    """
    lim = {r.test_txt: r for r in limits.itertuples()}
    n = len(results)
    param_fail = np.zeros(n, dtype=bool)
    func_fail = np.zeros(n, dtype=bool)
    for name in exec_order:
        if name in drop:
            continue
        col_pass = results[f"{name}_pass"].to_numpy(dtype=bool)
        if name in lim:  # 参数项：用测试限重判（删项后其余项判定不变）
            v = results[name].to_numpy(dtype=float)
            lo, hi = lim[name].lo_limit, lim[name].hi_limit
            ok = np.ones(n, dtype=bool)
            if lo is not None and not np.isnan(lo):
                ok &= v >= lo
            if hi is not None and not np.isnan(hi):
                ok &= v <= hi
            param_fail |= ~ok
        else:            # 功能项：沿用 FTR 结果
            func_fail |= ~col_pass
    return pd.Series(np.where(func_fail, 3, np.where(param_fail, 2, 1)),
                     index=results.index, name="hard_bin")


def test_time_report(meas_times: dict[str, float], drop: list[str],
                     baseline_s: float | None = None,
                     setup_time_s: float = 0.0) -> dict:
    """测试时间优化报告：基线（全部 EXECUTE 项）→ 删冗余项后。"""
    base = baseline_s if baseline_s is not None else sum(meas_times.values())
    kept = {k: v for k, v in meas_times.items() if k not in drop}
    opt = setup_time_s + sum(kept.values())
    base_total = setup_time_s + base
    return {
        "baseline_items": len(meas_times),
        "dropped_items": drop,
        "baseline_s": base_total,
        "optimized_s": opt,
        "saved_s": base_total - opt,
        "saved_pct": (base_total - opt) / base_total if base_total else 0.0,
        "kept_items": sorted(kept),
    }


def save_report(report: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return out_path


def plot_correlation(corr: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("参数测试项相关矩阵（HBin1，50k dies）")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
