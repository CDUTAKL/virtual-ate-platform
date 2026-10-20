"""量产良率分析（设计文档 §3.6）。

指标（TE 口径）：
- yield = good/total（hard bin 1）
- FPY（first-pass yield）：本流程无 retest，FPY = yield（真实产线 FPY ≤ yield）
- DPPM = 不良数/总数 × 10^6
- bin 帕累托：soft/hard bin 计数与累计占比（找主要失效模式）
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def summary(results: pd.DataFrame) -> dict:
    """lot 级良率摘要。"""
    total = len(results)
    good = int((results["hard_bin"] == 1).sum())
    bad = total - good
    return {
        "total": total,
        "good": good,
        "bad": bad,
        "yield": good / total,
        "fpy": good / total,          # 无 retest，FPY=yield
        "dppm": bad / total * 1e6,
    }


def bin_pareto(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """hard/soft bin 帕累托表：计数、占比、累计占比。"""
    def _pareto(df: pd.DataFrame, col: str, name_col: str) -> pd.DataFrame:
        t = (df.groupby([col, name_col]).size().rename("count")
             .reset_index().sort_values("count", ascending=False))
        t["pct"] = t["count"] / t["count"].sum()
        t["cum_pct"] = t["pct"].cumsum()
        return t.reset_index(drop=True)

    hard = results.groupby("hard_bin").size().rename("count").reset_index()
    hard["name"] = hard["hard_bin"].map({1: "GOOD", 2: "PARAM_FAIL", 3: "FUNC_FAIL"})
    hard = hard.sort_values("count", ascending=False).reset_index(drop=True)
    hard["pct"] = hard["count"] / hard["count"].sum()
    hard["cum_pct"] = hard["pct"].cumsum()

    soft = (results.groupby(["soft_bin", "soft_bin_name"]).size().rename("count")
            .reset_index().sort_values("count", ascending=False))
    soft["pct"] = soft["count"] / soft["count"].sum()
    soft["cum_pct"] = soft["pct"].cumsum()
    return hard.reset_index(drop=True), soft.reset_index(drop=True)


def plot_bin_pareto(results: pd.DataFrame, out_path: str | Path) -> Path:
    """soft bin 帕累托图（bar + 累计占比折线），输出 PNG。"""
    _, soft = bin_pareto(results)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"S{r.soft_bin}\n{r.soft_bin_name}" for r in soft.itertuples()]
    ax.bar(range(len(soft)), soft["count"], color="#4c72b0")
    ax.set_xticks(range(len(soft)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("fail 计数")
    ax.set_title("Soft Bin Pareto（25°C CP，含 PASS）")
    ax2 = ax.twinx()
    ax2.plot(range(len(soft)), soft["cum_pct"] * 100, "ro-", markersize=4)
    ax2.set_ylabel("累计占比 %")
    ax2.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
