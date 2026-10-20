"""关键参数分布直方图 + 正态拟合 + 规格限/测试限竖线（设计文档 §3.5）。

对 4 个关键参数（VOUT1 / VOUT3 / RIPPLE3 / IQ1）画：
- 直方图：CP 良品（hard_bin=1）实测值分布（工艺分布主体；缺陷 die 超出规格
  1~2 个数量级，画进同一张图只会压扁坐标，故以 n_out_of_spec 统计标注）
- 正态拟合曲线：N(μ, σ²) 按组距缩放叠加
- 竖线：规格上限/下限（红虚线，判定依据）+ 测试限（橙点线，guardband 内缩后，
  test_limit = spec_limit ∓ k·σ_meas，k=3，testexec/limits.py）
- guardband 区（测试限与规格限之间）浅橙阴影 = 过杀/漏杀权衡区

输出：outputs/char/dist_<param>.png × 4 + dist_summary.json。
数据与测试限全部来自 W2 产物（cp_results.parquet / TestPlan），只读不改。
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scistats

from testexec.plan import TestPlan

#: 文档 §4 要求展示 guardband 余量的关键参数
PARAMS_TO_PLOT = ("VOUT1", "VOUT3", "RIPPLE3", "IQ1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CP = PROJECT_ROOT / "outputs" / "cp_results.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "char"


def load_data(cp_path: Path = DEFAULT_CP) -> pd.DataFrame:
    return pd.read_parquet(cp_path)


def _count_out_of_spec(v: np.ndarray, spec_lo: float | None,
                       spec_hi: float | None) -> int:
    bad = np.zeros(len(v), dtype=bool)
    if spec_lo is not None:
        bad |= v < spec_lo
    if spec_hi is not None:
        bad |= v > spec_hi
    return int(bad.sum())


def compute_stats(cp: pd.DataFrame, plan: TestPlan) -> dict:
    """每个参数的分布统计 + 限值 + 余量（μ 到最近测试限的距离，σ 数）。

    分布主体取 hard_bin=1 的良品（工艺分布）；越规格统计覆盖全部 die。
    """
    good = cp[cp["hard_bin"] == 1]
    out = {}
    for name in PARAMS_TO_PLOT:
        item = plan.item_by_name[name]
        v = good[name].to_numpy(dtype=float)
        mu, sd = float(v.mean()), float(v.std())
        all_v = cp[name].to_numpy(dtype=float)
        dists = []
        if item.test_lo is not None:
            dists.append(mu - item.test_lo)
        if item.test_hi is not None:
            dists.append(item.test_hi - mu)
        out[name] = {
            "n_good": int(len(v)),
            "n_fail_total": int((cp["hard_bin"] != 1).sum()),
            "n_out_of_spec": _count_out_of_spec(all_v, item.spec_lo, item.spec_hi),
            "mean": mu,
            "std": sd,
            "spec_lo": item.spec_lo,
            "spec_hi": item.spec_hi,
            "test_lo": item.test_lo,
            "test_hi": item.test_hi,
            "unit": item.unit,
            "margin_to_test_limit_sigma": float(min(dists) / (3.0 * sd)),
        }
    return out


def plot_distribution(stats: dict, values: dict[str, np.ndarray],
                      out_dir: Path) -> dict:
    """按参数出图（values 为各参数良品实测数组），返回 {param: path}。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, st in stats.items():
        v = np.asarray(values[name], dtype=float)
        # x 轴必须覆盖规格限/测试限（否则竖线落在数据范围外被裁掉）
        lims = [l for l in (st["spec_lo"], st["spec_hi"],
                            st["test_lo"], st["test_hi"]) if l is not None]
        xmin = min(float(v.min()), *lims)
        xmax = max(float(v.max()), *lims)
        pad = (xmax - xmin) * 0.04
        xmin, xmax = xmin - pad, xmax + pad
        fig, ax = plt.subplots(figsize=(8, 5))
        counts, bins, _ = ax.hist(v, bins=100, range=(xmin, xmax),
                                  color="#9ecae1",
                                  edgecolor="none", alpha=0.85,
                                  label=f"good dies (n={st['n_good']:,})")
        x = np.linspace(xmin, xmax, 400)
        bin_w = (xmax - xmin) / 100
        curve = scistats.norm.pdf(x, st["mean"], st["std"]) * len(v) * bin_w
        ax.plot(x, curve, "b-", lw=2, label="normal fit")
        # 规格限 / 测试限竖线（标签错开两档高度，避免重叠）
        ymax = ax.get_ylim()[1]
        for lim, lbl, color, ls, ytxt in (
                (st["spec_lo"], "spec lo", "#d62728", "--", 0.97),
                (st["spec_hi"], "spec hi", "#d62728", "--", 0.97),
                (st["test_lo"], "test lo (guardband)", "#ff7f0e", ":", 0.55),
                (st["test_hi"], "test hi (guardband)", "#ff7f0e", ":", 0.55)):
            if lim is not None:
                ax.axvline(lim, color=color, ls=ls, lw=1.8)
                ha = "right" if lim > (xmin + xmax) / 2 else "left"
                ax.text(lim, ymax * ytxt, f"{lbl}\n{lim:g} ",
                        color=color, fontsize=8, va="top", ha=ha)
        # guardband 区阴影（测试限 ↔ 规格限）
        for lo, hi in ((st["test_lo"], st["spec_lo"]),
                       (st["spec_hi"], st["test_hi"])):
            if lo is not None and hi is not None and hi > lo:
                ax.axvspan(lo, hi, color="#ffbb78", alpha=0.25, lw=0)
        ax.set_title(f"{name} distribution (good dies)  "
                     f"μ={st['mean']:.5g}  σ={st['std']:.4g} {st['unit']}")
        ax.set_xlabel(f"{name} ({st['unit']})")
        ax.set_ylabel("count")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        ax.text(0.98, 0.72,
                f"margin to nearest test limit: "
                f"{st['margin_to_test_limit_sigma']:.2f}×3σ\n"
                f"out-of-spec dies (all): {st['n_out_of_spec']}",
                transform=ax.transAxes, ha="right", fontsize=9)
        path = out_dir / f"dist_{name.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[name] = path
    return paths


def run(cp_path: Path = DEFAULT_CP, out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    """执行完整分布分析：统计 → 画图 → summary JSON。"""
    cp = load_data(cp_path)
    plan = TestPlan.load()
    stats = compute_stats(cp, plan)
    good = cp[cp["hard_bin"] == 1]
    values = {name: good[name].to_numpy(dtype=float) for name in stats}
    paths = plot_distribution(stats, values, out_dir)

    (Path(out_dir) / "dist_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, p in paths.items():
        st = stats[name]
        print(f"{name}: n={st['n_good']}, μ={st['mean']:.5g}, σ={st['std']:.4g}, "
              f"余量={st['margin_to_test_limit_sigma']:.2f}×3σ -> {p}")
    return {"stats": stats, "paths": paths}


if __name__ == "__main__":
    run()
