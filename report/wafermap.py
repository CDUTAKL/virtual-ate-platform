"""Wafer map 可视化：按 (X, Y) 画 fail die 分布。

颜色约定：
- PASS   浅绿（good die）
- PARAM  红（参数越限，CP 参数测试可抓）
- FUNC   蓝（功能 fail）
- BOTH   紫（参数+功能都 fail）

输出一张 PNG，分两部分：
- 上部：整批 25 片小图（5×5）
- 下部：三种空间缺陷模式的代表片放大图（自动选取）
  - 边缘环：非簇边缘 fail 率最高的片
  - 成簇：簇内 fail 率最高的片
  - 随机：无簇且边缘效应最弱的片

预期表现：边缘环片外圈 fail 密度明显高于中心（×3 概率）；
成簇片在簇区域（半径 5~30 die）成片聚集（×10 概率）；
随机片 fail 全片均匀散布（D0）。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dut import config

# 中文字体：macOS 自带 PingFang SC / Hiragino Sans GB
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 颜色映射
FAIL_COLORS = {
    "PASS": "#c8e6c9",
    "PARAM": "#d62728",
    "FUNC": "#1f77b4",
    "BOTH": "#7b3294",
}
EDGE_COLOR = "#555555"


def plot_wafer(ax, df_wafer: pd.DataFrame, title: str = "", marker_size: int = 16) -> None:
    """在指定 Axes 上画单片 wafer map。"""
    r_max = np.hypot(df_wafer["x"], df_wafer["y"]).max()
    for kind, color in FAIL_COLORS.items():
        sub = df_wafer[df_wafer["fail_kind"] == kind]
        if len(sub):
            ax.scatter(sub["x"], sub["y"], s=marker_size, c=color, marker="s",
                       linewidths=0, label=kind, zorder=2)
    # 晶圆轮廓
    circle = plt.Circle((0, 0), r_max + 0.6, fill=False, color=EDGE_COLOR,
                        linewidth=1.0, zorder=3)
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlim(-r_max - 1.5, r_max + 1.5)
    ax.set_ylim(-r_max - 1.5, r_max + 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    bad = (~df_wafer["pass"]).sum()
    ax.set_title(f"{title}  yield {df_wafer['pass'].mean() * 100:.1f}%"
                 f"  ({int(bad)} fail)", fontsize=9)


def _mode_picks(dies: pd.DataFrame) -> dict[str, int]:
    """为三种空间模式各选一片代表 wafer（依据 region 存档标签）。"""
    stats = []
    for w, grp in dies.groupby("wafer"):
        edge = grp[grp["region"] == "EDGE"]
        inner = grp[grp["region"] == "INNER"]
        clu = grp[grp["region"] == "CLUSTER"]
        stats.append({
            "wafer": w,
            "edge_fail": (~edge["pass"]).mean() if len(edge) else np.nan,
            "inner_fail": (~inner["pass"]).mean() if len(inner) else np.nan,
            "cluster_fail": (~clu["pass"]).mean() if len(clu) else np.nan,
            "n_cluster_dies": len(clu),
        })
    s = pd.DataFrame(stats).set_index("wafer")
    edge_pick = int(s["edge_fail"].idxmax())
    has_cluster = s[s["n_cluster_dies"] > 0]
    # 成簇代表：簇内 fail 的 die 数最多（大簇 × 高簇内 fail 率，图上一眼成团）
    cluster_pick = int((has_cluster["n_cluster_dies"] * has_cluster["cluster_fail"]).idxmax())
    no_cluster = s[s["n_cluster_dies"] == 0]
    random_pick = int(no_cluster["edge_fail"].idxmin())
    return {"边缘环": edge_pick, "成簇": cluster_pick, "随机": random_pick}


def plot_lot(dies: pd.DataFrame, out_path: str | Path,
             lot_id: str = config.LOT_ID) -> Path:
    """画整批 wafer map（25 片小图 + 三种模式代表片放大图），输出 PNG。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_wafers = dies["wafer"].nunique()
    cols = 5
    rows = int(np.ceil(n_wafers / cols))
    overall_yield = dies["pass"].mean()

    fig = plt.figure(figsize=(4 * cols, 4 * rows + 5.0))
    # 显式分区：上部 5×5 小图，下部模式代表片 + 图例
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[rows * 4.0, 4.6],
                                hspace=0.05, left=0.02, right=0.98,
                                top=0.955, bottom=0.01)
    sub_top = fig.add_subfigure(gs_outer[0, 0])
    sub_bot = fig.add_subfigure(gs_outer[1, 0])
    gs_b = sub_bot.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.45], wspace=0.15)

    axes = np.atleast_1d(sub_top.subplots(rows, cols)).ravel()
    for w in range(1, n_wafers + 1):
        plot_wafer(axes[w - 1], dies[dies["wafer"] == w], title=f"W{w:02d}")
    for ax in axes[n_wafers:]:
        ax.axis("off")

    # 底部：三种空间模式代表片放大
    picks = _mode_picks(dies)
    notes = {
        "边缘环": "外 1~2 圈缺陷概率 ×3：外圈 fail 明显多于中心",
        "成簇": "每片 0~3 簇、半径 5~30 die：簇内缺陷概率 ×10",
        "随机": "无簇片：基准缺陷密度 D0，fail 全片均匀散布",
    }
    axes_b = [sub_bot.add_subplot(gs_b[0, k]) for k in range(3)]
    for ax, (mode, w) in zip(axes_b, picks.items()):
        plot_wafer(ax, dies[dies["wafer"] == w],
                   title=f"{mode}模式代表 W{w:02d}", marker_size=42)
        ax.set_xlabel(notes[mode], fontsize=9)

    # 图例独占第 4 格，避免与模式说明文字重叠
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
                          markersize=8, label=k) for k, c in FAIL_COLORS.items()]
    ax_leg = sub_bot.add_subplot(gs_b[0, 3])
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, loc="center left", fontsize=10,
                  title="25°C CP 结果\n（缺陷标签仅存档，\n测试程序不可见）")

    fig.suptitle(f"{lot_id} wafer map — 整体良率 {overall_yield * 100:.2f}%"
                 f"（{len(dies):,} dies，{n_wafers} wafers × "
                 f"{len(dies) // n_wafers:,} dies）", fontsize=14)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
