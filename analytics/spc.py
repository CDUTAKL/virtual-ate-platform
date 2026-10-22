"""SPC 统计过程控制（设计文档 §3.6）：I-MR 控制图 + Western Electric 判异规则。

监控对象：关键参数 VOUT1 的 lot 均值序列（按 wafer 顺序，每片一点）。
统计口径：HBin1（良品）die 的 wafer 均值——缺陷 die 的极端值不代表工艺漂移。
控制限（I-MR 标准做法）：
    CL = x̄（wafer 均值序列的均值）
    σ̂ = MR̄ / d2，d2 = 1.128（n=2 移动极差无偏常数）
    I 图：UCL/LCL = x̄ ± 3σ̂；MR 图：UCL = 3.267·MR̄，LCL = 0

Western Electric 四条判异规则（文档 §3.6，作用于 I 图）：
    ① 任一点超出 ±3σ 控制限
    ② 连续 9 点在中心线同侧
    ③ 连续 6 点单调递增或递减
    ④ 连续 14 点上下交替

注入漂移演示（文档 §3.6）：对第 13~19 片施加微小 lot 偏移
（DEF_DRIFT 弱化版，线性爬坡），验证 SPC 报警并报告"第几片触发哪条规则"；
主 lot（seed=20261020 干净数据）不得出现①类（超 3σ）失控报警——③/④
运行/趋势规则在程序生成噪声上存在跨平台边界翻转（见 tools/fingerprint.py
诊断注记），其判定正确性由手工确定性序列单测守护。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

D2 = 1.128          # 移动极差无偏常数（n=2）
MR_UCL_FACTOR = 3.267

# 注入漂移默认参数：第 13~19 片线性爬坡，总位移经实测标定（见 tests/test_spc.py）
DRIFT_WAFERS = (13, 19)
DRIFT_TOTAL_SHIFT = 0.021      # V，21 mV ≈ 0.64% 标称值（微小但 SPC 可逮）


def wafer_means(results: pd.DataFrame, param: str = "VOUT1",
                good_only: bool = True) -> pd.Series:
    """关键参数的 wafer 均值序列（按 wafer 顺序）。good_only=良品口径。"""
    df = results[results["hard_bin"] == 1] if good_only else results
    return df.groupby("wafer")[param].mean().sort_index()


def imr_chart(means: pd.Series) -> dict:
    """I-MR 控制图参数。"""
    x = means.to_numpy(dtype=float)
    mr = np.abs(np.diff(x))
    mr_bar = float(mr.mean())
    x_bar = float(x.mean())
    sigma_hat = mr_bar / D2
    return {
        "means": means,
        "x_bar": x_bar, "mr_bar": mr_bar, "sigma_hat": sigma_hat,
        "ucl": x_bar + 3 * sigma_hat, "lcl": x_bar - 3 * sigma_hat,
        "mr_ucl": MR_UCL_FACTOR * mr_bar,
        "wafers": means.index.to_numpy(),
    }


def western_electric_alarms(means: pd.Series, chart: dict | None = None) -> list[dict]:
    """WE 四规则判异 → 报警列表 [{'wafer': w, 'rule': n, 'detail': str}]。

    返回按 wafer 顺序去重后的报警（同一片多规则全列）。
    """
    if chart is None:
        chart = imr_chart(means)
    x = means.to_numpy(dtype=float)
    wafers = means.index.to_numpy()
    cl, ucl, lcl = chart["x_bar"], chart["ucl"], chart["lcl"]
    alarms: list[dict] = []

    def _emit(i: int, rule: int, detail: str) -> None:
        alarms.append({"wafer": int(wafers[i]), "rule": rule, "detail": detail})

    # ① 任一点超 ±3σ
    for i in range(len(x)):
        if x[i] > ucl:
            _emit(i, 1, f"{x[i]:.5f} > UCL {ucl:.5f}")
        elif x[i] < lcl:
            _emit(i, 1, f"{x[i]:.5f} < LCL {lcl:.5f}")
    # ② 连续 9 点同侧
    for i in range(len(x) - 8):
        seg = x[i:i + 9]
        if (seg > cl).all():
            _emit(i + 8, 2, f"W{wafers[i]}~W{wafers[i+8]} 连续 9 点在 CL 上侧")
        elif (seg < cl).all():
            _emit(i + 8, 2, f"W{wafers[i]}~W{wafers[i+8]} 连续 9 点在 CL 下侧")
    # ③ 连续 6 点单调
    for i in range(len(x) - 5):
        seg = x[i:i + 6]
        if all(np.diff(seg) > 0):
            _emit(i + 5, 3, f"W{wafers[i]}~W{wafers[i+5]} 连续 6 点递增")
        elif all(np.diff(seg) < 0):
            _emit(i + 5, 3, f"W{wafers[i]}~W{wafers[i+5]} 连续 6 点递减")
    # ④ 连续 14 点交替
    for i in range(len(x) - 13):
        d = np.sign(np.diff(x[i:i + 14]))
        if np.all(d[0::2] > 0) and np.all(d[1::2] < 0):
            _emit(i + 13, 4, f"W{wafers[i]}~W{wafers[i+13]} 连续 14 点上下交替")
        elif np.all(d[0::2] < 0) and np.all(d[1::2] > 0):
            _emit(i + 13, 4, f"W{wafers[i]}~W{wafers[i+13]} 连续 14 点下上交替")
    return alarms


def inject_wafer_drift(results: pd.DataFrame, param: str = "VOUT1",
                       wafers: tuple[int, int] = DRIFT_WAFERS,
                       total_shift: float = DRIFT_TOTAL_SHIFT) -> pd.DataFrame:
    """构造演示 lot：对 wafer 区间施加线性爬坡 lot 偏移（DEF_DRIFT 弱化版）。

    只改测量值列，不重判 bin（SPC 监控的是量测分布，不是 bin 结果）。
    """
    start, end = wafers
    df = results.copy()
    mask_span = end - start + 1
    for w in range(start, end + 1):
        delta = total_shift * (w - start + 1) / mask_span
        sel = df["wafer"] == w
        df.loc[sel, param] = df.loc[sel, param] + delta
    return df


def plot_imr(clean: pd.Series, injected: pd.Series, out_path: str | Path,
             param: str = "VOUT1") -> Path:
    """I-MR 控制图对比 PNG：左列干净 lot（应零报警），右列注入漂移（应报警）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for col, (title, means) in enumerate((("干净 lot", clean), ("注入漂移 lot", injected))):
        chart = imr_chart(means)
        alarms = western_electric_alarms(means, chart)
        w = chart["wafers"]
        ax = axes[0, col]
        ax.plot(w, means.to_numpy(), "b-o", markersize=4, label="wafer 均值")
        ax.axhline(chart["x_bar"], color="green", lw=1, label="CL")
        ax.axhline(chart["ucl"], color="red", ls="--", lw=1, label="UCL/LCL (±3σ)")
        ax.axhline(chart["lcl"], color="red", ls="--", lw=1)
        for a in alarms:
            if a["rule"] == 1:
                ax.axvline(a["wafer"], color="orange", ls=":", lw=1.5)
        ax.set_title(f"I 图（{title}）：报警 {len(alarms)} 次")
        ax.set_ylabel(f"{param} wafer 均值 (V)")
        ax.legend(fontsize=8, loc="best")
        ax = axes[1, col]
        mr = np.abs(np.diff(means.to_numpy()))
        ax.plot(w[1:], mr, "k-o", markersize=3)
        ax.axhline(chart["mr_bar"], color="green", lw=1, label="MR̄")
        ax.axhline(chart["mr_ucl"], color="red", ls="--", lw=1, label="UCL=3.267·MR̄")
        ax.set_title("MR 图（移动极差）")
        ax.set_xlabel("wafer 序号")
        ax.legend(fontsize=8)
    fig.suptitle(f"I-MR 控制图 — {param}（左：主 lot 零报警；右：W13~W19 注入 +21mV 爬坡）")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
