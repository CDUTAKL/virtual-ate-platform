"""失效签名提取（设计文档 §3.7：对每颗 fail die 构造特征向量）。

特征向量（每颗 hard_bin≠1 的 fail die 一行）：
- fail 的测试项名集合 failed_items（12 参数项 + 3 功能项中 _pass==False 者）
  及计数 n_failed_items / n_param_failed / func_fail_count
- 参数偏离方向与幅度 dev_<name>：相对良品参考分布 (μ_good, σ_good) 的 z 分数，
  截断到 ±DEV_CLIP（缺陷 die 偏离可达数百 σ，截断防止距离被单一特征支配；
  方向信息保留在符号里：正=偏高/负=偏低）
- fail 温度点：CP 为 25 °C 单温（config.CP_TEMP_C），无温度维度可提取——
  按任务口径省略该维，仅以常数列 fail_temp_c=25.0 留档并在此注释说明
- 坐标 wafer/x/y 与归一化半径 u_radius（r/r_max，捕捉边缘环效应）
- 是否处于缺陷簇内 in_defect_cluster：由 wafer map 上 fail die 的局部空间
  密度导出（同片 r ≤ CLUSTER_RADIUS 内 fail 邻居 ≥ CLUSTER_MIN_NEIGHBORS 判定
  "成片"），不从存档标签取——真实 FA 里工程师拿到的只有测试数据

聚类特征矩阵 feature_matrix()（signature → 数值矩阵）的设计考虑
（对应"特征工程优先于降低目标"的硬约束）：
1. dev 保留 12 个测试项级维度（哪一项 fail 是最强的失效模式信号）；
2. 追加 vout_dev_max / iq_dev_max 聚合维：VOUT1/2/3 任一偏离 ↔ DEF_OFFSET
   机制，IQ1/2 任一偏离 ↔ DEF_LEAK 机制，帮助 K=5 预算下的机制级归并；
3. func_fail_flag × FUNC_FEATURE_WEIGHT 强特征：DEF_OPEN 的 die 参数全部正常、
   仅功能 fail（物理红线：必须与参数失效类分开），该维给予与最大参数偏离
   同量级的权重，确保 OPEN 类不被并进任何参数失效簇。

输入（只读）：outputs/cp_results.parquet（测试数据，无缺陷标签）、
outputs/lot_dies.csv.gz（真实缺陷标签，仅混淆矩阵对照验证用，不进特征）。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from dut import config
from dut.pmu300 import PARAM_SPECS, FUNCTIONAL_ITEMS

PARAM_NAMES = list(PARAM_SPECS.keys())                 # 12 参数项
FUNC_NAMES = list(FUNCTIONAL_ITEMS)                    # 3 功能项
PASS_COLS = {n: f"{n}_pass" for n in PARAM_NAMES}
FUNC_PASS_COLS = {n: f"{n}_pass" for n in FUNC_NAMES}

#: 偏离 z 分数截断（方向保留在符号，幅度只保留到 ±8σ）
DEV_CLIP = 8.0
#: "成片 fail"判定：同片 r 范围内 fail 邻居数阈值（空间簇特征）
CLUSTER_RADIUS = 4.0
CLUSTER_MIN_NEIGHBORS = 3
#: 功能 fail 强特征权重（与最大截断偏离同量级，保证 DEF_OPEN 独立成簇）
FUNC_FEATURE_WEIGHT = DEV_CLIP

#: KMeans 特征列（相对签名 DataFrame 的列名）
DEV_COLS = [f"dev_{n}" for n in PARAM_NAMES]
FEATURE_COLS = (DEV_COLS
                + ["vout_dev_max", "iq_dev_max",
                   "func_fail_flag", "n_param_failed"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CP = PROJECT_ROOT / "outputs" / "cp_results.parquet"
DEFAULT_LOT = PROJECT_ROOT / "outputs" / "lot_dies.csv.gz"


def load_cp(path: Path = DEFAULT_CP) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_archive(path: Path = DEFAULT_LOT) -> pd.DataFrame:
    """真实缺陷标签存档（仅对照验证用，严禁进入特征）。"""
    return pd.read_csv(path)


def fail_dies(cp: pd.DataFrame) -> pd.DataFrame:
    """CP fail 群体：hard_bin 1=good / 2=参数 fail / 3=功能 fail。

    索引重置为 0..n-1，保证后续按位置构造的签名列对齐。
    """
    return cp[cp["hard_bin"] != 1].copy().reset_index(drop=True)


def good_reference(cp: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """良品（hard_bin=1）参考分布：参数偏离 z 分数的 (μ, σ) 基准。"""
    good = cp[cp["hard_bin"] == 1]
    return {n: (float(good[n].mean()), float(good[n].std())) for n in PARAM_NAMES}


def spatial_cluster_flag(fails: pd.DataFrame, all_cp: pd.DataFrame) -> np.ndarray:
    """由 wafer map fail 密度导出"是否处于缺陷簇内"（逐片 cKDTree 邻域计数）。

    与存档 region 列（生成器布局元数据）的区别：本特征完全从测试数据的
    空间分布算出（真实 FA 的唯一信息源）；存档 region 只在报告里做对照验证。
    """
    # 每片归一化半径分母：该片全部 die 的最大极径
    coords_all = all_cp[["wafer", "x", "y"]].copy()
    coords_all["r"] = np.hypot(coords_all["x"], coords_all["y"])
    rmax = coords_all.groupby("wafer")["r"].max()
    flag = np.zeros(len(fails), dtype=bool)
    u = np.zeros(len(fails), dtype=float)
    wafer_arr = fails["wafer"].to_numpy()
    for w, grp in fails.groupby("wafer"):
        coords = grp[["x", "y"]].to_numpy(dtype=float)
        tree = cKDTree(coords)
        neighbors = tree.query_ball_point(coords, r=CLUSTER_RADIUS)
        p = np.nonzero(wafer_arr == w)[0]
        flag[p] = [len(nb) - 1 >= CLUSTER_MIN_NEIGHBORS for nb in neighbors]
        u[p] = np.hypot(coords[:, 0], coords[:, 1]) / rmax[w]
    return flag, u


def build_signatures(cp: pd.DataFrame,
                     lot: pd.DataFrame | None = None) -> pd.DataFrame:
    """构造 fail die 的失效签名表（一行一颗 die）。

    lot 仅用于合并真实缺陷标签列 true_defect（验证用，可省略）。
    """
    fails = fail_dies(cp)
    ref = good_reference(cp)
    flag, u = spatial_cluster_flag(fails, cp)

    sig = pd.DataFrame({
        "die_id": fails["die_id"].to_numpy(),
        "wafer": fails["wafer"].to_numpy(),
        "x": fails["x"].to_numpy(),
        "y": fails["y"].to_numpy(),
        "hard_bin": fails["hard_bin"].to_numpy(),
        "soft_bin_name": fails["soft_bin_name"].to_numpy(),
        # CP 单温：温度维度省略（见模块 docstring），常数列留档
        "fail_temp_c": float(config.CP_TEMP_C),
        "u_radius": u,
        "in_defect_cluster": flag,
    })

    # fail 测试项集合（参数 + 功能）
    param_fail_cols = [PASS_COLS[n] for n in PARAM_NAMES]
    func_fail_cols = [FUNC_PASS_COLS[n] for n in FUNC_NAMES]
    sig["n_param_failed"] = (
        len(PARAM_NAMES) - fails[param_fail_cols].sum(axis=1)).clip(upper=3)
    sig["func_fail_count"] = len(FUNC_NAMES) - fails[func_fail_cols].sum(axis=1)

    p_fail = ~fails[param_fail_cols].to_numpy(dtype=bool)   # (n, 12)
    f_fail = ~fails[func_fail_cols].to_numpy(dtype=bool)    # (n, 3)
    pnames = np.array(PARAM_NAMES)
    fnames = np.array(FUNC_NAMES)
    sig["failed_items"] = [
        tuple(pnames[m].tolist()) + tuple(fnames[m2].tolist())
        for m, m2 in zip(p_fail, f_fail)]
    sig["n_failed_items"] = sig["failed_items"].map(len)
    sig["func_failed_items"] = [tuple(fnames[m].tolist()) for m in f_fail]

    # 参数偏离方向与幅度（z 分数，截断 ±DEV_CLIP；符号=偏离方向）
    for n in PARAM_NAMES:
        mu, sd = ref[n]
        sig[f"dev_{n}"] = ((fails[n].to_numpy(dtype=float) - mu) / sd).clip(
            -DEV_CLIP, DEV_CLIP)
    sig["vout_dev_max"] = sig[[f"dev_VOUT{i}" for i in (1, 2, 3)]].abs().max(axis=1)
    sig["iq_dev_max"] = sig[["dev_IQ1", "dev_IQ2"]].abs().max(axis=1)
    sig["func_fail_flag"] = (sig["func_fail_count"] > 0).astype(float) \
        * FUNC_FEATURE_WEIGHT

    if lot is not None:
        keys = ["wafer", "x", "y"]
        assert not lot.duplicated(keys).any(), "存档坐标键不唯一"
        sig = sig.merge(lot[keys + ["defect", "region"]],
                        on=keys, how="left", validate="1:1")
        sig = sig.rename(columns={"defect": "true_defect"})
    return sig.reset_index(drop=True)


def feature_matrix(sig: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """签名 → KMeans 数值特征矩阵，返回 (X, 特征名列表)。"""
    return sig[FEATURE_COLS].to_numpy(dtype=float), list(FEATURE_COLS)
