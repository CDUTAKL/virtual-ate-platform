"""晶圆/批次生成器（设计文档 §3.1）。

规模：25 片 wafer × 2000 die = 50,000 die（简历主数字）。
空间缺陷模式：
- 边缘环：最外 1~2 圈 die 缺陷概率 ×3（edge yield loss）
- 成簇缺陷：每片随机 0~3 个簇中心（半径 5~30 die），簇内缺陷概率 ×10
- 随机缺陷：基准缺陷密度 D0（dut/config.py）

每 die 输出：坐标 (x, y)、批次、缺陷标签（仅存档，测试程序不可见）、
12 个参数实测值（25°C CP 条件）、3 项功能测试结果、pass/fail。

缺陷与参数值的关系物理合理：
- DEF_OFFSET / DEF_LEAK 在 25°C CP 下直接越限（参数 fail）
- DEF_OPEN 功能 fail、参数正常
- DEF_DRIFT / DEF_TIMING 在 25°C 潜伏（分别需高温/高温重载才暴露），
  与文档"目标检出手段 = CP 多温 / Shmoo / FT"一致。

可复现：单个 numpy default_rng(seed) 顺序驱动全部随机量。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from . import pmu300
from .defects import Defect, choose_kind, make_defect


@dataclass
class WaferGeometry:
    """一片晶圆的 die 布局。"""

    x: np.ndarray          # 中心化坐标（int）
    y: np.ndarray
    r: np.ndarray          # 到圆心距离（die 单位）
    u: np.ndarray          # r / r_max ∈ [0, 1]
    edge: np.ndarray       # 边缘环掩码（最外 1~2 圈）
    in_cluster: np.ndarray # 落在任一缺陷簇内
    clusters: list[tuple[float, float, float]]  # (cx, cy, radius) 存档


def wafer_geometry(rng) -> WaferGeometry:
    """生成一片 2000-die 晶圆的布局与空间缺陷权重。

    布局：50×50 网格中取距圆心最近的 2000 个点（圆形晶圆）。
    成簇缺陷：0~3 个簇中心均匀落在晶圆内，半径 5~30 die。
    """
    n = config.GRID_N
    idx = np.arange(n)
    gx, gy = np.meshgrid(idx, idx)
    cx = cy = (n - 1) // 2   # 整数网格中心，保证坐标唯一
    r_all = np.hypot(gx - cx, gy - cy)
    flat = np.argsort(r_all, axis=None)[: config.DIE_PER_WAFER]
    xs = (gx.ravel()[flat] - cx).astype(int)
    ys = (gy.ravel()[flat] - cy).astype(int)
    r = np.hypot(xs, ys)
    u = r / r.max()

    # 边缘环：最外 1~2 圈
    edge = r >= (r.max() - config.EDGE_RING_WIDTH)

    # 成簇缺陷
    k = int(rng.integers(config.CLUSTER_MIN, config.CLUSTER_MAX + 1))
    in_cluster = np.zeros(config.DIE_PER_WAFER, dtype=bool)
    clusters: list[tuple[float, float, float]] = []
    for _ in range(k):
        rad = config.CLUSTER_R_MIN + (config.CLUSTER_R_MAX - config.CLUSTER_R_MIN) * rng.random()
        # 簇中心：极坐标均匀落在晶圆内
        theta = 2.0 * np.pi * rng.random()
        rr = r.max() * np.sqrt(rng.random())
        ccx, ccy = rr * np.cos(theta), rr * np.sin(theta)
        clusters.append((ccx, ccy, rad))
        in_cluster |= np.hypot(xs - ccx, ys - ccy) <= rad

    return WaferGeometry(xs, ys, r, u, edge, in_cluster, clusters)


def generate_lot(seed: int | None = None, d0: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成一个完整批次（默认 25 片 × 2000 die）。

    参数
    ----
    seed : None 时用 config.SEED；d0 : None 时用 config.D0（测试可传 0 生成无缺陷批）。

    返回
    ----
    dies : 每 die 一行（坐标/批次/缺陷标签/参数实测值/功能结果/pass）
    wafers : 每片一行（die 数、缺陷簇、边缘 die 数、良率等汇总）
    """
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    d0 = config.D0 if d0 is None else d0

    # 批次级：每个参数一次 lot 偏移（lot-to-lot，整批共享）
    lot_off = {name: rng.normal(0.0, s.sigma_lot) for name, s in pmu300.PARAM_SPECS.items()}

    die_frames: list[pd.DataFrame] = []
    wafer_rows: list[dict] = []

    for wafer_idx in range(1, config.N_WAFERS + 1):
        geo = wafer_geometry(rng)
        n = config.DIE_PER_WAFER

        # 缺陷概率：D0 × 边缘倍率 × 簇内倍率
        weight = np.ones(n)
        weight[geo.edge] *= config.EDGE_FACTOR
        weight[geo.in_cluster] *= config.CLUSTER_FACTOR
        p_defect = np.minimum(d0 * weight, 0.95)

        # 空间区域标签：EDGE / CLUSTER（簇内优先）/ INNER —— 布局元数据，存档用
        region = np.where(geo.in_cluster, "CLUSTER",
                          np.where(geo.edge, "EDGE", "INNER"))

        hit = rng.random(n) < p_defect
        defects: list[Defect | None] = [None] * n
        for i in np.nonzero(hit)[0]:
            defects[int(i)] = make_defect(choose_kind(rng), rng)

        # 参数实测值（25°C）：nominal + lot_offset + 径向梯度 + die_noise
        data: dict[str, np.ndarray] = {
            "lot": config.LOT_ID,
            "wafer": wafer_idx,
            "x": geo.x,
            "y": geo.y,
            "region": region,
            "defect": [d.kind if d is not None else "" for d in defects],
        }
        for name, spec in pmu300.PARAM_SPECS.items():
            grad_amp = rng.normal(0.0, spec.sigma_grad)          # 本片梯度幅度
            v = (spec.nominal + lot_off[name]
                 + grad_amp * geo.u
                 + pmu300.draw_die_noise(spec, n, rng))
            # 缺陷作用（25°C 常规条件）
            for i, d in enumerate(defects):
                if d is not None:
                    v[i] = pmu300.apply_defect_to_value(
                        name, v[i], d, config.CP_TEMP_C, config.CP_LOAD_A)
            data[name] = np.round(v, 6)

        # 功能测试（25°C）
        for fname in pmu300.FUNCTIONAL_ITEMS:
            data[fname] = [d is None or d.kind != "DEF_OPEN" for d in defects]

        df = pd.DataFrame(data)
        df["fail_kind"] = _classify(df)
        df["pass"] = df["fail_kind"] == "PASS"
        die_frames.append(df)

        wafer_rows.append({
            "wafer": wafer_idx,
            "n_dies": n,
            "n_clusters": len(geo.clusters),
            "n_edge_dies": int(geo.edge.sum()),
            "n_cluster_dies": int(geo.in_cluster.sum()),
            "n_defect_dies": int(hit.sum()),
            "n_fail": int((~df["pass"]).sum()),
            "yield": float(df["pass"].mean()),
        })

    return pd.concat(die_frames, ignore_index=True), pd.DataFrame(wafer_rows)


def _classify(df: pd.DataFrame) -> pd.Series:
    """按 25°C 测试结果分类：PASS / PARAM（参数越限）/ FUNC（功能 fail）/ BOTH。"""
    param_pass = np.ones(len(df), dtype=bool)
    for name, spec in pmu300.PARAM_SPECS.items():
        col = df[name].to_numpy(dtype=float)
        ok = np.ones(len(df), dtype=bool)
        if spec.spec_lo is not None:
            ok &= col >= spec.spec_lo
        if spec.spec_hi is not None:
            ok &= col <= spec.spec_hi
        param_pass &= ok
    func_pass = df[list(pmu300.FUNCTIONAL_ITEMS)].all(axis=1).to_numpy()
    kind = np.where(param_pass & func_pass, "PASS",
                    np.where(param_pass & ~func_pass, "FUNC",
                             np.where(~param_pass & func_pass, "PARAM", "BOTH")))
    return pd.Series(kind, index=df.index)
