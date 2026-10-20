"""晶圆生成器数据口径：25 × 2000 = 50,000 die、坐标合法、良率落在 85%~97%。"""

import numpy as np

from dut import config, wafer


def test_total_die_count():
    dies, wafers = wafer.generate_lot()
    assert len(dies) == config.N_WAFERS * config.DIE_PER_WAFER == 50_000
    assert len(wafers) == config.N_WAFERS
    assert (wafers["n_dies"] == config.DIE_PER_WAFER).all()


def test_coordinates_unique_and_in_grid():
    dies, _ = wafer.generate_lot()
    for w, grp in dies.groupby("wafer"):
        assert len(grp) == grp[["x", "y"]].drop_duplicates().shape[0], \
            f"W{w} 存在重复坐标 die"
        assert grp["x"].abs().max() <= config.GRID_N // 2
        assert grp["y"].abs().max() <= config.GRID_N // 2
        r = np.hypot(grp["x"], grp["y"])
        # 圆形晶圆：最大半径与最小外接圆一致（允许 1.5 die 公差）
        assert r.max() - r.min() > 10, "半径分布异常"


def test_overall_yield_in_target_band():
    dies, wafers = wafer.generate_lot()
    y = dies["pass"].mean()
    assert 0.85 <= y <= 0.97, f"整体良率 {y:.2%} 不在 85%~97% 目标区间"
    # 单片良率有合理散布（簇缺陷导致片间差异）
    assert wafers["yield"].min() > 0.60
    assert wafers["yield"].max() <= 1.0


def test_defect_labels_valid_and_sparse():
    dies, wafers = wafer.generate_lot()
    valid = {"", *config.DEFECT_WEIGHTS}
    assert set(dies["defect"].unique()) <= valid
    frac = (dies["defect"] != "").mean()
    assert 0.03 <= frac <= 0.18, f"缺陷 die 比例 {frac:.2%} 异常"
    # 每片至少有一处缺陷注入
    assert (wafers["n_defect_dies"] > 0).all()


def test_spatial_modes_present():
    """空间模式存档：整批至少存在边缘环 die 与成簇 die。"""
    _, wafers = wafer.generate_lot()
    assert wafers["n_edge_dies"].min() > 0
    assert wafers["n_cluster_dies"].sum() > 0
    assert wafers["n_clusters"].sum() > 0
    # 边缘环覆盖比例合理（外 1~2 圈约占 12%~20%）
    edge_frac = wafers["n_edge_dies"].iloc[0] / config.DIE_PER_WAFER
    assert 0.10 <= edge_frac <= 0.22


def test_spatial_mode_statistics():
    """空间缺陷模式的统计验证：边缘 ×3、簇内 ×10 在 fail 率上必须体现出来。"""
    dies, wafers = wafer.generate_lot()
    rate = dies.groupby("region")["pass"].apply(lambda s: 1.0 - s.mean())
    inner = rate["INNER"]
    # 边缘环（非簇）fail 率约为内部 3 倍（随机涨落允许 2~4 倍窗口）
    assert 2.0 * inner < rate["EDGE"] < 4.5 * inner, \
        f"边缘环 fail 率 {rate['EDGE']:.2%} 未体现 ×3（内部 {inner:.2%}）"
    # 簇内 fail 率至少为内部 5 倍
    assert rate["CLUSTER"] > 5.0 * inner, \
        f"簇内 fail 率 {rate['CLUSTER']:.2%} 未体现 ×10（内部 {inner:.2%}）"
    # 三种模式共存：有簇片与无簇片都存在
    assert (wafers["n_clusters"] > 0).any()
    assert (wafers["n_clusters"] == 0).any()


def test_fail_kind_matches_defect_physics():
    """fail 分类与缺陷物理一致：OPEN→FUNC fail；OFFSET/LEAK→参数越限。"""
    dies, _ = wafer.generate_lot()
    open_dies = dies[dies["defect"] == "DEF_OPEN"]
    assert (open_dies["fail_kind"].isin(["FUNC", "BOTH"])).all()
    assert (open_dies["fail_kind"] == "FUNC").mean() > 0.99  # 参数尾部极少数 BOTH

    offset_leak = dies[dies["defect"].isin(["DEF_OFFSET", "DEF_LEAK"])]
    assert (offset_leak["fail_kind"].isin(["PARAM", "BOTH"])).all()
    assert (offset_leak["fail_kind"] == "PARAM").mean() > 0.99

    # 潜伏型缺陷（DRIFT/TIMING）绝大多数在 25°C CP 通过
    latent = dies[dies["defect"].isin(["DEF_DRIFT", "DEF_TIMING"])]
    assert latent["pass"].mean() > 0.90

    # PASS 的 die 不应带有非潜伏缺陷标签
    bad_labels = {"DEF_OFFSET", "DEF_LEAK", "DEF_OPEN"}
    assert not set(dies.loc[dies["pass"], "defect"].unique()) & bad_labels
