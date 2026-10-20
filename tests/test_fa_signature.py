"""失效签名：fail 群体与 parquet 一致、特征维度完整、空间/温度字段口径正确。"""

import numpy as np
import pytest

from fa import signature as sigmod


@pytest.fixture(scope="module")
def built():
    cp = sigmod.load_cp()
    lot = sigmod.load_archive()
    sig = sigmod.build_signatures(cp, lot)
    return cp, lot, sig


def test_fail_count_matches_parquet(built):
    """签名行数 = parquet 中 hard_bin≠1 的 fail die 数。"""
    cp, _lot, sig = built
    assert len(sig) == int((cp["hard_bin"] != 1).sum())
    assert set(sig["hard_bin"]) == {2, 3}


def test_signature_dimensions_complete(built):
    """特征向量维度完整：12 项偏离 + 聚合维 + 测试项集合 + 空间维。"""
    _cp, _lot, sig = built
    required = ["die_id", "wafer", "x", "y", "hard_bin", "soft_bin_name",
                "fail_temp_c", "u_radius", "in_defect_cluster",
                "failed_items", "n_failed_items", "n_param_failed",
                "func_fail_count", "func_failed_items",
                "vout_dev_max", "iq_dev_max", "func_fail_flag",
                "true_defect", "region"]
    for col in required:
        assert col in sig.columns, f"缺列 {col}"
    for n in sigmod.PARAM_NAMES:
        col = f"dev_{n}"
        assert col in sig.columns
        assert np.isfinite(sig[col]).all()
    X, names = sigmod.feature_matrix(sig)
    assert X.shape == (len(sig), len(sigmod.FEATURE_COLS))
    assert names == sigmod.FEATURE_COLS
    assert np.isfinite(X).all()


def test_failed_items_nonempty(built):
    """fail die 的 fail 测试项集合非空（CP 判 fail 必有 fail 项）。"""
    _cp, _lot, sig = built
    assert (sig["n_failed_items"] >= 1).all()
    assert sig["failed_items"].map(len).equals(sig["n_failed_items"])


def test_func_fail_encoding(built):
    """功能 fail 编码：DEF_OPEN 全功能 fail（3 项），其余 die 功能全 pass。

    物理红线：功能 fail 与参数 fail 的编码必须可分（func_fail_flag 强特征）。
    """
    _cp, _lot, sig = built
    assert set(sig["func_fail_count"].unique()).issubset({0, 3})
    open_mask = sig["true_defect"] == "DEF_OPEN"
    assert open_mask.sum() > 0
    assert (sig.loc[open_mask, "func_fail_count"] == 3).all()
    # DEF_OPEN die 参数偏离必须在正常工艺散布内（无大幅偏离；
    # 790 die × 12 参数的高斯尾部允许到 ~4σ，但绝不该接近缺陷级的截断值 8）
    dev_cols = [f"dev_{n}" for n in sigmod.PARAM_NAMES]
    assert sig.loc[open_mask, dev_cols].abs().max().max() < 4.0
    assert sig.loc[open_mask, dev_cols].abs().mean().mean() < 1.0
    # func_fail_flag 强特征只在功能 fail 时非零
    assert ((sig["func_fail_flag"] > 0) == (sig["func_fail_count"] > 0)).all()


def test_temp_dimension_omitted_constant(built):
    """CP 为 25 °C 单温：fail 温度维度省略（常数列留档，模块注释说明）。"""
    _cp, _lot, sig = built
    assert (sig["fail_temp_c"] == 25.0).all()


def test_spatial_fields_sane(built):
    _cp, _lot, sig = built
    assert sig["u_radius"].between(0, 1).all()
    assert sig["in_defect_cluster"].dtype == bool
    # 空间簇标志与存档 region 应大体吻合（验证用对照，不作特征）
    ct = sig.groupby("region")["in_defect_cluster"].mean()
    assert ct["CLUSTER"] > ct["INNER"], \
        "CLUSTER 区域的成片 fail 比例应高于片内（空间特征导出合理）"
