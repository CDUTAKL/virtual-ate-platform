"""失效聚类：混淆矩阵对角线 ≥90%、轮廓系数合理、DEF_OPEN 分离（物理红线）、可复现。"""

import json

import numpy as np
import pytest

from fa import cluster as fa_cluster
from fa import signature as sigmod
from dut import config


@pytest.fixture(scope="module")
def fa(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("fa")
    return fa_cluster.run_fa(out_dir=out_dir)


def test_diagonal_hit_rate_at_least_90pct(fa):
    """聚类 vs 真实标签：对角线命中率 ≥ 90%（文档 §4 目标）。"""
    rate = fa["report"]["diagonal_hit_rate"]
    assert rate >= 0.90, f"对角线命中率 {rate:.4f} 未达标（先查特征工程）"


def test_silhouette_reasonable(fa):
    """K=5 时轮廓系数应 > 0.25（>0.5 为结构清晰）。"""
    assert fa["silhouette"] > 0.25


def test_five_nonempty_clusters(fa):
    prof = fa["profile"]
    assert len(prof) == 5
    assert (prof["size"] > 0).all()
    # 每个 cluster 有明确多数缺陷（纯度 > 50% 才算一个"模式"）
    assert (prof["purity"] > 0.5).all()


def test_open_cluster_separated(fa):
    """物理红线：DEF_OPEN（功能 fail、参数正常）必须与参数失效类分开。"""
    open_mask = fa["signature"]["true_defect"] == "DEF_OPEN"
    labels = fa["labels"][open_mask.to_numpy()]
    # 全部 DEF_OPEN die 落在单一 cluster
    assert len(set(labels.tolist())) == 1
    c = labels[0]
    prof = fa["profile"].set_index("cluster").loc[c]
    assert prof["majority_kind"] == "DEF_OPEN"
    # 簇内功能 fail 均值 ≈ 3（混有少量过杀 die 时略低，但必须远高于参数失效簇）
    assert prof["func_fail_mean"] > 2.5
    # 该簇几乎不含参数大幅偏离的成员（参数正常的强特征）
    assert "mean_dev_by_param" in prof and not prof["mean_dev_by_param"], \
        "DEF_OPEN 簇不应有显著的参数偏离证据"


def test_param_defect_recall_perfect(fa):
    """DEF_OFFSET / DEF_LEAK 两类参数失效应被完美召回。"""
    recall = fa["report"]["per_kind_recall"]
    assert recall["DEF_OFFSET"] >= 0.99
    assert recall["DEF_LEAK"] >= 0.99


def test_latent_defects_documented(fa):
    """DEF_DRIFT / DEF_TIMING 25 °C 潜伏：CP 检出率应极低（物理口径）。"""
    latent = fa["report"]["latent_defects"]
    assert latent["DEF_DRIFT"]["cp_detection_rate"] < 0.05
    assert latent["DEF_TIMING"]["cp_detection_rate"] < 0.05


def test_report_files_written(fa):
    """JSON / Markdown / CSV 产物齐全且与运行结果一致。"""
    out = fa["report"]
    assert out["n_fail"] == len(fa["signature"])
    assert 0 < out["yield"] < 1
    out_dir = fa["out_dir"]
    report = json.loads((out_dir / "fa_report.json").read_text())
    assert report["diagonal_hit_rate"] == out["diagonal_hit_rate"]
    assert len(report["clusters"]) == 5
    assert set(map(int, report["confusion_matrix"])) == set(range(5))
    md = (out_dir / "fa_report.md").read_text()
    for section in ("现象", "数据", "聚类", "根因", "结论", "混淆矩阵"):
        assert section in md
    import pandas as pd
    df = pd.read_csv(out_dir / "signatures.csv")
    assert len(df) == out["n_fail"]
    assert (out_dir / "confusion_matrix.csv").exists()


def test_report_file_contents(tmp_path):
    fa_cluster.run_fa(out_dir=tmp_path)
    report = json.loads((tmp_path / "fa_report.json").read_text())
    assert report["diagonal_hit_rate"] >= 0.90
    assert len(report["clusters"]) == 5
    assert set(map(int, report["confusion_matrix"])) == set(range(5))
    md = (tmp_path / "fa_report.md").read_text()
    assert "现象" in md and "结论" in md and "混淆矩阵" in md
    sig_csv = tmp_path / "signatures.csv"
    assert sig_csv.exists()
    import pandas as pd
    df = pd.read_csv(sig_csv)
    assert len(df) == report["n_fail"]
    assert (tmp_path / "confusion_matrix.csv").exists()


def test_reproducible_labels():
    """固定种子：两次独立聚类标签完全一致。"""
    cp = sigmod.load_cp()
    sig = sigmod.build_signatures(cp)
    X, _ = sigmod.feature_matrix(sig)
    km1, sil1 = fa_cluster.cluster_fail_dies(X, seed=config.SEED)
    km2, sil2 = fa_cluster.cluster_fail_dies(X, seed=config.SEED)
    np.testing.assert_array_equal(km1.labels_, km2.labels_)
    assert sil1 == sil2
