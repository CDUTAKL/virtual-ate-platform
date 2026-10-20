"""guardband 限值计算（设计文档 §3.3：test_limit = spec_limit ∓ k·σ_meas）。

guardband 的意义（文档 §3.2）：测量本身有不确定度 σ_meas，若测试限=规格限，
测量误差会把规格边缘的好品判 fail（过杀）或把坏品判 pass（漏杀）。
测试限向规格内缩 k·σ_meas（k 可配，默认 3.0），在质量与良率间折中。
"""

import numpy as np


def guardband_limits(spec_lo: float | None, spec_hi: float | None,
                     sig_meas: float, k: float) -> tuple[float | None, float | None]:
    """由规格限与 σ_meas 计算测试限：下限内缩 +k·σ_meas，上限内缩 −k·σ_meas。

    单边规格只内缩相应一侧；k=0 时测试限=规格限（无 guardband）。
    """
    test_lo = spec_lo + k * sig_meas if spec_lo is not None else None
    test_hi = spec_hi - k * sig_meas if spec_hi is not None else None
    return test_lo, test_hi


def margin_sigma(values: np.ndarray, test_lo: float | None,
                 test_hi: float | None) -> float:
    """测试限相对工艺分布的余量（σ 数）：min(上限距, 下限距)/(3σ_pop)。

    余量越小 → 工艺漂移时越先顶到测试限 → 风险最高（文档 §3.3 guardband 报告）。
    """
    v = np.asarray(values, dtype=float)
    mu, sd = v.mean(), v.std()
    dists = []
    if test_hi is not None:
        dists.append(test_hi - mu)
    if test_lo is not None:
        dists.append(mu - test_lo)
    return float(min(dists) / (3.0 * sd))
