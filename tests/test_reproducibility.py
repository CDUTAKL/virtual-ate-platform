"""生成器可复现性：同 seed 两次生成结果完全一致，不同 seed 不同。"""

import pandas as pd

from dut import wafer


def test_same_seed_identical():
    dies1, wafers1 = wafer.generate_lot(seed=123)
    dies2, wafers2 = wafer.generate_lot(seed=123)
    pd.testing.assert_frame_equal(dies1, dies2)
    pd.testing.assert_frame_equal(wafers1, wafers2)


def test_default_seed_identical():
    dies1, _ = wafer.generate_lot()
    dies2, _ = wafer.generate_lot()
    pd.testing.assert_frame_equal(dies1, dies2)


def test_different_seed_differs():
    dies1, _ = wafer.generate_lot(seed=1)
    dies2, _ = wafer.generate_lot(seed=2)
    assert not dies1.equals(dies2)
