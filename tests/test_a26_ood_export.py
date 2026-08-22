"""잔차 맵 내보내기의 사례 선택 — 단위 테스트.

시각화는 눈으로 보는 것이라 회귀가 조용히 난다. 최소한 **무엇을 고르는지**는 박아 둔다.
'놓친 것' 을 고른다면서 잘 잡은 것을 그리면 문서가 거짓말을 하게 된다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a26_ood_residual_export import select_cases, subset_for_export  # noqa: E402


class TestSelectCases:
    def test_caught_are_the_highest_scoring_members_of_the_class(self):
        y = np.array([0, 0, 5, 5, 5, 5, 1])
        score = np.array([0.0, 0.0, 9.0, 1.0, 5.0, 2.0, 0.0])
        c = select_cases(score, y, cls=5, n=2)
        assert list(c["caught"]) == [2, 4]

    def test_missed_are_the_lowest_scoring_members_of_the_class(self):
        y = np.array([0, 0, 5, 5, 5, 5, 1])
        score = np.array([0.0, 0.0, 9.0, 1.0, 5.0, 2.0, 0.0])
        c = select_cases(score, y, cls=5, n=2)
        assert list(c["missed"]) == [3, 5]

    def test_false_alarms_are_the_highest_scoring_normals(self):
        y = np.array([0, 0, 0, 5])
        score = np.array([3.0, 1.0, 7.0, 0.0])
        c = select_cases(score, y, cls=5, n=2)
        assert list(c["false_alarm"]) == [2, 0]

    def test_typical_normal_sits_near_the_median_not_at_an_extreme(self):
        y = np.zeros(101, np.int64)
        y[100] = 5
        score = np.arange(101, dtype=np.float64)
        c = select_cases(score, y, cls=5, n=3)
        assert all(30 <= i <= 70 for i in c["typical_normal"])

    def test_asking_for_more_than_exist_returns_what_there_is(self):
        y = np.array([0, 5])
        c = select_cases(np.array([0.0, 1.0]), y, cls=5, n=10)
        assert len(c["caught"]) == 1
        assert len(c["missed"]) == 1


class TestSubsetForExport:
    def test_every_defect_is_kept(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 9, 500)
        idx = subset_for_export(y, n_normal=10, seed=0)
        assert set(np.flatnonzero(y != 0)).issubset(set(idx))

    def test_normals_are_subsampled_to_the_requested_count(self):
        y = np.zeros(500, np.int64)
        y[:20] = 3
        idx = subset_for_export(y, n_normal=50, seed=0)
        assert int((y[idx] == 0).sum()) == 50

    def test_selection_is_reproducible(self):
        y = np.zeros(300, np.int64)
        y[:10] = 2
        assert np.array_equal(subset_for_export(y, 30, 7), subset_for_export(y, 30, 7))

    def test_indices_are_sorted_so_npz_slicing_stays_cheap(self):
        y = np.zeros(300, np.int64)
        y[:10] = 2
        idx = subset_for_export(y, 30, 1)
        assert (np.diff(idx) > 0).all()
