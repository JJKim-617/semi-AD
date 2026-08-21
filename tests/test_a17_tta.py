"""추론 시 증강. 학습 때 쓴 대칭을 추론에서도 쓴다.

dihedral 증강은 이 프로젝트 최대 이득이었다(+0.101). 다른 모든 기법을 합친 것보다 크다.
웨이퍼의 회전, 반사 대칭이 실제 구조라는 뜻이고, 그렇다면 추론에서도 쓸 수 있다.
한 웨이퍼의 변환본들을 각각 모델에 넣고 확률을 평균한다. 라벨은 이 변환에 불변이라
역변환이 필요 없다. **학습이 전혀 없다.**

극좌표에는 다른 변환을 써야 한다. 극좌표 이미지에 rot90 을 걸면 반지름 축과 각도 축이
섞여 의미가 깨진다. 극좌표에서 회전은 열 방향 순환이동, 반사는 열 방향 뒤집기다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a17_tta import angular_variants, average_probs, dihedral_variants


def sample(n=5, h=8, w=8, seed=0):
    return np.random.default_rng(seed).integers(0, 3, (n, h, w), dtype=np.uint8)


class TestDihedralVariants:
    def test_yields_the_eight_group_elements(self):
        assert len(list(dihedral_variants(sample()))) == 8

    def test_the_identity_is_among_them(self):
        x = sample()
        assert any(np.array_equal(v, x) for v in dihedral_variants(x))

    def test_all_eight_are_distinct_for_a_generic_input(self):
        seen = {v.tobytes() for v in dihedral_variants(sample(n=1, h=8, w=8, seed=3))}
        assert len(seen) == 8

    def test_preserves_shape_and_dtype(self):
        x = sample()
        for v in dihedral_variants(x):
            assert v.shape == x.shape and v.dtype == x.dtype

    def test_invents_no_new_cell_values(self):
        """색인 재배열이라 값 집합이 그대로여야 한다. 보간이면 이 시험이 깨진다."""
        x = sample()
        for v in dihedral_variants(x):
            assert set(np.unique(v)).issubset(set(np.unique(x)))

    def test_rotations_only_yields_four(self):
        assert len(list(dihedral_variants(sample(), flips=False))) == 4


class TestAngularVariants:
    def test_yields_the_requested_count(self):
        assert len(list(angular_variants(sample(), n=4))) == 4

    def test_shifts_along_the_angle_axis_only(self):
        """행은 반지름이다. 건드리면 안 된다."""
        x = sample(n=1, h=8, w=8, seed=1)
        for v in angular_variants(x, n=4):
            assert np.array_equal(np.sort(v, axis=2), np.sort(x, axis=2))

    def test_the_identity_is_among_them(self):
        x = sample()
        assert any(np.array_equal(v, x) for v in angular_variants(x, n=4))

    def test_uses_circular_shift_not_truncation(self):
        x = np.arange(8, dtype=np.uint8).reshape(1, 1, 8)
        got = list(angular_variants(x, n=8))
        assert any(np.array_equal(v, np.roll(x, 1, axis=2)) for v in got)

    def test_flips_the_angle_axis_when_asked(self):
        x = sample(n=1, h=4, w=8, seed=2)
        got = list(angular_variants(x, n=4, flips=True))
        assert len(got) == 8
        assert any(np.array_equal(v, x[:, :, ::-1]) for v in got)


class TestAverageProbs:
    def test_averaging_one_set_returns_it_unchanged(self):
        p = np.array([[0.2, 0.8], [0.6, 0.4]])
        assert np.allclose(average_probs([p]), p)

    def test_averages_elementwise(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        assert np.allclose(average_probs([a, b]), [[0.5, 0.5]])

    def test_the_result_is_still_a_distribution(self):
        rng = np.random.default_rng(0)
        parts = []
        for _ in range(3):
            z = rng.random((10, 4))
            parts.append(z / z.sum(1, keepdims=True))
        out = average_probs(parts)
        assert np.allclose(out.sum(1), 1.0)

    def test_rejects_an_empty_list(self):
        with pytest.raises(ValueError):
            average_probs([])
