"""위치 민감도가 누출을 예측하는가 (18차).

`leakage_mechanism` 이 남긴 좁혀진 물음: **왜 translate 만 판별력을 올리는가.**
그럴듯한 이야기는 "translate 가 모델에게서 절대 위치를 빼앗는다" 이고,
실제로 누출의 반지름 의존이 국소 3종에만 있고 translate 가 그것을 눕힌다.

**하지만 translate 로 학습한 모델이 이동에 둔감한 것은 동어반복이다.**
그래서 **translate 를 안 쓴 모델들 안에서** 위치 민감도가 누출을 예측하는지를 본다.
E18 백본 7개는 전부 무증강인데 누출이 1,246~3,939 로 3배 넘게 벌어진다 —
거기서 관계가 보이면 순환이 아니다.

여기 담긴 함수는 순수 계산이라 정답을 손으로 적을 수 있다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diag_shift_sensitivity import mean_total_variation  # noqa: E402


def test_identical_distributions_have_zero_distance():
    p = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    assert mean_total_variation(p, p) == pytest.approx(0.0)


def test_disjoint_point_masses_have_distance_one():
    """완전히 다른 확정 예측은 1 이다. 척도의 위쪽 끝을 박는다."""
    p = np.array([[1.0, 0.0, 0.0]])
    q = np.array([[0.0, 1.0, 0.0]])
    assert mean_total_variation(p, q) == pytest.approx(1.0)


def test_it_is_the_half_l1_not_the_l1():
    """총변동거리는 L1 의 절반이다. 두 배로 세면 척도가 [0,2] 가 된다."""
    p = np.array([[0.6, 0.4]])
    q = np.array([[0.4, 0.6]])
    assert mean_total_variation(p, q) == pytest.approx(0.2)


def test_it_averages_over_samples():
    p = np.array([[1.0, 0.0], [1.0, 0.0]])
    q = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert mean_total_variation(p, q) == pytest.approx(0.5)


def test_shapes_must_match():
    with pytest.raises(ValueError):
        mean_total_variation(np.ones((2, 3)) / 3, np.ones((3, 3)) / 3)


def test_rows_must_be_distributions():
    """로짓을 그대로 넘기는 실수를 조용히 통과시키지 않는다."""
    with pytest.raises(ValueError):
        mean_total_variation(np.array([[2.0, -1.0]]), np.array([[1.0, 0.0]]))
