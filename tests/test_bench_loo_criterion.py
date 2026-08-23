"""LOO 판정선을 잡음 구조에서 유도하는 도구 (18차 마무리).

## 왜

E20(밀도), E21(선 필터), E22(백본 x translate)가 전부 **군 평균 LOO >= +0.003** 에서
기각됐다. 그런데 18차가 **그 선이 지금 풀에서 도달 불가능**함을 쟀다 —
구성원 하나의 평균 기여가 +0.0002 이고 개별 LOO 는 seed 잡음이 지배한다.

**자가 고장 났으면 그 자로 잰 판정이 무엇을 뜻하는지 알 수 없다.**
그래서 **먼저 잡음 분포를 재고 그 다음에 선을 정한다.** 순서를 지킨다.

## 어떻게

판정은 **군 평균 LOO** 를 쓴다. 귀무가설은 *"이 군의 구성원이 풀의 나머지와
교환 가능하다"* 이다. 그러면 **군 평균 LOO 의 귀무분포는
풀의 구성원 LOO 값에서 g 개를 비복원 추출한 평균의 분포**다.

구성원별 LOO 는 그 구성원이 어느 군에 속하는지와 무관하게 계산되므로
(풀의 조합에서만 나온다) **치환은 라벨만 바꾸면 되고 LOO 를 다시 계산할 필요가 없다.**
그래서 이 유도는 학습도, 로짓 재계산도 필요 없다.

아래는 전부 순수 계산이라 **정답을 손으로 적을 수 있다.**
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench_loo_criterion import (  # noqa: E402
    null_sd_of_group_mean,
    permutation_p,
    required_threshold,
)


# --- 귀무분포의 표준편차 -----------------------------------------------------

def test_null_sd_matches_simulation():
    """해석식이 실제 비복원 추출과 맞는지 확인한다. 이 자의 근간이다."""
    rng = np.random.default_rng(0)
    loo = rng.normal(0, 0.002, size=11)
    g = 3
    sim = np.array([rng.choice(loo, g, replace=False).mean() for _ in range(20000)])
    assert null_sd_of_group_mean(loo, g) == pytest.approx(sim.std(ddof=1), rel=0.05)


def test_null_sd_shrinks_as_the_group_grows():
    """군이 커지면 평균의 흔들림이 준다. 판정선도 같이 내려가야 한다."""
    loo = np.array([0.004, -0.002, 0.001, -0.003, 0.002, 0.000, -0.001, 0.003])
    assert null_sd_of_group_mean(loo, 4) < null_sd_of_group_mean(loo, 2)


def test_null_sd_is_zero_when_the_group_is_the_whole_pool():
    """풀 전체를 군으로 잡으면 뽑을 것이 없다 — 흔들림이 0 이고 판정이 불가능하다."""
    loo = np.array([0.01, -0.01, 0.02, -0.02])
    assert null_sd_of_group_mean(loo, 4) == pytest.approx(0.0)


def test_null_sd_needs_a_group_smaller_than_the_pool():
    with pytest.raises(ValueError):
        null_sd_of_group_mean(np.zeros(4), 5)


# --- 판정선 -----------------------------------------------------------------

def test_threshold_is_positive_and_scales_with_spread():
    """구성원 LOO 가 흩어질수록 넘어야 할 선이 높아진다."""
    tight = np.array([0.0005, -0.0005, 0.0003, -0.0003, 0.0002, -0.0002])
    wide = tight * 10
    assert 0 < required_threshold(tight, 3) < required_threshold(wide, 3)


def test_threshold_falls_as_the_group_grows():
    """3 seed 보다 9 seed 가 통과하기 쉬워야 한다 — 그것이 n 을 늘리는 이유다."""
    loo = np.array([0.004, -0.002, 0.001, -0.003, 0.002, 0.000,
                    -0.001, 0.003, -0.004, 0.002, -0.002, 0.001])
    assert required_threshold(loo, 6) < required_threshold(loo, 2)


def test_threshold_uses_a_one_sided_test():
    """기여는 방향이 있다. 양쪽 검정을 쓰면 선이 불필요하게 높아진다."""
    loo = np.random.default_rng(1).normal(0, 0.002, 12)
    one = required_threshold(loo, 3, alpha=0.05)
    two = required_threshold(loo, 3, alpha=0.025)
    assert one < two


# --- 치환 p 값 ---------------------------------------------------------------

def test_permutation_p_is_small_for_an_extreme_group():
    """군이 풀에서 가장 높은 값들이면 p 가 작아야 한다."""
    loo = np.array([-0.003, -0.002, -0.001, 0.000, 0.001, 0.002, 0.004, 0.005])
    p = permutation_p(loo, [6, 7])          # 가장 큰 둘
    assert p < 0.05


def test_permutation_p_is_near_half_for_a_typical_group():
    loo = np.array([-0.003, -0.002, -0.001, 0.000, 0.001, 0.002, 0.003, 0.004])
    p = permutation_p(loo, [3, 4])          # 한가운데
    assert 0.3 < p < 0.7


def test_permutation_p_rejects_a_group_index_out_of_range():
    with pytest.raises(ValueError):
        permutation_p(np.zeros(4), [0, 9])


# --- 선과 p 값이 동치여야 한다 (이산 분포에서 실제로 어긋났다) ---------------

def test_threshold_and_p_value_agree():
    """`obs >= 선` 과 `p <= alpha` 가 **동치**여야 한다.

    보간 분위수를 쓰면 C(N,g) 가 작을 때 어긋난다. E22(b) s2 집합에서
    관측 0.0020 이 보간 선 0.0017 을 넘는데 p 는 0.057 이었다 — 실제로 겪은 버그다.
    """
    rng = np.random.default_rng(3)
    for n, g in [(7, 3), (11, 3), (13, 9), (5, 3), (9, 4)]:
        loo = rng.normal(0, 0.002, n)
        thr = required_threshold(loo, g, alpha=0.05)
        for _ in range(30):
            idx = list(rng.choice(n, g, replace=False))
            obs = loo[idx].mean()
            p = permutation_p(loo, idx)
            assert (obs >= thr) == (p <= 0.05 + 1e-12), (n, g, obs, thr, p)


def test_small_pool_has_no_resolution():
    """C(N,g) 가 작으면 alpha 를 만들 해상도가 없다는 것을 드러낸다."""
    from bench_loo_criterion import null_resolution
    total, pmin = null_resolution(7, 3)
    assert total == 35 and pmin == pytest.approx(1 / 35)
    assert pmin < 0.05          # 1등이면 p=0.029 로 통과는 가능하다
    total2, pmin2 = null_resolution(5, 3)
    assert total2 == 10 and pmin2 == pytest.approx(0.1)
    assert pmin2 > 0.05         # **어떤 결과도 p<=0.05 를 못 만든다**
