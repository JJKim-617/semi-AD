"""요소 가중 — Fisher 의 독립 가정을 정상 데이터로 고친다. 단위 테스트.

설계는 `docs/experiments/candidate/ood_component_weighting.md` 에 **코드보다 먼저** 박았다.

핵심 성질 하나만 통과하면 이 arm 은 존재 이유가 있다:
**서로 닮은 요소는 가중이 깎이고 혼자 다른 것을 보는 요소는 안 깎인다.**
지금 채택된 융합이 정확히 그걸 못 해서, 둘 다 k2 밀도인 A 와 C 가 2표를 갖고
혼자 다른 것을 보는 선 필터 B 가 1표를 갖는다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a44_component_weighting import (  # noqa: E402
    fit_component_weights,
    optimal_weights,
    probit,
    weighted_score,
)


# --- probit ---------------------------------------------------------------------

def test_probit_inverts_the_normal_cdf():
    from scipy import stats
    u = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    assert np.allclose(probit(u), stats.norm.ppf(u), atol=1e-9)


def test_probit_is_monotone():
    u = np.linspace(1e-6, 1 - 1e-6, 500)
    assert np.all(np.diff(probit(u)) > 0)


def test_probit_stays_finite_at_the_boundaries():
    """백분위는 `ecdf_percentile` 이 만드는 값이라 0 이나 1 에 붙을 수 있다.
    무한대가 나오면 점수 전체가 오염된다."""
    assert np.all(np.isfinite(probit(np.array([0.0, 1.0, -0.5, 1.5]))))


# --- 가중이 상관을 실제로 고치는가 -------------------------------------------------

def test_identity_covariance_gives_equal_weights():
    w = optimal_weights(np.eye(3))
    assert np.allclose(w, w[0])


def test_a_redundant_pair_is_downweighted_and_the_lone_signal_is_not():
    """**이 arm 이 존재하는 이유.**

    요소 0 과 2 는 상관 0.8(둘 다 k2 밀도에 해당), 요소 1 은 무상관(선 필터에 해당).
    올바른 가중이라면 겹치는 둘이 깎이고 혼자인 하나는 안 깎여야 한다.
    """
    r = 0.8
    S = np.array([[1.0, 0.0, r], [0.0, 1.0, 0.0], [r, 0.0, 1.0]])
    w = optimal_weights(S)
    assert w[1] > w[0] and w[1] > w[2]
    assert w[0] == pytest.approx(w[2])


def test_downweighting_grows_with_the_correlation():
    def ratio(r):
        S = np.array([[1.0, 0.0, r], [0.0, 1.0, 0.0], [r, 0.0, 1.0]])
        w = optimal_weights(S)
        return w[0] / w[1]
    assert ratio(0.9) < ratio(0.5) < ratio(0.1)


def test_weights_sum_to_one():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 4))
    S = np.cov(a, rowvar=False)
    assert optimal_weights(S).sum() == pytest.approx(1.0)


def test_singular_covariance_does_not_explode():
    """두 요소가 **완전히** 같으면 공분산이 특이해진다. 터지면 안 된다."""
    S = np.array([[1.0, 1.0], [1.0, 1.0]])
    w = optimal_weights(S)
    assert np.all(np.isfinite(w)) and w.sum() == pytest.approx(1.0)


# --- 적합은 정상만 본다 -----------------------------------------------------------

def test_fit_refuses_labels_that_contain_defects():
    """**범위 A 가드.** 가중을 결함이 섞인 자료에서 적합하면 그 순간 one-class 가 아니다."""
    z = np.random.default_rng(0).normal(size=(50, 3))
    with pytest.raises(Exception):
        fit_component_weights(z, y=np.array([0] * 49 + [7]))


def test_fit_accepts_all_normal_labels_and_is_deterministic():
    z = np.random.default_rng(0).normal(size=(200, 3))
    y = np.zeros(200, dtype=np.int64)
    a = fit_component_weights(z, y=y)
    b = fit_component_weights(z, y=y)
    assert np.array_equal(a["weights"], b["weights"])
    assert a["corr"].shape == (3, 3)
    assert np.allclose(np.diag(a["corr"]), 1.0)


def test_fit_recovers_the_planted_correlation_structure():
    """심어 둔 상관을 되찾아야 한다 — 안 그러면 가중이 엉뚱한 것을 고친다."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=(20000, 2))
    z = np.column_stack([base[:, 0],
                         base[:, 1],
                         0.9 * base[:, 0] + np.sqrt(1 - 0.81) * rng.normal(size=20000)])
    r = fit_component_weights(z, y=np.zeros(20000, np.int64))
    assert r["corr"][0, 2] == pytest.approx(0.9, abs=0.02)
    assert abs(r["corr"][0, 1]) < 0.03
    assert r["weights"][1] > r["weights"][0]


# --- 점수 ------------------------------------------------------------------------

def test_weighted_score_is_the_linear_combination():
    z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
    w = np.array([0.2, 0.3, 0.5])
    assert np.allclose(weighted_score(z, w), np.array([1.0 * 0.2 + 2 * 0.3 + 3 * 0.5, 0.5]))


def test_equal_weights_reproduce_the_plain_sum_up_to_scale():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(100, 3))
    s = weighted_score(z, np.full(3, 1 / 3))
    assert np.allclose(np.argsort(s), np.argsort(z.sum(1)))
