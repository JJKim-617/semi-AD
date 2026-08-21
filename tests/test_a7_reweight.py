"""val 을 타겟 분포로 재가중해 오프셋을 튜닝하는 경로 테스트.

a6 에서 val 로 고른 오프셋이 test 를 악화시켰다(-0.0150). 이유는 val 이 train 분포
(none 67.4%)를 물려받는데 test 는 none 93.3% 라, val 에서 최적인 경계가 test 에서
결함 precision 을 붕괴시키기 때문이다. val 표본에 w_y = q_target(y)/p_val(y) 가중치를
주면 val 라벨만으로 "타겟 분포에서의 macro-F1" 을 근사할 수 있다.

타겟 사전확률은 모델이 라벨 없는 test 에 매긴 예측 빈도로 추정한다(classify-and-count).
test 라벨은 쓰지 않는다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a7_reweighted_offset import estimate_target_prior_cc, importance_weights


class TestTargetPrior:
    def test_sums_to_one(self):
        z = np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
        assert estimate_target_prior_cc(z, 2).sum() == pytest.approx(1.0)

    def test_counts_predicted_labels(self):
        z = np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        p = estimate_target_prior_cc(z, 2)
        assert p[0] == pytest.approx(0.75)

    def test_uses_no_labels(self):
        """서명에 라벨이 없다. 라벨을 쓰지 않는다는 것을 구조로 보장한다."""
        import inspect
        assert "y" not in inspect.signature(estimate_target_prior_cc).parameters

    def test_class_never_predicted_gets_positive_mass(self):
        """0 이면 뒤에서 나눗셈이 발산한다."""
        z = np.array([[5.0, 0.0], [5.0, 0.0]])
        assert estimate_target_prior_cc(z, 2)[1] > 0


class TestWeights:
    def test_identical_priors_give_uniform_weights(self):
        y = np.array([0, 0, 1, 1])
        w = importance_weights(y, np.array([0.5, 0.5]), num_classes=2)
        assert np.allclose(w, w[0])

    def test_upweights_class_more_common_in_target(self):
        y = np.array([0, 1])
        w = importance_weights(y, target_prior=np.array([0.9, 0.1]), num_classes=2)
        assert w[0] > w[1]

    def test_reweighted_class_mass_matches_target_prior(self):
        """가중치를 적용한 클래스별 질량 비율이 타겟 사전확률과 같아야 한다. 이것이 목적이다."""
        y = np.array([0] * 60 + [1] * 40)
        target = np.array([0.9, 0.1])
        w = importance_weights(y, target, num_classes=2)
        mass = np.array([w[y == c].sum() for c in (0, 1)])
        assert np.allclose(mass / mass.sum(), target, atol=1e-9)

    def test_weights_are_positive_and_finite(self):
        y = np.array([0, 1, 2])
        w = importance_weights(y, np.array([0.98, 0.01, 0.01]), num_classes=3)
        assert (w > 0).all() and np.isfinite(w).all()

    def test_absent_source_class_does_not_produce_inf(self):
        y = np.array([0, 0, 1])
        w = importance_weights(y, np.array([0.3, 0.3, 0.4]), num_classes=3)
        assert np.isfinite(w).all()


class TestWeightedEvaluation:
    """가중 macro-F1. 재가중한 val 위에서 오프셋을 최적화하려면 지표도 가중돼야 한다."""

    def test_uniform_weights_match_unweighted(self):
        from a4_eval_wm811k_cls import evaluate
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 50)
        pred = rng.integers(0, 3, 50)
        a = evaluate(y, pred, 3)
        b = evaluate(y, pred, 3, sample_weight=np.ones(50))
        assert b["macro_f1"] == pytest.approx(a["macro_f1"])

    def test_weights_change_the_metric(self):
        from a4_eval_wm811k_cls import evaluate
        y = np.array([0, 0, 1, 1])
        pred = np.array([0, 0, 1, 0])
        w = np.array([1.0, 1.0, 1.0, 10.0])
        assert evaluate(y, pred, 2, sample_weight=w)["macro_f1"] != pytest.approx(
            evaluate(y, pred, 2)["macro_f1"])

    def test_duplicating_a_sample_equals_doubling_its_weight(self):
        """가중치의 의미가 표본 복제와 같아야 한다."""
        from a4_eval_wm811k_cls import evaluate
        y = np.array([0, 1, 1])
        pred = np.array([0, 1, 0])
        dup = evaluate(np.array([0, 1, 1, 1]), np.array([0, 1, 0, 0]), 2)
        wt = evaluate(y, pred, 2, sample_weight=np.array([1.0, 1.0, 2.0]))
        assert wt["macro_f1"] == pytest.approx(dup["macro_f1"])


class TestWeightedOffsets:
    def test_optimizer_accepts_weights_and_still_improves(self):
        from a4_eval_wm811k_cls import evaluate
        from a6_perclass_offset import apply_offsets, optimize_offsets
        rng = np.random.default_rng(0)
        y = np.where(rng.random(300) < 0.9, 0, 1)
        z = np.zeros((300, 2)); z[:, 0] = 1.0
        z[y == 1, 1] = 0.85; z[y == 0, 1] = 0.2
        z += rng.normal(0, 0.05, z.shape)
        w = importance_weights(y, np.array([0.5, 0.5]), 2)
        base = evaluate(y, z.argmax(1), 2, sample_weight=w)["macro_f1"]
        off = optimize_offsets(z, y, 2, sample_weight=w)
        got = evaluate(y, apply_offsets(z, off).argmax(1), 2, sample_weight=w)["macro_f1"]
        assert got >= base

    def test_different_weights_give_different_offsets(self):
        """재가중이 실제로 다른 경계를 만들어야 이 접근에 의미가 있다."""
        from a6_perclass_offset import optimize_offsets
        rng = np.random.default_rng(1)
        y = np.where(rng.random(300) < 0.5, 0, 1)
        z = rng.normal(size=(300, 2)); z[y == 1, 1] += 0.4
        a = optimize_offsets(z, y, 2, sample_weight=importance_weights(y, np.array([0.95, 0.05]), 2))
        b = optimize_offsets(z, y, 2, sample_weight=importance_weights(y, np.array([0.05, 0.95]), 2))
        assert not np.array_equal(a, b)
