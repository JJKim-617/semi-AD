"""BBSE 로 타겟 사전확률을 추정하고, 그것으로 선택 지표를 고친다.

a7 의 classify-and-count 는 분류기 자신의 편향을 물려받는다. 실측에서 Edge-Loc 을 1.93배,
Scratch 를 1.58배 과대추정했고, 그 결과 재가중 val 의 순위 상관이 현행 val 과 똑같은
+0.100 에 그쳤다. 같은 재가중을 실제 test 사전확률로 하면 +0.900 이 나온다.
즉 메커니즘은 맞고 추정기가 문제다.

BBSE(Lipton et al. 2018)는 val 에서 잰 혼동행렬로 그 편향을 보정한다.
    C[i,j] = P_val(pred=i, y=j),  mu[i] = P_test(pred=i),  C w = mu,  q = w * p_val
CC 는 C 가 단위행렬(분류기가 완벽)일 때의 특수한 경우다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a12_shift_estimation import (
    estimate_target_prior_bbse,
    rank_by_shifted_val,
    shifted_val_macro_f1,
    total_variation,
)


def _biased_classifier_data(source_prior, target_prior, err, n=200_000, seed=0):
    """오류율이 알려진 이진 분류기로 source/target 표본을 만든다.

    클래스 1 을 err 확률로 0 이라 부르고, 클래스 0 은 항상 맞힌다.
    이 비대칭 때문에 classify-and-count 는 클래스 1 을 반드시 과소추정한다.
    """
    rng = np.random.default_rng(seed)

    def sample(prior):
        y = rng.choice(2, size=n, p=prior)
        pred = y.copy()
        flip = (y == 1) & (rng.random(n) < err)
        pred[flip] = 0
        return y, pred

    return sample(source_prior), sample(target_prior)


class TestBBSEUsesNoTargetLabels:
    def test_signature_has_no_target_label_argument(self):
        import inspect
        params = set(inspect.signature(estimate_target_prior_bbse).parameters)
        assert "target_y" not in params and "test_y" not in params


class TestBBSERecoversTruePrior:
    def test_no_shift_returns_source_prior(self):
        """타겟이 소스와 같으면 소스 사전확률이 그대로 나와야 한다."""
        (sy, sp), (_, tp) = _biased_classifier_data([0.7, 0.3], [0.7, 0.3], err=0.4)
        q = estimate_target_prior_bbse(sp, sy, tp, num_classes=2)
        assert q == pytest.approx([0.7, 0.3], abs=0.01)

    def test_recovers_shifted_prior_a_biased_classifier_gets_wrong(self):
        """이것이 이 모듈의 존재 이유다. 편향된 분류기에서 BBSE 는 맞고 CC 는 틀려야 한다."""
        from a7_reweighted_offset import estimate_target_prior_cc

        source, target = [0.5, 0.5], [0.9, 0.1]
        (sy, sp), (_, tp) = _biased_classifier_data(source, target, err=0.4)

        q_bbse = estimate_target_prior_bbse(sp, sy, tp, num_classes=2)
        # CC 는 로짓을 받으므로 예측을 원-핫 로짓으로 바꿔 같은 입력을 준다.
        q_cc = estimate_target_prior_cc(np.eye(2)[tp], num_classes=2)

        assert total_variation(q_bbse, target) < 0.01
        assert total_variation(q_cc, target) > 0.02
        assert total_variation(q_bbse, target) < total_variation(q_cc, target)

    def test_returns_a_probability_vector(self):
        (sy, sp), (_, tp) = _biased_classifier_data([0.5, 0.5], [0.95, 0.05], err=0.5)
        q = estimate_target_prior_bbse(sp, sy, tp, num_classes=2)
        assert q.sum() == pytest.approx(1.0)
        assert (q >= 0).all()

    def test_clips_negative_solutions_to_a_valid_prior(self):
        """선형해가 음수를 낼 수 있다. 사전확률로 쓸 수 있는 형태로 나와야 한다."""
        rng = np.random.default_rng(3)
        val_y = rng.integers(0, 3, 3000)
        val_pred = rng.integers(0, 3, 3000)      # 무작위 예측이라 C 가 거의 특이행렬
        test_pred = np.zeros(3000, dtype=int)    # 극단적 타겟
        q = estimate_target_prior_bbse(val_pred, val_y, test_pred, num_classes=3)
        assert np.isfinite(q).all() and (q >= 0).all()
        assert q.sum() == pytest.approx(1.0)

    def test_class_absent_from_val_does_not_crash(self):
        val_y = np.array([0, 0, 1, 1])
        val_pred = np.array([0, 0, 1, 1])
        test_pred = np.array([0, 1, 2, 2])
        q = estimate_target_prior_bbse(val_pred, val_y, test_pred, num_classes=3)
        assert np.isfinite(q).all() and q.sum() == pytest.approx(1.0)


class TestTotalVariation:
    def test_identical_distributions_are_zero(self):
        assert total_variation([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)

    def test_disjoint_distributions_are_one(self):
        assert total_variation([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_is_half_the_l1_distance(self):
        assert total_variation([0.6, 0.4], [0.4, 0.6]) == pytest.approx(0.2)


class TestShiftedValScore:
    def test_target_prior_equal_to_val_prior_matches_plain_macro_f1(self):
        from a4_eval_wm811k_cls import evaluate
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 400)
        logits = rng.normal(size=(400, 3))
        p_val = np.bincount(y, minlength=3) / len(y)
        got = shifted_val_macro_f1(logits, y, p_val, num_classes=3)
        assert got == pytest.approx(evaluate(y, logits.argmax(1), 3)["macro_f1"])

    def test_downweighting_a_class_changes_the_score(self):
        rng = np.random.default_rng(1)
        y = np.where(rng.random(400) < 0.5, 0, 1)
        logits = rng.normal(size=(400, 2))
        a = shifted_val_macro_f1(logits, y, np.array([0.5, 0.5]), num_classes=2)
        b = shifted_val_macro_f1(logits, y, np.array([0.95, 0.05]), num_classes=2)
        assert a != pytest.approx(b)

    def test_penalises_false_positives_on_the_majority_class(self):
        """타겟에서 none 비중이 커지면, none 을 결함이라 부르는 모델이 더 나쁘게 나와야 한다.

        이것이 우리가 원하는 성질 전부다. val 에서는 두 모델이 비슷해 보이지만
        test 분포에서는 오탐이 많은 쪽이 확실히 나빠야 한다.
        """
        y = np.array([0] * 100 + [1] * 100)
        clean = np.zeros((200, 2)); clean[:100, 0] = 1.0; clean[100:, 1] = 1.0
        leaky = clean.copy()
        leaky[:20] = [0.0, 1.0]          # none 20 장을 결함이라 부른다

        val_prior = np.array([0.5, 0.5])
        test_prior = np.array([0.95, 0.05])

        gap_val = (shifted_val_macro_f1(clean, y, val_prior, 2)
                   - shifted_val_macro_f1(leaky, y, val_prior, 2))
        gap_test = (shifted_val_macro_f1(clean, y, test_prior, 2)
                    - shifted_val_macro_f1(leaky, y, test_prior, 2))
        assert gap_test > gap_val


class TestRanking:
    def test_ranks_runs_by_the_shifted_metric(self):
        y = np.array([0] * 100 + [1] * 100)
        clean = np.zeros((200, 2)); clean[:100, 0] = 1.0; clean[100:, 1] = 1.0
        leaky = clean.copy(); leaky[:30] = [0.0, 1.0]
        order = rank_by_shifted_val(
            {"leaky": (leaky, y), "clean": (clean, y)},
            target_prior=np.array([0.95, 0.05]), num_classes=2)
        assert [t for t, _ in order] == ["clean", "leaky"]

    def test_returns_scores_alongside_tags(self):
        y = np.array([0, 0, 1, 1])
        z = np.eye(2)[y].astype(float)
        order = rank_by_shifted_val({"a": (z, y)}, np.array([0.5, 0.5]), 2)
        assert order[0][0] == "a" and 0.0 <= order[0][1] <= 1.0
