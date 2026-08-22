"""one-class 이상탐지 지표와 자명한 스칼라 채점기.

9-class 에서는 macro-F1 이 주 지표였지만 one-class 는 다르다. test 가 93.3% 정상이라
**AUROC 는 유병률에 불변이지만 절대 헛경보 부담을 반영하지 못한다** — 정상 110,701장에서
FPR 1% 도 1,107장이다. 그래서 주 지표를 AUPR 로 두고 FPR@95TPR 을 함께 본다.

E0 에서 학습 없이 불량 다이를 세기만 해도 AUROC 0.8167 이 나왔다. 이것이 바닥이고,
9-class 때 "전부 none 분류기 accuracy 0.9334" 를 테스트로 박았던 것처럼 여기서도 고정한다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a21_ood_metrics import aupr, auroc, evaluate_ood, fpr_at_tpr
from a21_ood_metrics import fail_count, fail_ratio


class TestAUROC:
    def test_perfect_separation_is_one(self):
        s = np.array([0.1, 0.2, 0.8, 0.9]); l = np.array([0, 0, 1, 1])
        assert auroc(s, l) == pytest.approx(1.0)

    def test_reversed_separation_is_zero(self):
        s = np.array([0.9, 0.8, 0.2, 0.1]); l = np.array([0, 0, 1, 1])
        assert auroc(s, l) == pytest.approx(0.0)

    def test_all_tied_scores_give_half(self):
        """전부 같은 점수면 정보가 없으므로 0.5 여야 한다. 동점 처리가 틀리면 여기서 깨진다."""
        s = np.ones(10); l = np.array([0, 1] * 5)
        assert auroc(s, l) == pytest.approx(0.5)

    def test_partial_ties_use_average_rank(self):
        s = np.array([1.0, 1.0, 2.0]); l = np.array([0, 1, 1])
        # 양성 둘의 평균 랭크 = (1.5 + 3)/2, 음성 하나
        assert auroc(s, l) == pytest.approx(0.75)

    def test_is_invariant_to_monotone_rescaling(self):
        rng = np.random.default_rng(0)
        s = rng.random(200); l = (rng.random(200) < 0.3).astype(int)
        assert auroc(s, l) == pytest.approx(auroc(np.exp(s * 3), l))


class TestAUPR:
    def test_perfect_separation_is_one(self):
        s = np.array([0.1, 0.2, 0.8, 0.9]); l = np.array([0, 0, 1, 1])
        assert aupr(s, l) == pytest.approx(1.0)

    def test_uninformative_score_approaches_prevalence(self):
        """무작위 점수의 AUPR 기준선은 유병률이다. 이것이 AUPR 을 주 지표로 쓰는 이유다."""
        rng = np.random.default_rng(1)
        n, prev = 20000, 0.0666
        l = (rng.random(n) < prev).astype(int)
        assert aupr(rng.random(n), l) == pytest.approx(l.mean(), abs=0.01)

    def test_is_sensitive_to_prevalence_unlike_auroc(self):
        """같은 분리도라도 양성이 드물면 AUPR 이 떨어진다. AUROC 는 안 떨어진다."""
        rng = np.random.default_rng(2)
        def make(prev, n=20000):
            l = (rng.random(n) < prev).astype(int)
            s = rng.normal(0, 1, n) + l * 1.5
            return s, l
        s1, l1 = make(0.5); s2, l2 = make(0.05)
        assert auroc(s1, l1) == pytest.approx(auroc(s2, l2), abs=0.03)
        assert aupr(s2, l2) < aupr(s1, l1) - 0.2


class TestFPRatTPR:
    def test_perfect_separator_has_zero_fpr(self):
        s = np.array([0.0, 0.1, 0.9, 1.0]); l = np.array([0, 0, 1, 1])
        assert fpr_at_tpr(s, l, 0.95) == pytest.approx(0.0)

    def test_uninformative_score_gives_high_fpr(self):
        rng = np.random.default_rng(3)
        n = 20000
        l = (rng.random(n) < 0.07).astype(int)
        assert fpr_at_tpr(rng.random(n), l, 0.95) > 0.85

    def test_stricter_tpr_target_never_lowers_fpr(self):
        rng = np.random.default_rng(4)
        l = (rng.random(5000) < 0.2).astype(int)
        s = rng.normal(0, 1, 5000) + l
        assert fpr_at_tpr(s, l, 0.99) >= fpr_at_tpr(s, l, 0.80) - 1e-12


class TestTrivialScorers:
    def _wafer(self, fails, dies, size=8):
        """앞쪽 dies 칸을 다이로 두고 그중 fails 개를 불량으로."""
        x = np.zeros((1, size, size), np.uint8)
        flat = x.reshape(1, -1)
        flat[0, :dies] = 1
        flat[0, :fails] = 2
        return x

    def test_fail_count_counts_only_value_two(self):
        assert fail_count(self._wafer(fails=7, dies=20))[0] == 7

    def test_fail_ratio_divides_by_die_count_not_pixel_count(self):
        """다이가 없는 칸(0)은 분모에서 빠져야 한다. 웨이퍼 크기가 다르면 결과가 달라진다."""
        assert fail_ratio(self._wafer(fails=5, dies=20))[0] == pytest.approx(0.25)

    def test_fail_ratio_is_zero_when_no_fails(self):
        assert fail_ratio(self._wafer(fails=0, dies=20))[0] == pytest.approx(0.0)

    def test_no_dies_does_not_divide_by_zero(self):
        assert np.isfinite(fail_ratio(np.zeros((1, 8, 8), np.uint8))).all()

    def test_scorers_handle_a_batch(self):
        x = np.concatenate([self._wafer(3, 20), self._wafer(9, 20)], 0)
        assert fail_count(x).tolist() == [3, 9]


class TestEvaluateOOD:
    def test_returns_the_three_headline_metrics(self):
        rng = np.random.default_rng(5)
        l = (rng.random(3000) < 0.1).astype(int)
        s = rng.normal(0, 1, 3000) + l
        m = evaluate_ood(s, l)
        assert set(m) >= {"auroc", "aupr", "fpr_at_95tpr", "prevalence", "n", "n_anomaly"}

    def test_reports_prevalence_as_the_aupr_floor(self):
        l = np.array([0] * 90 + [1] * 10)
        m = evaluate_ood(np.random.default_rng(6).random(100), l)
        assert m["prevalence"] == pytest.approx(0.10)

    def test_rejects_labels_that_are_not_binary(self):
        with pytest.raises(ValueError):
            evaluate_ood(np.array([0.1, 0.2, 0.3]), np.array([0, 1, 2]))

    def test_rejects_a_single_class_label_vector(self):
        """양성이 없으면 AUROC 가 정의되지 않는다. 조용히 nan 을 내면 안 된다."""
        with pytest.raises(ValueError):
            evaluate_ood(np.array([0.1, 0.2, 0.3]), np.zeros(3, dtype=int))
