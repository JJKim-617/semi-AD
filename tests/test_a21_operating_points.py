"""운영 지점 표 — AUPR 총합이 가리는 것을 드러낸다.

k=7 국소 밀도는 AUPR 0.7145 인데, 결함의 50% 를 잡을 때 precision 0.862,
80% 를 잡을 때 0.357 이다. **50%~80% 사이에서 반토막 난다.**
총합 지표만 보면 이 붕괴가 안 보인다. 그래서 표를 상설로 만든다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a21_ood_metrics import operating_points  # noqa: E402


class TestOperatingPoints:
    def test_a_perfect_score_gives_precision_one_everywhere(self):
        label = np.array([0] * 90 + [1] * 10)
        score = label.astype(np.float64)
        rows = operating_points(score, label, recalls=(0.5, 0.8, 1.0))
        for r in rows:
            assert r["precision"] == pytest.approx(1.0)
            assert r["false_positives"] == 0

    def test_random_score_gives_precision_near_prevalence(self):
        rng = np.random.default_rng(0)
        label = (rng.random(20000) < 0.1).astype(np.int64)
        rows = operating_points(rng.random(20000), label, recalls=(0.5,))
        assert rows[0]["precision"] == pytest.approx(0.1, abs=0.02)

    def test_recall_actually_reached_is_at_least_the_target(self):
        rng = np.random.default_rng(1)
        label = (rng.random(5000) < 0.08).astype(np.int64)
        score = rng.random(5000) + label * 0.7
        for r in operating_points(score, label, recalls=(0.5, 0.8, 0.95)):
            assert r["recall"] >= r["target_recall"] - 1e-9

    def test_false_positive_count_matches_precision_and_true_positives(self):
        rng = np.random.default_rng(2)
        label = (rng.random(8000) < 0.07).astype(np.int64)
        score = rng.random(8000) + label
        for r in operating_points(score, label, recalls=(0.5, 0.8, 0.95)):
            tp, fp = r["true_positives"], r["false_positives"]
            assert r["precision"] == pytest.approx(tp / (tp + fp))
            assert tp + r["false_negatives"] == int(label.sum())

    def test_more_recall_costs_more_false_positives(self):
        rng = np.random.default_rng(3)
        label = (rng.random(9000) < 0.09).astype(np.int64)
        score = rng.random(9000) + label * 0.8
        rows = operating_points(score, label, recalls=(0.5, 0.8, 0.95))
        fps = [r["false_positives"] for r in rows]
        assert fps[0] <= fps[1] <= fps[2]

    def test_it_rejects_targets_outside_the_unit_interval(self):
        label = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError):
            operating_points(np.arange(4.0), label, recalls=(0.0,))
        with pytest.raises(ValueError):
            operating_points(np.arange(4.0), label, recalls=(1.5,))


class TestTiesAreReported:
    """동점이 많으면 운영 지점이 **한 점이 아니라 구간**이 된다.

    창 3x3 밀도는 고유값이 24개뿐이라 문턱 하나가 수만 장을 한꺼번에 넘긴다.
    그걸 모르고 precision 을 한 숫자로 읽으면 안 된다.
    """

    def test_it_reports_how_many_samples_sit_exactly_at_the_threshold(self):
        label = np.array([1] * 10 + [0] * 90)
        score = np.zeros(100)
        score[:10] = 1.0
        score[10:60] = 1.0                      # 정상 50장이 결함과 같은 점수
        rows = operating_points(score, label, recalls=(0.5,))
        assert rows[0]["n_tied_at_threshold"] >= 60

    def test_no_ties_reports_one(self):
        label = np.array([0, 0, 1, 1])
        rows = operating_points(np.array([0.1, 0.2, 0.3, 0.4]), label, recalls=(0.5,))
        assert rows[0]["n_tied_at_threshold"] == 1
