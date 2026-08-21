"""a4 평가 지표 테스트.

이 데이터셋은 none 이 85% 라 accuracy 만 보면 안 된다. 그 성질을 테스트로 못박는다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a4_eval_wm811k_cls import evaluate


def test_perfect_prediction_scores_one():
    y = np.array([0, 1, 2, 0, 1, 2])
    r = evaluate(y, y.copy(), num_classes=3)
    assert r["accuracy"] == pytest.approx(1.0)
    assert r["macro_f1"] == pytest.approx(1.0)


def test_majority_only_prediction_has_high_accuracy_but_low_macro_f1():
    """none 만 찍는 모델. 이 데이터셋의 핵심 함정을 재현한다."""
    y = np.array([0] * 85 + [1] * 10 + [2] * 5)
    pred = np.zeros_like(y)
    r = evaluate(y, pred, num_classes=3)
    assert r["accuracy"] == pytest.approx(0.85)
    assert r["macro_f1"] < 0.35


def test_per_class_recall_covers_every_class():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 2, 2])
    r = evaluate(y, pred, num_classes=3)
    assert sorted(r["per_class_recall"].keys()) == [0, 1, 2]


def test_class_never_predicted_gets_zero_recall():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 0, 0])
    r = evaluate(y, pred, num_classes=2)
    assert r["per_class_recall"][1] == pytest.approx(0.0)


def test_class_absent_from_labels_does_not_crash():
    """소수 클래스가 특정 분할에 아예 없을 수 있다."""
    y = np.array([0, 0, 1])
    pred = np.array([0, 0, 1])
    r = evaluate(y, pred, num_classes=9)
    assert len(r["per_class_recall"]) == 9
    assert not np.isnan(r["macro_f1"])


def test_recall_is_computed_per_class_not_globally():
    y = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([0, 0, 0, 0, 0, 1])
    r = evaluate(y, pred, num_classes=2)
    assert r["per_class_recall"][0] == pytest.approx(1.0)
    assert r["per_class_recall"][1] == pytest.approx(0.5)


def test_confusion_matrix_shape_matches_num_classes():
    y = np.array([0, 1])
    pred = np.array([0, 1])
    r = evaluate(y, pred, num_classes=9)
    assert np.array(r["confusion"]).shape == (9, 9)


def test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        evaluate(np.array([0, 1]), np.array([0]), num_classes=2)
