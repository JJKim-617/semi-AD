"""클래스별 logit 오프셋 최적화 테스트.

스칼라 tau 는 오프셋을 log(prior) 방향 한 축으로만 움직인다(a5 결과 +0.0007 로 무의미).
macro-F1 은 분해되지 않는 지표라 최적 결정규칙이 단순 argmax 가 아니고, 클래스마다
독립적인 오프셋이 필요하다. 좌표 상승법으로 그 벡터를 찾는다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a6_perclass_offset import apply_offsets, optimize_offsets


class TestApply:
    def test_zero_offsets_leave_predictions_unchanged(self):
        z = np.array([[1.0, 0.5], [0.2, 0.9]])
        assert np.array_equal(apply_offsets(z, np.zeros(2)).argmax(1), z.argmax(1))

    def test_offset_shifts_that_class_only(self):
        z = np.array([[1.0, 0.9, 0.8]])
        out = apply_offsets(z, np.array([0.0, 0.5, 0.0]))
        assert out.argmax(1)[0] == 1

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError):
            apply_offsets(np.zeros((2, 3)), np.zeros(2))


class TestOptimize:
    def _skewed(self, n=400, seed=0):
        """다수 클래스가 소수 클래스를 흡수하는 상황. 오프셋으로 되돌릴 수 있어야 한다."""
        rng = np.random.default_rng(seed)
        y = np.where(rng.random(n) < 0.9, 0, 1)
        z = np.zeros((n, 2))
        z[:, 0] = 1.0
        z[y == 1, 1] = 0.85
        z[y == 0, 1] = 0.2
        return z + rng.normal(0, 0.05, z.shape), y

    def test_returns_one_offset_per_class(self):
        z, y = self._skewed()
        assert optimize_offsets(z, y, 2).shape == (2,)

    def test_never_scores_worse_than_zero_offsets(self):
        """좌표 상승법은 0 벡터에서 시작하므로 결과가 기준선보다 나쁠 수 없다."""
        from a4_eval_wm811k_cls import evaluate
        z, y = self._skewed()
        base = evaluate(y, z.argmax(1), 2)["macro_f1"]
        off = optimize_offsets(z, y, 2)
        assert evaluate(y, apply_offsets(z, off).argmax(1), 2)["macro_f1"] >= base

    def test_recovers_absorbed_minority_class(self):
        from a4_eval_wm811k_cls import evaluate
        z, y = self._skewed()
        base = evaluate(y, z.argmax(1), 2)["macro_f1"]
        off = optimize_offsets(z, y, 2)
        assert evaluate(y, apply_offsets(z, off).argmax(1), 2)["macro_f1"] > base + 0.05

    def test_is_deterministic(self):
        z, y = self._skewed()
        assert np.array_equal(optimize_offsets(z, y, 2), optimize_offsets(z, y, 2))

    def test_more_rounds_never_hurt(self):
        from a4_eval_wm811k_cls import evaluate
        z, y = self._skewed()
        s = [evaluate(y, apply_offsets(z, optimize_offsets(z, y, 2, rounds=r)).argmax(1),
                      2)["macro_f1"] for r in (1, 3)]
        assert s[1] >= s[0]

    def test_already_optimal_input_yields_no_gain(self):
        from a4_eval_wm811k_cls import evaluate
        z = np.array([[5.0, 0.0], [0.0, 5.0]] * 20)
        y = np.array([0, 1] * 20)
        off = optimize_offsets(z, y, 2)
        assert evaluate(y, apply_offsets(z, off).argmax(1), 2)["macro_f1"] == pytest.approx(1.0)
