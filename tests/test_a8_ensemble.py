"""앙상블 테스트.

수렴된 모델이 3개 있고(focal seed0, focal seed1, plain CE) 서로 다른 클래스에서 강하다
(seed 간 Edge-Loc ±0.049, Scratch ±0.029). 재학습 없이 합칠 수 있다.

logit 평균과 확률 평균은 다르다. 확률 평균이 앙상블의 표준이며, 한 모델이 극단적으로
확신하는 logit 에 결과가 끌려가는 것을 막는다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a8_ensemble import combine


class TestCombine:
    def _two(self):
        a = np.array([[2.0, 0.0], [0.0, 1.0]])
        b = np.array([[0.0, 2.0], [0.0, 3.0]])
        return [a, b]

    def test_single_model_is_returned_unchanged_in_ranking(self):
        a = np.array([[2.0, 0.5], [0.1, 3.0]])
        assert np.array_equal(combine([a]).argmax(1), a.argmax(1))

    def test_identical_models_match_single_model(self):
        a = np.array([[2.0, 0.5], [0.1, 3.0]])
        assert np.array_equal(combine([a, a]).argmax(1), a.argmax(1))

    def test_output_shape_matches_inputs(self):
        assert combine(self._two()).shape == (2, 2)

    def test_probability_and_logit_averaging_can_disagree(self):
        """두 방식이 실제로 다르다는 것을 확인한다. 같다면 선택이 무의미하다."""
        a = np.array([[10.0, 9.0]])
        b = np.array([[0.0, 0.5]])
        p = combine([a, b], mode="prob")
        l = combine([a, b], mode="logit")
        assert not np.allclose(p / p.sum(), l / l.sum())

    def test_probability_mode_output_is_a_distribution(self):
        out = combine(self._two(), mode="prob")
        assert np.allclose(out.sum(1), 1.0)
        assert (out >= 0).all()

    def test_weights_shift_the_result_toward_the_heavier_model(self):
        a = np.array([[5.0, 0.0]])
        b = np.array([[0.0, 5.0]])
        assert combine([a, b], weights=[10.0, 1.0]).argmax(1)[0] == 0
        assert combine([a, b], weights=[1.0, 10.0]).argmax(1)[0] == 1

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            combine([])

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError):
            combine([np.zeros((2, 3)), np.zeros((2, 4))])

    def test_rejects_weight_count_mismatch(self):
        with pytest.raises(ValueError):
            combine(self._two(), weights=[1.0])
