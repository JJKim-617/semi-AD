"""결함률 하나만 보정하는 선택 지표.

9차원 사전확률 전체를 추정하는 BBSE 는 반증 조건 둘 다 실패했다
(TV 0.0129 vs CC 0.0128, Spearman +0.100 vs 요구 +0.6).
축을 쪼개보니 순위 능력의 대부분이 **결함률 하나**에서 나왔다.

| 보정 | Spearman | 고르는 실행 |
|---|---:|---|
| 없음(현행 val) | +0.100 | 실제 4위 |
| 결함률만 | +0.700 | **실제 1위** |
| 구성만 | +0.300 | 실제 2위 |
| 둘 다(oracle) | +0.900 | 실제 2위 |

9차원 추정이 실패한 이유는 가중치가 q/p_val 이라 절대오차가 아니라 **비율오차**가
문제이고, 참 비율이 0.0008 인 Near-full 같은 클래스에서 비율오차가 폭발하기 때문이다.
이진으로 축소하면 각 셀에 표본이 수천 장씩 있어 그 문제가 사라진다.

주의. 이 발견은 사후적이다(5개 실행에서 찾았다). pad 3 seed 로 독립 검증한다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a12_shift_estimation import blend_prior, estimate_defect_rate


class TestBlendPrior:
    def test_sums_to_one(self):
        q = blend_prior(0.07, np.array([0.5, 0.3, 0.2]))
        assert q.sum() == pytest.approx(1.0)

    def test_first_class_holds_the_non_defect_mass(self):
        q = blend_prior(0.07, np.array([0.5, 0.5]))
        assert q[0] == pytest.approx(0.93)

    def test_defect_mass_follows_the_composition(self):
        q = blend_prior(0.10, np.array([0.8, 0.2]))
        assert q[1] == pytest.approx(0.08)
        assert q[2] == pytest.approx(0.02)

    def test_normalises_an_unnormalised_composition(self):
        q = blend_prior(0.10, np.array([8.0, 2.0]))
        assert q[1] == pytest.approx(0.08) and q.sum() == pytest.approx(1.0)

    def test_rejects_a_rate_outside_the_unit_interval(self):
        with pytest.raises(ValueError):
            blend_prior(1.5, np.array([0.5, 0.5]))


def _detector(source_rate, target_rate, miss, n=200_000, seed=0):
    """결함을 miss 확률로 놓치는 검출기. 정상은 항상 맞힌다.

    이 비대칭 때문에 '예측된 결함 비율' 은 참 결함률을 반드시 과소추정한다.
    """
    rng = np.random.default_rng(seed)

    def sample(rate):
        y = (rng.random(n) < rate).astype(int)
        pred = y.copy()
        pred[(y == 1) & (rng.random(n) < miss)] = 0
        return y, pred

    return sample(source_rate), sample(target_rate)


class TestEstimateDefectRate:
    def test_uses_no_target_labels(self):
        import inspect
        params = set(inspect.signature(estimate_defect_rate).parameters)
        assert "target_y" not in params and "test_y" not in params

    def test_no_shift_returns_the_source_rate(self):
        (sy, sp), (_, tp) = _detector(0.32, 0.32, miss=0.3)
        assert estimate_defect_rate(sp, sy, tp) == pytest.approx(0.32, abs=0.01)

    def test_recovers_a_shifted_rate_the_raw_count_gets_wrong(self):
        """이 모듈의 존재 이유. 검출기가 결함을 놓쳐도 참 결함률이 나와야 한다."""
        (sy, sp), (_, tp) = _detector(0.32, 0.067, miss=0.3)
        raw = tp.mean()
        got = estimate_defect_rate(sp, sy, tp)
        assert abs(raw - 0.067) > 0.01, "이 설정에서 단순 계수는 편향돼야 한다"
        assert got == pytest.approx(0.067, abs=0.005)
        assert abs(got - 0.067) < abs(raw - 0.067)

    def test_stays_within_the_unit_interval(self):
        rng = np.random.default_rng(5)
        val_y = (rng.random(2000) < 0.3).astype(int)
        val_pred = rng.integers(0, 2, 2000)      # 무작위 예측
        target_pred = np.ones(2000, dtype=int)   # 전부 결함이라 예측
        r = estimate_defect_rate(val_pred, val_y, target_pred)
        assert 0.0 <= r <= 1.0

    def test_accepts_multiclass_labels_and_collapses_them(self):
        """실제 호출부는 9-class 라벨을 준다. 0 이 아니면 결함이다."""
        val_y = np.array([0, 0, 1, 3, 7])
        val_pred = np.array([0, 0, 1, 3, 0])
        target_pred = np.array([0, 0, 0, 5, 0])
        r = estimate_defect_rate(val_pred, val_y, target_pred)
        assert 0.0 <= r <= 1.0

    def test_a_perfect_detector_reduces_to_counting(self):
        (sy, sp), (_, tp) = _detector(0.32, 0.067, miss=0.0)
        assert estimate_defect_rate(sp, sy, tp) == pytest.approx(tp.mean(), abs=0.005)
