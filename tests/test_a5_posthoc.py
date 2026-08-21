"""사후 logit 보정 테스트.

Menon et al. ICLR 2021 의 post-hoc logit adjustment: argmax_y [f_y(x) - tau * log(pi_y)].
tau > 0 이면 경계가 희소 클래스 쪽으로, tau < 0 이면 다수 클래스 쪽으로 움직인다.

우리 test 는 none 이 93.3% 로 train(67.4%)보다 더 쏠려 있어 표준 long-tail 가정과 방향이
반대다. 그래서 tau 를 양수로 고정하지 않고 음수까지 열어 스윕한다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a5_posthoc_adjust import adjust_logits, estimate_prior, sweep_tau


class TestPrior:
    def test_sums_to_one(self):
        y = np.array([0, 0, 0, 1, 2])
        assert estimate_prior(y, 3).sum() == pytest.approx(1.0)

    def test_matches_empirical_frequency(self):
        y = np.array([0, 0, 0, 1])
        p = estimate_prior(y, 2)
        assert p[0] == pytest.approx(0.75)

    def test_absent_class_gets_positive_mass_not_zero(self):
        """log(0) 이 되면 보정이 발산한다. 스무딩이 필요하다."""
        y = np.array([0, 0, 1])
        p = estimate_prior(y, 3)
        assert p[2] > 0
        assert np.isfinite(np.log(p)).all()


class TestAdjust:
    def _logits(self):
        return np.array([[2.0, 1.9], [0.5, 0.4]])

    def test_tau_zero_leaves_logits_unchanged(self):
        z = self._logits()
        out = adjust_logits(z, np.array([0.9, 0.1]), tau=0.0)
        assert np.allclose(out, z)

    def test_positive_tau_shifts_prediction_toward_rare_class(self):
        z = self._logits()
        prior = np.array([0.9, 0.1])
        assert adjust_logits(z, prior, tau=0.0).argmax(1).tolist() == [0, 0]
        assert adjust_logits(z, prior, tau=2.0).argmax(1).tolist() == [1, 1]

    def test_negative_tau_shifts_prediction_toward_frequent_class(self):
        z = np.array([[1.0, 1.2]])
        prior = np.array([0.9, 0.1])
        assert adjust_logits(z, prior, tau=0.0).argmax(1).tolist() == [1]
        assert adjust_logits(z, prior, tau=-2.0).argmax(1).tolist() == [0]

    def test_preserves_shape(self):
        z = np.zeros((5, 4))
        assert adjust_logits(z, np.full(4, 0.25), tau=1.0).shape == (5, 4)

    def test_rejects_prior_length_mismatch(self):
        with pytest.raises(ValueError):
            adjust_logits(np.zeros((2, 3)), np.array([0.5, 0.5]), tau=1.0)


class TestSweep:
    def test_returns_tau_that_maximizes_the_metric(self):
        """희소 클래스를 놓치는 모델. 양수 tau 가 macro-F1 을 올려야 한다."""
        rng = np.random.default_rng(0)
        y = np.array([0] * 90 + [1] * 10)
        z = np.zeros((100, 2))
        z[:, 0] = 1.0
        z[y == 1, 1] = 0.9   # 희소 클래스는 살짝 못 미침
        z += rng.normal(0, 0.01, z.shape)
        r = sweep_tau(z, y, num_classes=2, prior=np.array([0.9, 0.1]),
                      taus=np.linspace(-2, 2, 41))
        assert r["best_tau"] > 0
        assert r["best_score"] > r["scores"][np.argmin(np.abs(r["taus"]))]

    def test_reports_score_for_every_tau(self):
        y = np.array([0, 1])
        z = np.array([[1.0, 0.0], [0.0, 1.0]])
        taus = np.linspace(-1, 1, 5)
        r = sweep_tau(z, y, 2, np.array([0.5, 0.5]), taus)
        assert len(r["scores"]) == len(taus)

    def test_tau_zero_score_equals_unadjusted_macro_f1(self):
        from a4_eval_wm811k_cls import evaluate
        rng = np.random.default_rng(1)
        y = rng.integers(0, 3, 60)
        z = rng.normal(size=(60, 3))
        r = sweep_tau(z, y, 3, np.full(3, 1 / 3), np.array([0.0]))
        assert r["scores"][0] == pytest.approx(evaluate(y, z.argmax(1), 3)["macro_f1"])
