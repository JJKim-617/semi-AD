"""짝지은 **blocked** AUPR 차이 — 주 지표로 판정하려면 검정도 주 지표여야 한다.

`paired_aupr_diff_ci` 는 순서 의존 AUPR 로 차이를 잰다. 동점이 적으면 상관없지만
`k2` 계열은 고유값이 25개뿐이라 두 관례가 0.017 까지 벌어진다.
**주 지표를 blocked 로 정해 놓고 검정만 순서 의존으로 하면 판정이 다른 것을 재게 된다.**
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a21_ood_metrics import aupr_blocked  # noqa: E402
from a23_ood_template_eval import paired_aupr_blocked_diff_ci  # noqa: E402


class TestPairedBlocked:
    def test_a_score_against_itself_is_exactly_zero(self):
        rng = np.random.default_rng(0)
        l = (rng.random(3000) < 0.09).astype(np.int64)
        s = np.floor(rng.random(3000) * 12)
        lo, hi, p = paired_aupr_blocked_diff_ci(s, s, l, n_boot=60, seed=1)
        assert lo == pytest.approx(0.0, abs=1e-12)
        assert hi == pytest.approx(0.0, abs=1e-12)

    def test_a_clearly_better_score_wins(self):
        rng = np.random.default_rng(2)
        l = (rng.random(4000) < 0.1).astype(np.int64)
        good = np.floor((rng.random(4000) + l * 1.2) * 10)
        bad = np.floor((rng.random(4000) + l * 0.1) * 10)
        assert aupr_blocked(good, l) > aupr_blocked(bad, l)
        lo, hi, p = paired_aupr_blocked_diff_ci(good, bad, l, n_boot=120, seed=3)
        assert lo > 0 and p < 0.05

    def test_two_uninformative_scores_do_not_separate(self):
        rng = np.random.default_rng(4)
        l = (rng.random(4000) < 0.1).astype(np.int64)
        a = np.floor(rng.random(4000) * 10)
        b = np.floor(rng.random(4000) * 10)
        lo, hi, p = paired_aupr_blocked_diff_ci(a, b, l, n_boot=150, seed=5)
        assert lo < 0 < hi

    def test_it_is_reproducible(self):
        rng = np.random.default_rng(6)
        l = (rng.random(2000) < 0.1).astype(np.int64)
        a = np.floor(rng.random(2000) * 8)
        b = np.floor(rng.random(2000) * 8)
        assert (paired_aupr_blocked_diff_ci(a, b, l, 50, 9)
                == paired_aupr_blocked_diff_ci(a, b, l, 50, 9))

    def test_it_can_disagree_with_the_order_dependent_test_when_ties_are_heavy(self):
        """왜 따로 필요한지를 박는다 — 동점이 많으면 두 검정의 점추정이 다르다."""
        from a23_ood_template_eval import paired_aupr_diff_ci
        rng = np.random.default_rng(7)
        l = (rng.random(6000) < 0.1).astype(np.int64)
        a = np.floor((rng.random(6000) + l * 0.5) * 4)      # 고유값 5개 수준
        b = rng.random(6000) + l * 0.35                     # 동점 없음
        lo_b, hi_b, _ = paired_aupr_blocked_diff_ci(a, b, l, n_boot=120, seed=11)
        lo_o, hi_o, _ = paired_aupr_diff_ci(a, b, l, n_boot=120, seed=11)
        mid_b, mid_o = (lo_b + hi_b) / 2, (lo_o + hi_o) / 2
        assert abs(mid_b - mid_o) > 1e-3
