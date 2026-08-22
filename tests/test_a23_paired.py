"""짝지은 AUPR 차이의 부트스트랩 — 같은 test 집합 위의 두 점수를 비교하는 올바른 검정.

## 왜 필요한가

`ood_template_residual.md` 반증 조건 6 에 "두 arm 의 **독립 CI 가 겹치지 않아야**
이겼다고 한다" 고 박아 뒀다. **그 기준은 짝지은 비교에 틀린 검정이다.**
두 점수는 같은 118,595장 위에서 계산되므로 표본 변동이 공유된다.
독립 CI 겹침은 이 경우 지나치게 보수적이라, 실제로 있는 차이를 놓친다.

여기서 고정하는 것은 그 대안이 최소한 자기 자신과의 비교에서 정확히 0 을 주고
명백한 차이는 0 을 배제한다는 것이다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a21_ood_metrics import aupr  # noqa: E402
from a23_ood_template_eval import paired_aupr_diff_ci  # noqa: E402


class TestPairedAuprDiff:
    def test_comparing_a_score_with_itself_gives_exactly_zero(self):
        """같은 재표본을 두 점수에 똑같이 먹였다는 것의 증거."""
        rng = np.random.default_rng(0)
        l = (rng.random(4000) < 0.08).astype(np.int64)
        s = rng.normal(size=4000) + l
        lo, hi, p = paired_aupr_diff_ci(s, s, l, n_boot=100, seed=1)
        assert lo == pytest.approx(0.0, abs=1e-12)
        assert hi == pytest.approx(0.0, abs=1e-12)

    def test_a_clearly_better_score_wins_with_the_interval_above_zero(self):
        rng = np.random.default_rng(2)
        l = (rng.random(6000) < 0.08).astype(np.int64)
        good = rng.normal(size=6000) + l * 1.5
        bad = rng.normal(size=6000) + l * 0.2
        assert aupr(good, l) > aupr(bad, l)
        lo, hi, p = paired_aupr_diff_ci(good, bad, l, n_boot=200, seed=3)
        assert lo > 0.0
        assert p < 0.01

    def test_two_equally_uninformative_scores_do_not_separate(self):
        rng = np.random.default_rng(4)
        l = (rng.random(5000) < 0.08).astype(np.int64)
        a = rng.normal(size=5000)
        b = rng.normal(size=5000)
        lo, hi, p = paired_aupr_diff_ci(a, b, l, n_boot=200, seed=5)
        assert lo < 0.0 < hi
        assert p > 0.05

    def test_is_reproducible_for_a_fixed_seed(self):
        rng = np.random.default_rng(6)
        l = (rng.random(3000) < 0.1).astype(np.int64)
        a = rng.normal(size=3000) + l
        b = rng.normal(size=3000) + l * 0.5
        assert paired_aupr_diff_ci(a, b, l, 100, 7) == paired_aupr_diff_ci(a, b, l, 100, 7)
