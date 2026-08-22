"""동점을 블록으로 묶는 AUPR — 관례를 정한다.

## 왜

k=3 국소 밀도는 고유값이 108개뿐이다. 결정론적 AUPR 0.4986 이
동점 무작위 해소 범위 [0.4112, 0.4248] **바깥**에 있었다.
동점 안에서 원래 인덱스 순서를 쓰는데 test 배열 위치와 결함 여부에 약한 상관이 있어
(0.0116) **동점 처리로 라벨이 샌다.**

같은 점수를 받은 표본은 **어떤 문턱으로도 서로 갈리지 않는다.** 그러니 지표도
그것들을 갈라서는 안 된다. 서로 다른 문턱 값에서만 곡선 위의 점을 찍으면
**입력 순서와 무관한 하나의 값**이 나온다. 그것을 관례로 박는다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a21_ood_metrics import aupr, aupr_blocked  # noqa: E402


class TestBlockedAuprIsWellDefined:
    def test_it_does_not_depend_on_input_order(self):
        """**이 함수의 존재 이유.** 기존 aupr 은 여기서 값이 바뀐다."""
        rng = np.random.default_rng(0)
        label = (rng.random(4000) < 0.1).astype(np.int64)
        score = np.floor(rng.random(4000) * 8)          # 고유값 8개, 동점 투성이
        base = aupr_blocked(score, label)
        for seed in range(5):
            p = np.random.default_rng(seed).permutation(len(label))
            assert aupr_blocked(score[p], label[p]) == pytest.approx(base, abs=1e-12)

    def test_the_old_one_does_depend_on_input_order(self):
        """왜 고쳐야 하는지를 같이 박는다."""
        rng = np.random.default_rng(1)
        label = (rng.random(4000) < 0.1).astype(np.int64)
        score = np.floor(rng.random(4000) * 6)
        vals = []
        for seed in range(6):
            p = np.random.default_rng(seed).permutation(len(label))
            vals.append(aupr(score[p], label[p]))
        assert max(vals) - min(vals) > 1e-3

    def test_a_constant_score_gives_exactly_the_prevalence(self):
        """정보가 없는 점수는 무작위 기준선이어야 한다. 한 블록이므로 정확히 유병률이다."""
        label = np.array([1] * 7 + [0] * 93)
        assert aupr_blocked(np.ones(100), label) == pytest.approx(0.07, abs=1e-12)

    def test_it_matches_the_old_one_when_there_are_no_ties(self):
        rng = np.random.default_rng(2)
        label = (rng.random(3000) < 0.09).astype(np.int64)
        score = rng.normal(size=3000) + label
        assert len(np.unique(score)) == 3000
        assert aupr_blocked(score, label) == pytest.approx(aupr(score, label), abs=1e-12)

    def test_a_perfect_score_is_one(self):
        label = np.array([0] * 80 + [1] * 20)
        assert aupr_blocked(label.astype(np.float64), label) == pytest.approx(1.0)

    def test_hand_computed_two_block_example(self):
        """블록 2개: 상위 블록에 양성 2 음성 2, 하위 블록에 양성 1 음성 5.

        블록 1 끝: recall 2/3, precision 2/4 → 기여 (2/3)*(1/2) = 1/3
        블록 2 끝: recall 1,   precision 3/10 → 기여 (1/3)*(3/10) = 1/10
        합 = 1/3 + 1/10 = 13/30
        """
        score = np.array([2, 2, 2, 2, 1, 1, 1, 1, 1, 1], dtype=np.float64)
        label = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int64)
        assert aupr_blocked(score, label) == pytest.approx(13.0 / 30.0, abs=1e-12)

    def test_it_is_bounded_by_the_random_tie_break_extremes(self):
        """블록 값은 낙관/비관 해소 사이에 있어야 한다. 밖으로 나가면 계산이 틀린 것이다."""
        rng = np.random.default_rng(3)
        label = (rng.random(5000) < 0.12).astype(np.int64)
        score = np.floor(rng.random(5000) * 10)
        opt = aupr(score - 1e-9 * label, label)        # 양성을 동점 앞으로
        pes = aupr(score + 1e-9 * label, label)        # 양성을 동점 뒤로
        b = aupr_blocked(score, label)
        assert min(opt, pes) - 1e-9 <= b <= max(opt, pes) + 1e-9
