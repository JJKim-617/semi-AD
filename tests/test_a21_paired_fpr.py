"""`FPR@95TPR` 의 짝지은 부트스트랩 — 고재현율 arm 계열이 쓰는 검정.

## 왜 새로 필요한가

`paired_aupr_blocked_diff_ci` 는 주 지표(blocked AUPR)용이다.
`candidate/ood_high_recall_arm.md` 는 **그 arm 계열 안에서만** `FPR@95TPR` 을
주 지표로 선언했고(§3), §9.3 이 **짝지은 부트스트랩으로 판정한다**고 못 박았다.
**지표를 바꿨으면 검정도 바꿔야 한다 — 정정 9 가 정확히 그 실수였다.**

## 부호 관례

`FPR@95TPR` 은 **낮을수록 좋다.** 그래서 `a - b` 가 **음수면 a 가 이긴 것**이고
**CI 상한이 0 보다 작아야** 이겼다고 말할 수 있다. 이 방향을 테스트로 박는다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a21_ood_metrics import fpr_at_tpr, paired_fpr_at_tpr_diff_ci  # noqa: E402


def toy(seed=0, n_pos=400, n_neg=4000, shift_a=1.6, shift_b=1.0):
    rng = np.random.default_rng(seed)
    lab = np.concatenate([np.zeros(n_neg, np.int64), np.ones(n_pos, np.int64)])
    neg = rng.normal(0, 1, n_neg)
    a = np.concatenate([neg, rng.normal(shift_a, 1, n_pos)])
    b = np.concatenate([neg + rng.normal(0, 0.01, n_neg), rng.normal(shift_b, 1, n_pos)])
    return a, b, lab


def test_identical_scores_give_exactly_zero_difference():
    """**재표본을 두 arm 에 똑같이 먹이는지**를 이 시험이 잡는다.

    재표본이 다르면 같은 점수인데도 차이가 퍼진다.
    """
    a, _, lab = toy()
    lo, hi, p = paired_fpr_at_tpr_diff_ci(a, a, lab, n_boot=200, seed=0)
    assert lo == 0.0 and hi == 0.0


def test_a_better_arm_gives_a_negative_interval():
    """낮을수록 좋은 지표다. 이긴 쪽은 **CI 상한이 0 미만**이어야 한다."""
    a, b, lab = toy()
    assert fpr_at_tpr(a, lab, 0.95) < fpr_at_tpr(b, lab, 0.95)
    lo, hi, p = paired_fpr_at_tpr_diff_ci(a, b, lab, n_boot=300, seed=0)
    assert hi < 0.0
    assert lo <= hi
    assert p < 0.05


def test_the_sign_flips_when_the_arms_are_swapped():
    a, b, lab = toy()
    lo1, hi1, _ = paired_fpr_at_tpr_diff_ci(a, b, lab, n_boot=200, seed=3)
    lo2, hi2, _ = paired_fpr_at_tpr_diff_ci(b, a, lab, n_boot=200, seed=3)
    assert lo2 == pytest.approx(-hi1, abs=1e-12)
    assert hi2 == pytest.approx(-lo1, abs=1e-12)


def test_two_indistinguishable_arms_give_an_interval_containing_zero():
    rng = np.random.default_rng(1)
    a, _, lab = toy(seed=5)
    b = a + rng.normal(0, 1e-9, len(a))       # 사실상 같은 채점기
    lo, hi, p = paired_fpr_at_tpr_diff_ci(a, b, lab, n_boot=300, seed=0)
    assert lo <= 0.0 <= hi


def test_is_deterministic_given_a_seed():
    a, b, lab = toy()
    r1 = paired_fpr_at_tpr_diff_ci(a, b, lab, n_boot=100, seed=7)
    r2 = paired_fpr_at_tpr_diff_ci(a, b, lab, n_boot=100, seed=7)
    r3 = paired_fpr_at_tpr_diff_ci(a, b, lab, n_boot=100, seed=8)
    assert r1 == r2
    assert r1 != r3


def test_the_point_estimate_lies_inside_the_interval():
    a, b, lab = toy()
    d = fpr_at_tpr(a, lab, 0.95) - fpr_at_tpr(b, lab, 0.95)
    lo, hi, _ = paired_fpr_at_tpr_diff_ci(a, b, lab, n_boot=400, seed=0)
    assert lo <= d <= hi


def test_target_recall_is_honoured():
    """95% 만이 아니라 임의 목표에서도 동작해야 한다 — 운영 지점이 여럿이다."""
    a, b, lab = toy()
    lo80, hi80, _ = paired_fpr_at_tpr_diff_ci(a, b, lab, target=0.80, n_boot=200, seed=0)
    lo95, hi95, _ = paired_fpr_at_tpr_diff_ci(a, b, lab, target=0.95, n_boot=200, seed=0)
    assert (lo80, hi80) != (lo95, hi95)


def test_rejects_a_resample_with_only_one_class():
    """재표본에 결함이 하나도 없으면 `FPR@TPR` 이 정의되지 않는다. nan 으로 빼야 한다."""
    lab = np.array([0] * 40 + [1] * 2, np.int64)
    a = np.arange(42, dtype=float)
    b = a[::-1].copy()
    lo, hi, p = paired_fpr_at_tpr_diff_ci(a, b, lab, n_boot=200, seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)
