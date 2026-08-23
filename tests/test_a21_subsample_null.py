"""작은 n 에서 클래스별 AUROC 이 얼마나 흔들리는가 — 부표집 귀무분포.

## 왜 필요한가

11차 사이클이 `val_unseen` 에서 **Scratch AUROC 0.7835 (n=92)** 를 관측하고
"가장 자랑한 클래스가 흔들린 유일한 지점" 이라 적었다. 그런데
**n=92 면 원래 얼마나 흔들리는지를 재지 않았다.**

이 워크스트림은 **n=2 로 판정했다가 세 번 뒤집혔다.** 그래서 작은 n 의 관측값은
먼저 "같은 분포에서 그 n 으로 재면 어떤 값이 나오는가" 에 대고 본다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a21_ood_metrics import auroc, subsample_auroc_null  # noqa: E402


def toy(seed=0, n_pos=200, n_neg=2000):
    rng = np.random.default_rng(seed)
    pos = rng.normal(1.2, 1.0, n_pos)
    neg = rng.normal(0.0, 1.0, n_neg)
    return pos, neg


def test_full_size_subsample_is_the_point_estimate():
    """n 을 그대로 두면 뽑을 것이 하나뿐이라 **분포가 한 점**이어야 한다."""
    pos, neg = toy()
    r = subsample_auroc_null(pos, neg, n=len(pos), n_rep=7, seed=0)
    full = auroc(np.concatenate([neg, pos]),
                 np.concatenate([np.zeros(len(neg), int), np.ones(len(pos), int)]))
    assert np.allclose(r["values"], full)
    assert r["ci95"][0] == pytest.approx(full)
    assert r["ci95"][1] == pytest.approx(full)


def test_smaller_n_widens_the_interval():
    """**이 함수가 존재하는 이유.** n 이 줄면 구간이 넓어져야 한다."""
    pos, neg = toy()
    wide = subsample_auroc_null(pos, neg, n=20, n_rep=400, seed=0)
    narrow = subsample_auroc_null(pos, neg, n=150, n_rep=400, seed=0)
    w = wide["ci95"][1] - wide["ci95"][0]
    nw = narrow["ci95"][1] - narrow["ci95"][0]
    assert w > nw


def test_mean_is_unbiased_for_the_point_estimate():
    """비복원 부표집은 AUROC 을 편향시키지 않는다 — 평균이 전체값 근처여야 한다."""
    pos, neg = toy()
    full = auroc(np.concatenate([neg, pos]),
                 np.concatenate([np.zeros(len(neg), int), np.ones(len(pos), int)]))
    r = subsample_auroc_null(pos, neg, n=50, n_rep=2000, seed=3)
    assert r["mean"] == pytest.approx(full, abs=0.01)


def test_is_deterministic_given_seed():
    pos, neg = toy()
    a = subsample_auroc_null(pos, neg, n=30, n_rep=50, seed=11)
    b = subsample_auroc_null(pos, neg, n=30, n_rep=50, seed=11)
    assert np.array_equal(a["values"], b["values"])
    c = subsample_auroc_null(pos, neg, n=30, n_rep=50, seed=12)
    assert not np.array_equal(a["values"], c["values"])


def test_rejects_n_larger_than_available_positives():
    pos, neg = toy(n_pos=10)
    with pytest.raises(ValueError):
        subsample_auroc_null(pos, neg, n=11, n_rep=5, seed=0)


def test_reports_the_fraction_below_an_observed_value():
    """판정 규칙 S1 이 쓰는 값. '관측값이 귀무의 몇 분위인가'."""
    pos, neg = toy()
    r = subsample_auroc_null(pos, neg, n=40, n_rep=500, seed=0, observed=0.0)
    assert r["frac_below_observed"] == 0.0
    r = subsample_auroc_null(pos, neg, n=40, n_rep=500, seed=0, observed=1.0)
    assert r["frac_below_observed"] == 1.0


def test_subsampling_is_without_replacement():
    """복원 추출이면 동점이 생겨 AUROC 이 위로 치우친다. 비복원이어야 한다."""
    pos = np.arange(6, dtype=float) + 10.0
    neg = np.arange(6, dtype=float)
    r = subsample_auroc_null(pos, neg, n=3, n_rep=200, seed=0)
    assert np.allclose(r["values"], 1.0)
