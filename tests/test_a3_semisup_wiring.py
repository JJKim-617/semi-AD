"""준지도 손실을 a3 학습 루프에 잇는 배선 (E25, 18차에 구현만).

**18차는 이 실험을 돌리지 않는다.** 사전 등록(`candidate/semi_supervised.md`)과
구현, 시험, 1에폭 스모크까지만 하고 넘긴다.

배선을 `a3_train_wm811k_cls.train_one_epoch` 안에 둔 이유:
**cosine 일정, val 기반 체크포인트 선택, a4 평가 연결이 이미 거기 있고 시험돼 있다.**
새 학습 스크립트를 따로 쓰면 그것들을 다시 만들어야 하고, 급하게 쓴 학습 루프가
조용히 틀리는 것이 이 프로젝트에서 가장 비싼 사고다.

**가장 중요한 시험은 `unlabeled=None` 일 때 동작이 조금도 안 바뀐다는 것**이다.
바뀌면 기존 체크포인트 42개의 재현이 깨진다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a3_train_wm811k_cls import build_model, train_one_epoch  # noqa: E402


def _wafers(n, size=32, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.hypot(yy - (size - 1) / 2, xx - (size - 1) / 2)
    base = np.where(d <= size // 2 - 2, 1, 0).astype(np.uint8)
    out = np.repeat(base[None], n, 0)
    out[(rng.random(out.shape) < 0.08) & (out > 0)] = 2
    return out


def _fresh(seed=0):
    torch.manual_seed(seed)
    return build_model(num_classes=3)


def _run(unlabeled=None, **kw):
    X = _wafers(24)
    y = np.arange(24) % 3
    m = _fresh()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    return train_one_epoch(m, X, y, opt, batch_size=8, device="cpu",
                           rng=np.random.default_rng(0), unlabeled=unlabeled, **kw), m


def test_no_unlabeled_is_bit_identical_to_before():
    """**기존 경로가 조금도 안 바뀌어야 한다.** 바뀌면 체크포인트 42개가 재현 안 된다."""
    l1, m1 = _run(unlabeled=None)
    l2, m2 = _run(unlabeled=None)
    assert l1 == l2
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(a, b)


def test_unlabeled_changes_the_update():
    """미라벨을 주면 실제로 다른 자리에 도달해야 한다 — 안 그러면 손실이 안 걸린 것이다."""
    _, m0 = _run(unlabeled=None)
    _, m1 = _run(unlabeled=_wafers(48, seed=1), mu=2, tau=0.0)
    same = all(torch.equal(a, b) for a, b in zip(m0.parameters(), m1.parameters()))
    assert not same


def test_impossible_threshold_makes_it_a_no_op():
    """tau 가 1 을 넘으면 통과하는 표본이 없으므로 라벨만 쓴 것과 같아야 한다.

    이것이 일관성 항이 **마스크를 정말로 존중하는지**를 가른다.
    """
    l0, m0 = _run(unlabeled=None)
    l1, m1 = _run(unlabeled=_wafers(48, seed=1), mu=2, tau=1.01)
    assert l0 == l1
    for a, b in zip(m0.parameters(), m1.parameters()):
        assert torch.equal(a, b)


def test_loss_is_finite_and_positive():
    loss, _ = _run(unlabeled=_wafers(48, seed=2), mu=3, tau=0.5)
    assert np.isfinite(loss) and loss > 0


def test_mu_controls_how_many_unlabeled_are_drawn():
    """mu 가 크면 미라벨을 더 본다 — 결과가 달라져야 한다."""
    _, m1 = _run(unlabeled=_wafers(64, seed=3), mu=1, tau=0.0)
    _, m2 = _run(unlabeled=_wafers(64, seed=3), mu=4, tau=0.0)
    same = all(torch.equal(a, b) for a, b in zip(m1.parameters(), m2.parameters()))
    assert not same


def test_it_runs_when_unlabeled_is_smaller_than_the_request():
    """미라벨이 요청량보다 적어도 죽지 않아야 한다(마지막 배치에서 실제로 생긴다)."""
    loss, _ = _run(unlabeled=_wafers(5, seed=4), mu=8, tau=0.0)
    assert np.isfinite(loss)
