"""준지도 학습의 순수 부분 (E25, 18차에 구현만. 학습은 다음 사이클).

반증 조건은 `docs/experiments/candidate/semi_supervised.md` 에 **실행 전에** 박았다.
여기 담긴 것은 학습이 없는 순수 계산이라 **정답을 손으로 적을 수 있다.**

**18차는 이 실험을 돌리지 않는다.** 마감 안에 3 seed 가 안 들어간다.
구현과 시험, 그리고 **에폭당 실제 비용 측정**까지만 하고 넘긴다 —
사전 등록 문서가 비용을 "모른다" 로 남겨 뒀기 때문이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a26_semisup import consistency_loss, pseudo_labels  # noqa: E402


def test_only_confident_samples_are_kept():
    """임계 미만은 버린다. 이것이 FixMatch 의 전부다."""
    p = torch.tensor([[0.97, 0.01, 0.02],
                      [0.50, 0.30, 0.20],
                      [0.02, 0.96, 0.02]])
    lab, mask = pseudo_labels(p, tau=0.95)
    assert mask.tolist() == [True, False, True]
    assert lab[mask].tolist() == [0, 1]


def test_nothing_is_kept_when_all_below_threshold():
    """전부 애매하면 미라벨 손실이 0 이어야 한다 — 초기 학습에서 실제로 일어난다."""
    p = torch.full((5, 4), 0.25)
    lab, mask = pseudo_labels(p, tau=0.95)
    assert mask.sum().item() == 0


def test_label_is_the_argmax():
    p = torch.tensor([[0.1, 0.2, 0.7]])
    lab, mask = pseudo_labels(p, tau=0.5)
    assert lab.tolist() == [2] and mask.tolist() == [True]


def test_threshold_is_inclusive_at_the_boundary():
    """경계값 처리를 박아 둔다 — 안 박으면 tau 를 바꿀 때 조용히 달라진다."""
    p = torch.tensor([[0.95, 0.05]])
    _, mask = pseudo_labels(p, tau=0.95)
    assert mask.tolist() == [True]


def test_probabilities_are_required_not_logits():
    """로짓을 넘기는 실수를 조용히 통과시키지 않는다."""
    with pytest.raises(ValueError):
        pseudo_labels(torch.tensor([[3.0, -1.0]]), tau=0.95)


# --- 일관성 손실 -------------------------------------------------------------

def test_loss_is_zero_when_nothing_is_confident():
    logits = torch.randn(4, 3)
    mask = torch.zeros(4, dtype=torch.bool)
    lab = torch.zeros(4, dtype=torch.long)
    assert consistency_loss(logits, lab, mask).item() == pytest.approx(0.0)


def test_loss_ignores_unconfident_rows():
    """버린 행이 손실에 끼면 애매한 표본이 학습을 끌고 간다."""
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    lab = torch.tensor([0, 0])
    both = consistency_loss(logits, lab, torch.tensor([True, True]))
    first = consistency_loss(logits, lab, torch.tensor([True, False]))
    assert first.item() < both.item()


def test_loss_normalises_by_batch_not_by_kept_count():
    """**분모는 배치 크기다.** 남은 개수로 나누면 초기에 한 표본이 손실을 지배한다.

    FixMatch 가 그렇게 정의돼 있고, 이 선택이 학습 안정성을 가른다.
    """
    logits = torch.tensor([[0.0, 5.0], [0.0, 5.0], [0.0, 5.0], [0.0, 5.0]])
    lab = torch.zeros(4, dtype=torch.long)
    one = consistency_loss(logits, lab, torch.tensor([True, False, False, False]))
    two = consistency_loss(logits, lab, torch.tensor([True, True, False, False]))
    assert two.item() == pytest.approx(2 * one.item(), rel=1e-5)


def test_gradient_does_not_flow_into_the_pseudo_label():
    """의사 라벨은 상수여야 한다. 여기로 기울기가 흐르면 자기강화가 폭주한다."""
    p = torch.softmax(torch.randn(6, 3), 1).requires_grad_(True)
    lab, mask = pseudo_labels(p, tau=0.3)
    assert not lab.requires_grad
