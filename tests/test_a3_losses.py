"""손실 함수 테스트.

focal loss 는 쉬운 샘플의 기여를 (1-p)^gamma 로 줄인다. 이 데이터셋은 93% 가 none 이라
대부분이 쉬운 샘플이고, 고정 배율 class weight 와 달리 샘플별로 조절돼 gradient 분산이 작다.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a3_train_wm811k_cls import FocalLoss


def test_gamma_zero_equals_cross_entropy():
    """gamma=0 이면 focal 은 정의상 CE 와 같아야 한다."""
    logits = torch.randn(8, 4)
    target = torch.randint(0, 4, (8,))
    focal = FocalLoss(gamma=0.0)(logits, target)
    ce = torch.nn.functional.cross_entropy(logits, target)
    assert torch.allclose(focal, ce, atol=1e-6)


def test_perfect_prediction_gives_near_zero_loss():
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    target = torch.tensor([0, 1])
    assert FocalLoss(gamma=2.0)(logits, target).item() < 1e-3


def test_downweights_easy_examples_relative_to_ce():
    """잘 맞춘 샘플에서 focal 손실은 CE 보다 작아야 한다. 이것이 focal 의 존재 이유다."""
    logits = torch.tensor([[3.0, -3.0]])
    target = torch.tensor([0])
    focal = FocalLoss(gamma=2.0)(logits, target)
    ce = torch.nn.functional.cross_entropy(logits, target)
    assert focal < ce


def test_hard_example_is_less_downweighted_than_easy_one():
    """어려운 샘플일수록 focal/CE 비율이 1 에 가까워야 한다."""
    def ratio(margin):
        logits = torch.tensor([[margin, -margin]])
        target = torch.tensor([0])
        ce = torch.nn.functional.cross_entropy(logits, target)
        return (FocalLoss(gamma=2.0)(logits, target) / ce).item()
    assert ratio(0.1) > ratio(3.0)


def test_larger_gamma_downweights_more():
    logits = torch.tensor([[3.0, -3.0]])
    target = torch.tensor([0])
    assert FocalLoss(gamma=5.0)(logits, target) < FocalLoss(gamma=1.0)(logits, target)


def test_alpha_weights_classes():
    """alpha 를 주면 해당 클래스의 손실이 커진다."""
    logits = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([1])
    plain = FocalLoss(gamma=2.0)(logits, target)
    weighted = FocalLoss(gamma=2.0, alpha=torch.tensor([1.0, 5.0]))(logits, target)
    assert weighted > plain


def test_gradient_flows():
    logits = torch.randn(4, 3, requires_grad=True)
    FocalLoss(gamma=2.0)(logits, torch.randint(0, 3, (4,))).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_rejects_negative_gamma():
    with pytest.raises(ValueError):
        FocalLoss(gamma=-1.0)
