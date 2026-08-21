"""BN 통계 타겟 적응 테스트.

가중치는 그대로 두고 BatchNorm 의 running mean, var 만 타겟 데이터로 다시 잰다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a9_bn_adapt import adapt_bn_stats, count_bn_layers


def tiny_net():
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4),
                         nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(4, 2))


def shifted_data(n=64, shift=3.0, seed=0):
    """불량 die 비율이 shift 에 따라 달라지는 합성 웨이퍼맵.

    임계값은 0 으로 **고정**한다. shift 와 함께 움직이면 이진화 결과의 분포가
    shift 와 무관해져 분포 이동을 만들지 못한다.
    """
    rng = np.random.default_rng(seed)
    return (rng.normal(shift, 1.0, size=(n, 8, 8)) > 0).astype(np.uint8) * 2


class TestCount:
    def test_finds_batchnorm_layers(self):
        assert count_bn_layers(tiny_net()) == 1

    def test_reports_zero_when_none(self):
        assert count_bn_layers(nn.Sequential(nn.Linear(2, 2))) == 0


class TestAdapt:
    def test_running_stats_change(self):
        m = tiny_net()
        bn = m[1]
        before = bn.running_mean.clone()
        adapt_bn_stats(m, shifted_data(), batch_size=16, device="cpu")
        assert not torch.allclose(before, bn.running_mean)

    def test_weights_are_untouched(self):
        """가중치를 건드리면 이건 더 이상 '통계만 다시 재기' 가 아니다."""
        m = tiny_net()
        w0 = m[0].weight.detach().clone()
        w1 = m[5].weight.detach().clone()
        g = m[1].weight.detach().clone()
        adapt_bn_stats(m, shifted_data(), batch_size=16, device="cpu")
        assert torch.allclose(w0, m[0].weight)
        assert torch.allclose(w1, m[5].weight)
        assert torch.allclose(g, m[1].weight)

    def test_model_is_left_in_eval_mode(self):
        m = tiny_net()
        adapt_bn_stats(m, shifted_data(), batch_size=16, device="cpu")
        assert not m.training

    def test_adapting_to_same_data_twice_is_stable(self):
        m = tiny_net()
        x = shifted_data()
        adapt_bn_stats(m, x, batch_size=16, device="cpu")
        first = m[1].running_mean.clone()
        adapt_bn_stats(m, x, batch_size=16, device="cpu")
        assert torch.allclose(first, m[1].running_mean, atol=1e-5)

    def test_different_distributions_give_different_stats(self):
        a, b = tiny_net(), tiny_net()
        b.load_state_dict(a.state_dict())
        adapt_bn_stats(a, shifted_data(shift=-2.0), batch_size=16, device="cpu")
        adapt_bn_stats(b, shifted_data(shift=2.0), batch_size=16, device="cpu")
        assert not torch.allclose(a[1].running_mean, b[1].running_mean)

    def test_no_bn_model_is_a_noop(self):
        m = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
        w = m[1].weight.detach().clone()
        adapt_bn_stats(m, shifted_data(), batch_size=16, device="cpu")
        assert torch.allclose(w, m[1].weight)

    def test_rejects_empty_data(self):
        with pytest.raises(ValueError):
            adapt_bn_stats(tiny_net(), np.zeros((0, 8, 8), dtype=np.uint8),
                           batch_size=16, device="cpu")
