"""가중치의 지수이동평균(EMA).

이 프로젝트 최대 불안정 요인은 seed 분산이고(pad 3 seed 범위 0.0495),
그 3/4 이 Scratch 붕괴 하나에서 나온다(F1 0.613 / 0.280 / 0.608).
자기지도 사전학습이 그 분산을 20배 줄였지만(0.0025) 평균도 함께 떨어뜨려 채택하지 못했다.

EMA 는 학습 궤적을 평균해 분산을 줄이는 표준 기법이고, 학습 비용이 사실상 0 이다
(추가 forward 도 backward 도 없다). 앙상블이 **여러 실행**을 평균하는 것이라면
EMA 는 **한 실행 안의 여러 시점**을 평균한다.

정확성의 핵심은 정수 버퍼다. BatchNorm 의 `num_batches_tracked` 는 int64 인데
여기에 소수 가중치를 곱하면 값이 망가진다. 실수 텐서만 평균하고 나머지는 복사해야 한다.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a20_ema import EMA


def tiny():
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4),
                         nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(4, 2))


def randomize(m, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn(p.shape, generator=g))
    return m


class TestDecayEndpoints:
    def test_decay_zero_tracks_the_model_exactly(self):
        m = randomize(tiny(), 0)
        ema = EMA(m, decay=0.0)
        randomize(m, 1)
        ema.update(m)
        for k, v in m.state_dict().items():
            if v.dtype.is_floating_point:
                assert torch.allclose(ema.shadow[k], v.float())

    def test_decay_one_never_moves(self):
        m = randomize(tiny(), 0)
        ema = EMA(m, decay=1.0)
        before = {k: v.clone() for k, v in ema.shadow.items()}
        randomize(m, 2)
        ema.update(m)
        for k in before:
            assert torch.allclose(ema.shadow[k], before[k])

    def test_rejects_decay_outside_the_unit_interval(self):
        with pytest.raises(ValueError):
            EMA(tiny(), decay=1.5)
        with pytest.raises(ValueError):
            EMA(tiny(), decay=-0.1)


class TestConvergence:
    def test_repeated_updates_approach_the_target(self):
        """모델이 한 값에 고정돼 있으면 shadow 가 거기로 수렴해야 한다."""
        m = randomize(tiny(), 0)
        ema = EMA(m, decay=0.9)
        target = randomize(tiny(), 5)
        m.load_state_dict(target.state_dict())
        first = None
        for i in range(200):
            ema.update(m)
            gap = max((ema.shadow[k] - v.float()).abs().max().item()
                      for k, v in m.state_dict().items() if v.dtype.is_floating_point)
            if first is None:
                first = gap
        assert gap < first * 1e-3, f"수렴하지 않았다: {first:.3e} -> {gap:.3e}"

    def test_a_higher_decay_moves_more_slowly(self):
        m = randomize(tiny(), 0)
        slow, fast = EMA(m, decay=0.99), EMA(m, decay=0.5)
        randomize(m, 3)
        slow.update(m); fast.update(m)
        k = "0.weight"
        cur = m.state_dict()[k].float()
        assert (slow.shadow[k] - cur).abs().sum() > (fast.shadow[k] - cur).abs().sum()


class TestIntegerBuffers:
    def test_integer_buffers_keep_their_dtype(self):
        """BatchNorm 의 num_batches_tracked 는 int64 다. 평균내면 망가진다."""
        m = tiny()
        ema = EMA(m, decay=0.9)
        for _ in range(3):
            m(torch.randn(2, 3, 8, 8))     # num_batches_tracked 를 올린다
            ema.update(m)
        out = tiny()
        ema.copy_to(out)
        for k, v in out.state_dict().items():
            if not v.dtype.is_floating_point:
                assert v.dtype == m.state_dict()[k].dtype

    def test_integer_buffers_follow_the_live_model(self):
        m = tiny().train()
        ema = EMA(m, decay=0.9)
        for _ in range(4):
            m(torch.randn(2, 3, 8, 8))
            ema.update(m)
        out = tiny()
        ema.copy_to(out)
        assert out.state_dict()["1.num_batches_tracked"].item() == \
            m.state_dict()["1.num_batches_tracked"].item()


class TestCopyTo:
    def test_writes_the_shadow_into_a_model(self):
        m = randomize(tiny(), 0)
        ema = EMA(m, decay=0.5)
        randomize(m, 7)
        ema.update(m)
        out = tiny()
        ema.copy_to(out)
        for k, v in out.state_dict().items():
            if v.dtype.is_floating_point:
                assert torch.allclose(v, ema.shadow[k], atol=1e-6)

    def test_does_not_disturb_the_live_model(self):
        m = randomize(tiny(), 0)
        ema = EMA(m, decay=0.5)
        randomize(m, 8)
        before = {k: v.clone() for k, v in m.state_dict().items()}
        ema.update(m)
        ema.copy_to(tiny())
        for k, v in m.state_dict().items():
            assert torch.equal(v, before[k])

    def test_running_stats_are_averaged_not_ignored(self):
        """BN running_mean 도 실수 버퍼다. 평균 대상에서 빠지면 EMA 가 반쪽이 된다."""
        m = tiny().train()
        ema = EMA(m, decay=0.5)
        for _ in range(3):
            m(torch.randn(4, 3, 8, 8) * 5 + 3)
            ema.update(m)
        assert ema.shadow["1.running_mean"].abs().sum() > 0


class TestDeviceMigration:
    """모델이 EMA 생성 뒤에 다른 장치로 옮겨가는 경우.

    a3 는 `build_model()` 로 CPU 에 모델을 만들고, 학습 루프가 첫 스텝에서 `.to(device)` 한다.
    EMA 를 그 사이에 만들면 shadow 는 CPU, 파라미터는 CUDA 가 되어 update 가 터진다.
    실제로 스모크 테스트에서 이 경로가 잡혔다.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 없음")
    def test_survives_the_model_moving_to_cuda(self):
        m = randomize(tiny(), 0)
        ema = EMA(m, decay=0.9)          # CPU 에서 생성
        m = m.cuda()                      # 그 뒤에 이동
        ema.update(m)                     # 터지면 안 된다
        out = tiny().cuda()
        ema.copy_to(out)
        for k, v in out.state_dict().items():
            assert v.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 없음")
    def test_still_averages_correctly_after_moving(self):
        m = randomize(tiny(), 0).cuda()
        ema = EMA(m, decay=0.0)           # decay 0 이면 현재 모델과 같아야 한다
        ema.update(m)
        sd = m.state_dict()
        for k, s in ema.shadow.items():
            assert torch.allclose(s.cpu(), sd[k].detach().float().cpu(), atol=1e-6)
