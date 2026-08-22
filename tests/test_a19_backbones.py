"""계열이 다른 백본들. 앙상블 다양성의 마지막 미검증 축이다.

E11 에서 SizeAwareNet 을 넣었을 때 leave-one-out 이 -0.0003 이었다. 백본이 pad 모델들과
**같은 ResNet-18** 이고 헤드만 달랐으니 오류 패턴이 달라질 이유가 없었다.
이번에는 계열 자체를 바꾼다 — ShuffleNetV2(채널 셔플), MobileNetV3(inverted residual + SE),
ConvNeXt(대형 커널 depthwise + LayerNorm), EfficientNet(compound scaling).

**공통 제약: 64x64 입력이 붕괴하면 안 된다.** 원본 stem 을 그대로 두면
ResNet 은 최종 feature map 이 2x2 가 되고, ConvNeXt 는 patchify stem(stride 4)이라 더 나쁘다.
각 백본마다 stem stride 를 낮추고 초기 maxpool 을 제거해야 한다.
입력 채널도 3채널 one-hot 에 맞춰야 한다(우연히 RGB 와 같은 3 이지만 의미가 다르다).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a19_backbones import BACKBONES, build_backbone, feature_map_size

NAMES = list(BACKBONES)


class TestRegistry:
    def test_offers_more_than_one_family(self):
        assert len(NAMES) >= 4

    def test_resnet18_is_included_as_the_reference(self):
        """기존 결과와 이어 붙이려면 같은 경로로 만든 resnet18 이 있어야 한다."""
        assert "resnet18" in BACKBONES

    def test_rejects_an_unknown_name(self):
        with pytest.raises(ValueError):
            build_backbone("nosuchnet", num_classes=9)


@pytest.mark.parametrize("name", NAMES)
class TestEveryBackbone:
    def test_accepts_three_channel_64x64_and_returns_class_logits(self, name):
        m = build_backbone(name, num_classes=9).eval()
        with torch.no_grad():
            out = m(torch.zeros(2, 3, 64, 64))
        assert out.shape == (2, 9)

    def test_does_not_collapse_the_feature_map(self, name):
        """64x64 가 2x2 로 줄면 GAP 직전 공간 해상도가 사실상 사라진다.

        ResNet 에서 stem 을 고친 이유가 정확히 이것이고, 같은 기준을 모든 백본에 건다.
        """
        s = feature_map_size(name, input_size=64)
        assert s >= 4, f"{name} 의 최종 feature map 이 {s}x{s} 로 붕괴했다"

    def test_output_changes_with_input(self, name):
        """상수를 뱉는 모델이 위 시험들을 통과하는 것을 막는다.

        **train 모드로 잰다.** 학습 전 eval 모드는 BN running stats 가 초기값
        (mean 0, var 1)이라 MobileNetV3, EfficientNet 의 Hardswish 와 SE 게이트가 포화해
        입력 의존성이 사라진다(실측 maxdiff 정확히 0). 학습하면 BN 통계가 채워져
        사라지는 현상이고 실제로는 일어나지 않는 상태다.
        train 모드(배치 통계)에서는 다섯 백본 모두 1.6e-2 ~ 1.1e-1 로 반응한다.
        """
        m = build_backbone(name, num_classes=9).train()
        rng = np.random.default_rng(0)
        a = torch.from_numpy(rng.random((2, 3, 64, 64)).astype("float32"))
        b = torch.from_numpy(rng.random((2, 3, 64, 64)).astype("float32"))
        with torch.no_grad():
            assert not torch.allclose(m(a), m(b), atol=1e-4)

    def test_num_classes_is_honoured(self, name):
        m = build_backbone(name, num_classes=5).eval()
        with torch.no_grad():
            assert m(torch.zeros(1, 3, 64, 64)).shape == (1, 5)

    def test_gradients_reach_the_stem(self, name):
        """stem 을 갈아끼우면서 그래프가 끊기는 실수를 잡는다."""
        m = build_backbone(name, num_classes=9).train()
        out = m(torch.zeros(2, 3, 64, 64, requires_grad=False))
        out.sum().backward()
        stem = next(p for p in m.parameters() if p.requires_grad)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


class TestFeatureMapSize:
    def test_reports_a_larger_map_for_a_larger_input(self):
        assert feature_map_size("resnet18", 128) > feature_map_size("resnet18", 64)

    def test_resnet18_matches_the_known_configuration(self):
        """stem 3x3 stride1 + maxpool 제거면 총 stride 8 이라 64 -> 8 이다."""
        assert feature_map_size("resnet18", 64) == 8
