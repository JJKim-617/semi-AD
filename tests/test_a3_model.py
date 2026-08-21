"""a3 모델과 입력 표현 테스트.

검증하는 성질:
- one-hot 인코딩이 명목형 {0,1,2} 를 3채널로 정확히 편다
- stem 수정으로 64x64 에서 feature map 이 2x2 로 붕괴하지 않는다
- 사전학습 토글이 실제로 가중치를 바꾼다
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a3_train_wm811k_cls import build_model, to_onehot


class TestOneHot:
    def test_maps_three_categories_to_three_channels(self):
        x = np.array([[[0, 1], [2, 0]]], dtype=np.uint8)  # (1,2,2)
        out = to_onehot(x)
        assert out.shape == (1, 3, 2, 2)

    def test_each_pixel_activates_exactly_one_channel(self):
        x = np.array([[[0, 1], [2, 0]]], dtype=np.uint8)
        out = to_onehot(x)
        assert torch.allclose(out.sum(dim=1), torch.ones(1, 2, 2))

    def test_channel_index_matches_category_value(self):
        x = np.array([[[0, 1, 2]]], dtype=np.uint8)  # (1,1,3)
        out = to_onehot(x)
        assert out[0, 0, 0, 0] == 1.0   # 배경 -> ch0
        assert out[0, 1, 0, 1] == 1.0   # 정상 -> ch1
        assert out[0, 2, 0, 2] == 1.0   # 불량 -> ch2

    def test_output_is_float_for_conv_input(self):
        x = np.zeros((1, 2, 2), dtype=np.uint8)
        assert to_onehot(x).dtype == torch.float32

    def test_rejects_values_outside_zero_one_two(self):
        x = np.array([[[0, 3]]], dtype=np.uint8)
        with pytest.raises(ValueError):
            to_onehot(x)


class TestModel:
    def test_outputs_one_logit_per_class(self):
        m = build_model(num_classes=9, pretrained=False)
        out = m(torch.zeros(2, 3, 64, 64))
        assert out.shape == (2, 9)

    def test_stem_is_modified_for_low_resolution(self):
        m = build_model(num_classes=9, pretrained=False)
        assert m.conv1.kernel_size == (3, 3)
        assert m.conv1.stride == (1, 1)
        assert isinstance(m.maxpool, torch.nn.Identity)

    def test_feature_map_does_not_collapse_at_64px(self):
        """표준 stem 이면 64x64 -> 2x2 로 붕괴한다. 수정 후엔 8x8 이어야 한다."""
        m = build_model(num_classes=9, pretrained=False)
        feats = torch.nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
        )(torch.zeros(1, 3, 64, 64))
        assert feats.shape[-2:] == (8, 8)

    def test_pretrained_flag_changes_weights(self):
        a = build_model(num_classes=9, pretrained=False)
        b = build_model(num_classes=9, pretrained=True)
        assert not torch.allclose(a.layer1[0].conv1.weight, b.layer1[0].conv1.weight)

    def test_pretrained_stem_is_freshly_initialized(self):
        """stem 은 형상이 달라 사전학습 가중치를 쓸 수 없다. 두 번 만들면 서로 달라야 한다."""
        a = build_model(num_classes=9, pretrained=True)
        b = build_model(num_classes=9, pretrained=True)
        assert not torch.allclose(a.conv1.weight, b.conv1.weight)
        assert torch.allclose(a.layer1[0].conv1.weight, b.layer1[0].conv1.weight)
