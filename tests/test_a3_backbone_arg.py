"""백본을 바꿔 학습한 체크포인트는 되읽을 때도 같은 백본으로 지어야 한다.

`build_model` 은 지금까지 ResNet-18 만 만들었고 a4(평가), a6(로짓 캐시), a17(TTA) 이
전부 그것을 전제로 체크포인트를 싣는다. 백본을 늘리면서 그 경로를 같이 열지 않으면
E18 로 학습한 모델을 평가할 수도, 앙상블에 넣을 수도 없다.

기본값은 반드시 `resnet18` 이어야 한다 — 기존 13개 체크포인트가 전부 그것이고
호출부 여섯 곳이 인자 없이 부르고 있다.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a3_train_wm811k_cls import build_model


class TestBackboneArgument:
    def test_defaults_to_resnet18(self):
        """기존 호출부가 전부 인자 없이 부른다. 기본값이 바뀌면 과거 체크포인트가 안 실린다."""
        m = build_model(num_classes=9)
        assert type(m).__name__ == "ResNet"

    def test_can_build_another_family(self):
        m = build_model(num_classes=9, backbone="shufflenet_v2")
        assert type(m).__name__ != "ResNet"

    def test_rejects_an_unknown_backbone(self):
        with pytest.raises(ValueError):
            build_model(num_classes=9, backbone="nosuchnet")

    @pytest.mark.parametrize("name", ["resnet18", "shufflenet_v2", "mobilenet_v3",
                                      "convnext_tiny", "efficientnet_b0"])
    def test_state_dict_round_trips(self, name):
        """학습 -> 저장 -> 평가 경로가 성립하는지가 요점이다.

        **eval 모드로 잰다.** a4(평가)와 a6(로짓 캐시)가 체크포인트를 싣는 방식이 그것이고,
        train 모드는 MobileNetV3, EfficientNet 의 classifier Dropout 때문에
        같은 가중치라도 출력이 달라져 가중치 전달 여부를 잴 수 없다.
        """
        a = build_model(num_classes=9, backbone=name)
        b = build_model(num_classes=9, backbone=name)
        b.load_state_dict(a.state_dict())
        for k, v in a.state_dict().items():
            assert torch.equal(v, b.state_dict()[k]), f"{k} 가 전달되지 않았다"
        a.eval(); b.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            assert torch.allclose(a(x), b(x), atol=1e-5)

    def test_pretrained_is_still_available_for_resnet18(self):
        """사전학습 토글은 resnet18 경로에만 있었고 그대로 살아 있어야 한다."""
        import inspect
        assert "pretrained" in inspect.signature(build_model).parameters
