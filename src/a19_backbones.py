"""계열이 다른 백본들을 64x64 명목형 입력에 맞게 고쳐 만든다.

E11 에서 SizeAwareNet(백본은 같은 ResNet-18, 헤드만 다름)의 앙상블 기여가 -0.0003 이었다.
백본이 같으면 오류 패턴이 달라질 이유가 없다는 뜻이므로, 이번에는 계열 자체를 바꾼다.
E14 에서 확인된 것은 **입력 표현 다양성**이 앙상블을 밀어올린다는 것이고,
구조 다양성은 아직 제대로 시험된 적이 없다.

## 공통 수정

원본 stem 은 ImageNet 224x224 를 전제로 초기에 크게 줄인다. 64x64 에 그대로 쓰면
최종 feature map 이 2x2 이하로 붕괴해 GAP 직전 공간 해상도가 사라진다.
따라서 각 백본마다 **stem stride 를 1 로 낮추고 초기 maxpool 을 제거**한다.
ConvNeXt 는 patchify stem(4x4 stride 4)이라 3x3 stride 1 로 통째로 갈아끼운다.

입력은 3채널 one-hot(배경/정상/불량)이다. 채널 수가 우연히 RGB 와 같지만 의미가 다르므로
ImageNet 사전학습은 쓰지 않는다(그리고 이 데이터셋 규모에서 전이 이득이 남지 않는다는
근거가 `docs/research/encoder_selection/README.md` 에 있다).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

NUM_CATEGORIES = 3


def _resnet18(num_classes: int) -> nn.Module:
    m = tvm.resnet18(weights=None)
    m.conv1 = nn.Conv2d(NUM_CATEGORIES, 64, 3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _shufflenet(num_classes: int) -> nn.Module:
    m = tvm.shufflenet_v2_x1_0(weights=None)
    old = m.conv1[0]
    m.conv1[0] = nn.Conv2d(NUM_CATEGORIES, old.out_channels, 3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _mobilenet_v3(num_classes: int) -> nn.Module:
    m = tvm.mobilenet_v3_large(weights=None)
    old = m.features[0][0]
    m.features[0][0] = nn.Conv2d(NUM_CATEGORIES, old.out_channels, 3,
                                 stride=1, padding=1, bias=False)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def _convnext(num_classes: int) -> nn.Module:
    m = tvm.convnext_tiny(weights=None)
    old = m.features[0][0]
    # patchify stem(4x4 stride 4)을 3x3 stride 1 로 바꾼다. 총 stride 32 -> 8.
    m.features[0][0] = nn.Conv2d(NUM_CATEGORIES, old.out_channels, 3, stride=1, padding=1)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def _efficientnet(num_classes: int) -> nn.Module:
    m = tvm.efficientnet_b0(weights=None)
    old = m.features[0][0]
    m.features[0][0] = nn.Conv2d(NUM_CATEGORIES, old.out_channels, 3,
                                 stride=1, padding=1, bias=False)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


BACKBONES = {
    "resnet18": _resnet18,
    "shufflenet_v2": _shufflenet,
    "mobilenet_v3": _mobilenet_v3,
    "convnext_tiny": _convnext,
    "efficientnet_b0": _efficientnet,
}


def build_backbone(name: str, num_classes: int = 9, pretrained: bool = False) -> nn.Module:
    """이름으로 백본을 만든다. 전부 3채널 64x64 입력에 맞게 stem 이 수정돼 있다."""
    if name not in BACKBONES:
        raise ValueError(f"모르는 백본: {name}. 가능한 것: {sorted(BACKBONES)}")
    if pretrained:
        raise ValueError("ImageNet 사전학습은 쓰지 않는다. 입력이 3채널 one-hot 명목형이다.")
    return BACKBONES[name](num_classes)


_FEATURES = {
    "resnet18": lambda m: nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                        m.layer1, m.layer2, m.layer3, m.layer4),
    "shufflenet_v2": lambda m: nn.Sequential(m.conv1, m.maxpool, m.stage2, m.stage3,
                                             m.stage4, m.conv5),
    "mobilenet_v3": lambda m: m.features,
    "convnext_tiny": lambda m: m.features,
    "efficientnet_b0": lambda m: m.features,
}


def feature_map_size(name: str, input_size: int = 64) -> int:
    """전역 풀링 직전의 공간 해상도. 붕괴 여부를 재기 위한 것이다."""
    m = build_backbone(name, num_classes=9).eval()
    feat = _FEATURES[name](m)
    with torch.no_grad():
        out = feat(torch.zeros(1, NUM_CATEGORIES, input_size, input_size))
    return int(out.shape[-1])
