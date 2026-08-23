"""합성곱 패딩 방식 (E24, 18차).

기전 시험이다 — 제로 패딩이 모델에게 **절대 위치**를 주는 출처인지 직접 끈다.
반증 조건은 `docs/experiments/candidate/padding_position_leak.md` 에 실행 전에 박았다.

**가장 중요한 시험은 기본값이 `zeros` 로 남는다는 것**이다.
기본값을 바꾸면 기존 체크포인트 42개의 결과를 되읽을 수 없다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a3_train_wm811k_cls import build_model, to_onehot  # noqa: E402


def conv_modes(model):
    return {m.padding_mode for m in model.modules()
            if isinstance(m, nn.Conv2d) and any(p > 0 for p in m.padding)}


def test_default_is_zeros_so_old_checkpoints_still_read():
    """**기본값을 바꾸면 과거 결과를 못 읽는다.** 이 시험이 그것을 막는다."""
    assert conv_modes(build_model()) == {"zeros"}


def test_reflect_reaches_every_padded_conv():
    """한 층이라도 zeros 로 남으면 개입이 새어 실험이 무의미해진다."""
    assert conv_modes(build_model(padding_mode="reflect")) == {"reflect"}


def test_reflect_reaches_every_padded_conv_on_other_backbones():
    """백본을 바꿔도 같아야 한다. a19 백본은 stem 이 따로 수정돼 있다."""
    assert conv_modes(build_model(backbone="shufflenet_v2",
                                  padding_mode="reflect")) == {"reflect"}


def test_unknown_mode_is_refused():
    """오타가 조용히 zeros 로 떨어지면 대조군과 구분이 안 된다."""
    with pytest.raises(ValueError):
        build_model(padding_mode="reflective")


def test_shape_is_unchanged():
    x = to_onehot(np.zeros((2, 64, 64), np.uint8))
    assert build_model(padding_mode="reflect")(x).shape == (2, 9)


def test_it_actually_changes_the_forward_pass():
    """같은 가중치인데 출력이 같으면 패딩 방식이 실제로 안 걸린 것이다."""
    torch.manual_seed(0)
    a = build_model()
    b = build_model(padding_mode="reflect")
    b.load_state_dict(a.state_dict())
    a.eval(), b.eval()
    rng = np.random.default_rng(0)
    x = to_onehot(rng.integers(0, 3, (4, 64, 64)).astype(np.uint8))
    with torch.no_grad():
        assert not torch.allclose(a(x), b(x), atol=1e-6)


def test_weights_are_identical_so_only_padding_differs():
    """개입이 패딩 하나여야 한다. 파라미터 수나 모양이 달라지면 교란이다."""
    a, b = build_model(), build_model(padding_mode="reflect")
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    assert all(sa[k].shape == sb[k].shape for k in sa)
    assert (sum(p.numel() for p in a.parameters())
            == sum(p.numel() for p in b.parameters()))
