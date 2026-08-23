"""위치, 크기, 회전에 불변인 웨이퍼 기술자 (18차).

`docs/research/label_ambiguity/README.md` 가 남긴 다음 할 일 1번:

> **더 나은 거리.** 위치, 크기에 불변인 기술자로 이웃을 찾아야 한다.
> 해밍 거리는 불량 다이 개수에 끌려간다는 것이 확인됐다.

여기 담긴 것은 학습이 없는 순수 계산이라 **정답을 손으로 적을 수 있다.**
불변성 셋을 시험으로 박는다 — 그것이 이 기술자의 존재 이유다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a25_wafer_descriptor import describe  # noqa: E402


def ring(size=64, r_in=0.7, r_out=0.9, seed=0, disc=0.95):
    """가장자리 링에 불량이 있는 합성 웨이퍼 (Edge-Ring 흉내).

    `disc` 를 줄이면 캔버스에 여백이 생겨 **이동해도 잘리지 않는다.**
    """
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    r = np.hypot(yy - c, xx - c) / (size / 2 * disc)
    x = np.where(r <= 1.0, 1, 0).astype(np.uint8)
    x[(r >= r_in) & (r <= r_out)] = 2
    return x[None]


def blob(size=64, cy=0.3, cx=0.3, rad=0.12):
    """한쪽에 뭉친 불량 (Loc 흉내)."""
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    r = np.hypot(yy - c, xx - c) / (size / 2)
    x = np.where(r <= 0.95, 1, 0).astype(np.uint8)
    d = np.hypot(yy / size - cy, xx / size - cx)
    x[(d <= rad) & (x > 0)] = 2
    return x[None]


def shift(x, dy, dx):
    out = np.zeros_like(x)
    h, w = x.shape[1:]
    ys, ye = max(0, dy), min(h, h + dy)
    xs, xe = max(0, dx), min(w, w + dx)
    out[0, ys:ye, xs:xe] = x[0, ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def test_descriptor_is_finite_and_fixed_length():
    d = describe(ring())
    assert d.ndim == 2 and d.shape[0] == 1
    assert np.isfinite(d).all()
    assert describe(blob()).shape[1] == d.shape[1]


def test_translation_invariance():
    """캔버스 안 위치는 이 표현에서 잡음 변수다. 기술자가 거기에 반응하면 안 된다."""
    # 웨이퍼를 캔버스보다 작게 만들어 **이동해도 잘리지 않게** 한다.
    # (잘리면 다이 영역 자체가 바뀌므로 기술자가 달라지는 게 맞다.)
    x = ring(disc=0.7)
    a = describe(x)[0]
    b = describe(shift(x, 5, -4))[0]
    assert np.abs(a - b).max() < 0.05, f"이동에 {np.abs(a - b).max():.3f} 만큼 반응했다"


def test_size_invariance():
    """웨이퍼 크기가 다르면 라벨이 달라지지 않는다. 크기에 불변이어야 한다."""
    a = describe(ring(size=48))[0]
    b = describe(ring(size=96))[0]
    assert np.abs(a - b).max() < 0.10, (
        f"크기에 {np.abs(a - b).max():.3f} 만큼 반응했다 — "
        "칸 개수로 센 항이 남아 있으면 여기서 걸린다")


def test_rotation_invariance():
    """Edge-Loc, Loc, Scratch 의 방향은 임의다. 90도 회전에 불변이어야 한다."""
    x = blob()
    a = describe(x)[0]
    b = describe(np.rot90(x[0])[None])[0]
    assert np.abs(a - b).max() < 0.05


def test_it_separates_a_ring_from_a_blob():
    """불변이기만 하고 아무것도 구분 못 하면 쓸모가 없다."""
    r, b = describe(ring())[0], describe(blob())[0]
    same = np.abs(describe(ring())[0] - describe(ring(seed=1))[0]).max()
    assert np.abs(r - b).max() > 5 * max(same, 1e-6)


def test_fail_rate_is_a_separate_field_so_it_can_be_controlled():
    """불량 다이 비율은 **따로** 나와야 한다.

    해밍 거리가 불량 개수에 끌려간 것이 앞선 조사의 교란이었다.
    개수를 층으로 통제하려면 기술자 안에 섞여 있으면 안 된다.
    """
    from a25_wafer_descriptor import fail_rate
    x = ring()
    assert 0.0 < fail_rate(x)[0] < 1.0
    dense = ring(r_in=0.3, r_out=0.95)
    assert fail_rate(dense)[0] > fail_rate(x)[0]


def test_empty_wafer_does_not_crash_and_is_marked():
    """다이가 없으면 기술자가 정의되지 않는다. NaN 으로 표시하고 넘긴다."""
    d = describe(np.zeros((1, 32, 32), np.uint8))
    assert np.isnan(d).all()
