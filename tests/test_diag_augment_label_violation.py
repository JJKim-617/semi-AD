"""증강이 라벨을 깨뜨리는 정도를 재는 도구 (18차).

`diag_leakage_mechanism` 이 보인 것: 추가 증강 넷 중 **translate 만 none -> 결함 누출을
줄이고 나머지 셋은 늘린다**(noise 는 3배로). 여기서 세우는 설명은
**"none 웨이퍼를 결함처럼 보이게 만드는 증강은 누출을 늘린다"** 이고,
그것을 재는 두 자를 시험한다. 둘 다 순수 계산이라 정답을 손으로 적을 수 있다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diag_augment_label_violation import (  # noqa: E402
    flip_rate,
    structural_change,
)


# --- 구조 변화 (모델 없이 재는 자) -------------------------------------------

def test_translate_adds_and_removes_nothing():
    """강체 이동은 불량 다이를 만들지도 지우지도 않는다. 이것이 핵심 대비다."""
    x = np.zeros((1, 8, 8), np.uint8); x[0, 2:6, 2:6] = 1; x[0, 3, 3] = 2
    shifted = np.zeros_like(x); shifted[0, 3:7, 1:5] = 1; shifted[0, 4, 2] = 2
    c = structural_change(x, shifted)
    assert c["fail_added_frac"] == pytest.approx(0.0)
    assert c["fail_removed_frac"] == pytest.approx(0.0)
    assert c["die_removed_frac"] == pytest.approx(0.0)


def test_added_fail_is_counted_as_a_fraction_of_dies():
    """정상 -> 불량으로 뒤집힌 칸의 비율. 분모는 원본의 다이 칸 수다."""
    x = np.ones((1, 1, 10), np.uint8)          # 다이 10칸, 전부 정상
    a = x.copy(); a[0, 0, :2] = 2              # 2칸을 불량으로
    c = structural_change(x, a)
    assert c["fail_added_frac"] == pytest.approx(0.2)
    assert c["fail_removed_frac"] == pytest.approx(0.0)


def test_removed_fail_and_removed_die_are_separate_numbers():
    """불량이 정상이 된 것과 다이가 사라진 것은 다른 사건이다. 섞으면 안 된다."""
    x = np.array([[[2, 2, 1, 1]]], np.uint8)
    a = np.array([[[1, 2, 0, 1]]], np.uint8)   # 1칸 불량->정상, 1칸 다이 제거
    c = structural_change(x, a)
    assert c["fail_removed_frac"] == pytest.approx(0.25)
    assert c["die_removed_frac"] == pytest.approx(0.25)
    assert c["fail_added_frac"] == pytest.approx(0.0)


def test_structural_change_needs_matching_shapes():
    """모양이 다르면 조용히 방송하지 말고 거절한다."""
    with pytest.raises(ValueError):
        structural_change(np.ones((1, 4, 4), np.uint8), np.ones((1, 8, 8), np.uint8))


def test_no_dies_gives_nan_not_zero():
    """다이가 없으면 비율이 정의되지 않는다. 0 은 '아무 일도 없었다' 와 헷갈린다."""
    z = np.zeros((1, 4, 4), np.uint8)
    assert np.isnan(structural_change(z, z)["fail_added_frac"])


# --- 뒤집힘 비율 (참조 모델로 재는 자) ---------------------------------------

def test_flip_rate_conditions_on_being_right_before():
    """원래 none 이라고 맞힌 것 중 증강 뒤 결함이 된 비율이다.

    원래도 틀렸던 웨이퍼는 분모에서 빠진다 — 안 그러면 참조 모델의 오류율이
    증강의 성질로 오인된다.
    """
    before = np.array([[9.0, 0], [9.0, 0], [0.0, 9]])   # 2장 맞음, 1장 원래 틀림
    after = np.array([[0.0, 9], [9.0, 0], [0.0, 9]])    # 맞았던 것 중 1장이 뒤집힘
    assert flip_rate(before, after, none_index=0) == pytest.approx(0.5)


def test_flip_rate_is_zero_when_nothing_changes():
    lg = np.array([[9.0, 0], [9.0, 0]])
    assert flip_rate(lg, lg, none_index=0) == pytest.approx(0.0)


def test_flip_rate_is_nan_when_nothing_was_right():
    """맞힌 것이 하나도 없으면 비율이 없다. 0 을 돌려주면 '완벽' 처럼 읽힌다."""
    lg = np.array([[0.0, 9], [0.0, 9]])
    assert np.isnan(flip_rate(lg, lg, none_index=0))


def test_flip_rate_ignores_recovery():
    """결함->none 으로 돌아온 것은 뒤집힘이 아니다. 방향이 있는 자다."""
    before = np.array([[0.0, 9], [9.0, 0]])
    after = np.array([[9.0, 0], [9.0, 0]])
    assert flip_rate(before, after, none_index=0) == pytest.approx(0.0)
