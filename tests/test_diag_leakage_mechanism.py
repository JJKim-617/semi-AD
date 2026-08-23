"""none -> 결함 누출의 기전을 가르는 도구 (18차).

17차가 "설명 없는 관측" 으로 남긴 것: **translate 만 누출의 총량을 줄인다**
(2,962 -> 1,190). 다른 기법은 누출의 **퍼짐**만 줄이고 **수준**은 못 줄였다.

여기 담긴 함수는 전부 학습이 없는 순수 계산이라 **정답을 손으로 적을 수 있다.**
아래는 회귀 방지가 아니라 정의 확인이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diag_leakage_mechanism import (  # noqa: E402
    fail_centroid_radius,
    leakage_by_class,
    none_bias_for_target_leak,
)

NONE = 0


# --- 누출 집계 ---------------------------------------------------------------

def test_leakage_counts_only_none_labelled_rows():
    """결함으로 라벨된 행은 누출이 아니다. 라벨이 none 인 것만 센다."""
    logits = np.array([
        [0.0, 9.0, 0, 0],   # y=none  -> 1 로 누출
        [9.0, 0.0, 0, 0],   # y=none  -> none, 안 샌다
        [0.0, 9.0, 0, 0],   # y=1     -> 맞은 것, 누출 아님
    ])
    y = np.array([NONE, NONE, 1])
    out = leakage_by_class(logits, y, none_index=NONE)
    assert out.tolist() == [1, 1, 0, 0]


def test_leakage_total_excludes_the_none_slot():
    """none 칸은 '지킨 것' 이지 누출이 아니다. 총 누출은 나머지의 합이다."""
    logits = np.array([[0.0, 1, 0, 5], [0.0, 1, 5, 0], [9.0, 1, 0, 0]])
    out = leakage_by_class(logits, y=np.zeros(3, int), none_index=NONE)
    assert out[NONE] == 1
    assert out[1:].sum() == 2


def test_leakage_with_no_none_rows_is_all_zero():
    """none 이 한 장도 없으면 없는 숫자를 만들지 않고 전부 0 을 돌려준다."""
    out = leakage_by_class(np.zeros((3, 4)), y=np.array([1, 2, 3]), none_index=NONE)
    assert out.sum() == 0


# --- 운영점 맞추기 (가설 C: 누출 감소가 그냥 확신도 하락인가) -----------------

def test_bias_zero_when_already_at_target():
    """이미 목표 누출이면 편향은 0 이다."""
    logits = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    y = np.zeros(3, int)
    assert none_bias_for_target_leak(logits, y, NONE, target=2) == pytest.approx(0.0, abs=1e-6)


def test_bias_pushes_leakage_down_to_target():
    """none 로짓에 편향을 더하면 누출이 목표까지 내려온다."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 4))
    y = np.zeros(500, int)
    before = leakage_by_class(logits, y, NONE)[1:].sum()
    b = none_bias_for_target_leak(logits, y, NONE, target=before // 3)
    shifted = logits.copy()
    shifted[:, NONE] += b
    after = leakage_by_class(shifted, y, NONE)[1:].sum()
    assert b > 0
    assert after <= before // 3


def test_bias_is_the_smallest_that_reaches_target():
    """조금 작은 편향으로는 목표에 못 미쳐야 한다 — 과하게 밀면 비교가 불공정해진다."""
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(400, 5))
    y = np.zeros(400, int)
    target = 50
    b = none_bias_for_target_leak(logits, y, NONE, target=target)
    lo = logits.copy(); lo[:, NONE] += b - 0.05
    assert leakage_by_class(lo, y, NONE)[1:].sum() > target


def test_bias_ignores_defect_rows_when_counting():
    """편향 탐색도 none 라벨 행만 본다."""
    logits = np.array([[0.0, 3.0], [0.0, 3.0], [0.0, 9.0]])
    y = np.array([NONE, NONE, 1])
    b = none_bias_for_target_leak(logits, y, NONE, target=0)
    shifted = logits.copy(); shifted[:, NONE] += b
    assert leakage_by_class(shifted, y, NONE)[1:].sum() == 0


# --- 불량 다이의 위치 (가설 P: 위치 지름길) ----------------------------------

def test_radius_is_zero_when_fails_sit_at_the_die_centre():
    """다이 영역 한가운데에 불량이 모이면 정규화 반지름이 0 이다."""
    x = np.ones((1, 9, 9), np.uint8)
    x[0, 4, 4] = 2
    assert fail_centroid_radius(x)[0] == pytest.approx(0.0, abs=1e-6)


def test_radius_is_larger_at_the_rim_than_at_the_centre():
    """가장자리 불량이 중심 불량보다 큰 값을 받는다. 이것이 이 자의 전부다."""
    centre = np.ones((1, 9, 9), np.uint8); centre[0, 4, 4] = 2
    rim = np.ones((1, 9, 9), np.uint8); rim[0, 0, 0] = 2
    assert fail_centroid_radius(rim)[0] > fail_centroid_radius(centre)[0]


def test_radius_is_nan_without_any_fail():
    """불량이 없으면 없는 값이다. 0 으로 채우면 '중심' 과 구분이 안 된다."""
    assert np.isnan(fail_centroid_radius(np.ones((1, 9, 9), np.uint8))[0])


def test_radius_ignores_cells_without_a_die():
    """다이가 없는 칸(0)은 기하에 끼지 않는다. 끼면 웨이퍼 바깥이 중심을 끈다."""
    full = np.ones((1, 9, 9), np.uint8); full[0, 4, 4] = 2
    holed = full.copy(); holed[0, :, :2] = 0      # 왼쪽 두 열을 다이 없음으로
    # 다이 영역의 중심이 오른쪽으로 옮겨가므로 같은 불량의 반지름은 커진다.
    assert fail_centroid_radius(holed)[0] > fail_centroid_radius(full)[0]


def test_radius_is_scale_free():
    """같은 배치를 크기만 키운 웨이퍼는 같은 값을 받아야 한다 — 크기 교란을 막는다."""
    small = np.ones((1, 11, 11), np.uint8); small[0, 0, 5] = 2
    big = np.ones((1, 21, 21), np.uint8); big[0, 0, 10] = 2
    assert fail_centroid_radius(small)[0] == pytest.approx(
        fail_centroid_radius(big)[0], rel=0.1)
