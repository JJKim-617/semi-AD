"""극좌표 표현. 웨이퍼는 원형이고 결함 클래스 절반이 반지름으로 정의된다.

E14 에서 앙상블 이득이 입력 표현의 다양성에 단조 증가하는 것이 확인됐다
(표현 1종 0.7420 < 2종 0.7641 < 3종 0.7732, 조합 56가지 전수 집계).
그런데 우리가 가진 표현은 pad64, resize64, resize96 셋뿐이고 셋 다 직교 격자다.
극좌표는 격자 자체가 다르므로 오류 패턴이 겹치지 않을 가능성이 높다.

행 = 반지름, 열 = 각도로 놓으면 두 가지가 따라온다.
  - Edge-Ring, Center, Donut 처럼 반지름으로 정의되는 결함이 행 방향 띠가 된다.
  - **웨이퍼의 회전이 열 방향 평행이동이 된다.** CNN 이 이미 가진 평행이동
    등변성이 회전 등변성으로 바뀐다. dihedral 증강이 +0.101 로 가장 크게 통했던
    것을 보면 이 데이터셋에서 회전 구조는 실제 신호다.

셀 값이 명목형이라 여기서도 보간은 금지다. nearest 만 쓴다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a1_preprocess_wm811k import polar_nn


def disc(n=64, radius=None):
    """정상 다이로 채운 원판. 바깥은 0(다이 없음)."""
    radius = radius or n // 2 - 1
    yy, xx = np.mgrid[0:n, 0:n]
    d = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2)
    return np.where(d <= radius, 1, 0).astype(np.uint8)


def wedge(n=64, start=0.0, span=np.pi / 2):
    """원판 위에 각도 구간 하나만 불량(2)으로 칠한다. 각도 패턴이 생겨야 회전을 잰다."""
    wm = disc(n)
    yy, xx = np.mgrid[0:n, 0:n]
    ang = np.arctan2(yy - (n - 1) / 2, xx - (n - 1) / 2) % (2 * np.pi)
    sel = ((ang - start) % (2 * np.pi)) < span
    wm[sel & (wm > 0)] = 2
    return wm


def ring(n=64, lo=18, hi=24):
    wm = disc(n)
    yy, xx = np.mgrid[0:n, 0:n]
    d = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2)
    wm[(d >= lo) & (d <= hi) & (wm > 0)] = 2
    return wm


class TestShapeAndValues:
    def test_returns_the_requested_square_size(self):
        assert polar_nn(disc(64), 32).shape == (32, 32)

    def test_keeps_the_categorical_dtype(self):
        assert polar_nn(disc(64), 32).dtype == np.uint8

    def test_invents_no_new_cell_values(self):
        """보간하면 존재하지 않는 값이 생긴다. 그것이 이 데이터셋에서 금지된 이유다."""
        out = polar_nn(wedge(64), 48)
        assert set(np.unique(out)).issubset({0, 1, 2})

    def test_handles_a_wafer_with_no_dies(self):
        assert polar_nn(np.zeros((16, 16), np.uint8), 8).sum() == 0

    def test_handles_a_non_square_wafer(self):
        wm = np.ones((20, 45), np.uint8)
        assert polar_nn(wm, 24).shape == (24, 24)


class TestGeometry:
    def test_the_first_row_is_the_wafer_centre(self):
        wm = disc(64)
        wm[31:33, 31:33] = 2          # 중심만 불량으로
        assert (polar_nn(wm, 64)[0] == 2).all()

    def test_a_concentric_ring_becomes_a_row_band(self):
        """반지름으로 정의된 결함이 행 방향으로 모여야 한다.

        띠의 가장자리 행은 이산화 때문에 부분적으로만 찬다(양끝 2행씩).
        요구할 것은 **내부 행이 모든 각도에서 불량**이라는 것이다.
        """
        out = polar_nn(ring(64), 64)
        rows = np.flatnonzero((out == 2).any(1))
        assert len(rows) > 4
        assert rows.max() - rows.min() + 1 == len(rows), "띠가 끊겨 있다"
        interior = rows[2:-2]
        assert (out[interior] == 2).all(), "내부 행에 불량 아닌 각도가 있다"

    def test_a_wedge_becomes_a_column_band(self):
        """각도로 정의된 결함은 열 방향으로 모여야 한다. 위 시험의 쌍대다."""
        out = polar_nn(wedge(64, span=np.pi / 2), 64)
        cols = np.flatnonzero((out == 2).any(0))
        assert 0 < len(cols) <= 64 // 4 + 3


class TestRotationBecomesTranslation:
    """이 표현을 쓰는 이유 자체다."""

    def test_rotating_the_wafer_shifts_the_angle_axis_by_a_quarter(self):
        n = 64
        a = polar_nn(wedge(n), n)
        b = polar_nn(np.rot90(wedge(n)), n)
        agree = [float((a == np.roll(b, s, axis=1)).mean()) for s in range(n)]
        best = int(np.argmax(agree))
        assert best in (n // 4, 3 * n // 4), f"최적 이동량이 {best} 다 (기대 16 또는 48)"
        assert agree[best] > 0.95

    def test_the_unshifted_comparison_does_not_match(self):
        """위 시험이 공허하지 않음을 보인다. 이동 없이는 안 맞아야 한다."""
        n = 64
        a = polar_nn(wedge(n), n)
        b = polar_nn(np.rot90(wedge(n)), n)
        assert float((a == b).mean()) < 0.9

    def test_a_ring_is_unchanged_by_rotation(self):
        """회전 대칭인 결함은 이동해도 그대로여야 한다."""
        n = 64
        a = polar_nn(ring(n), n)
        b = polar_nn(np.rot90(ring(n)), n)
        assert float((a == b).mean()) > 0.95
