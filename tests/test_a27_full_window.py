"""온전 창 가드 — 잘린 창은 분모가 무너져 거짓 포화를 만든다.

실측(표본 8,000장): 정상 웨이퍼의 70.63% 에서 길이 7 선 창이 포화하는데,
그 포화 창의 **43.1% 가 다이 3개짜리**이고 온전한 7개짜리는 **0.2%** 다.
온전 창만 세면 정상 포화가 **0.15%** 로 떨어진다.

그러니 포화는 "가장자리에 긴 불량 줄이 있어서" 가 아니라
**창이 경계에서 잘려 분모가 2~4로 무너져서** 생긴다.
불량률 11.6% 에서 2~4개 연속은 우연히 늘 생긴다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a24_ood_residual import local_fail_density_map  # noqa: E402
from a27_line_filter import line_density_map, line_density_max  # noqa: E402


def _disc(n=1, radius=25.0, size=64):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    m = ((yy - c) ** 2 + (xx - c) ** 2) <= radius ** 2
    x = np.zeros((size, size), np.uint8)
    x[m] = 1
    return np.repeat(x[None], n, axis=0)


class TestGuardRemovesClippedWindows:
    @staticmethod
    def _stub_wafer():
        """본체에서 떨어져 나온 다이 3개짜리 돌기. 길이 7 창이 여기서 3칸으로 잘린다.

        실제 pad 캐시에서도 웨이퍼 경계의 창은 이렇게 잘린다 —
        실측 포화 창의 43.1% 가 다이 3개짜리였다.
        """
        x = np.zeros((1, 64, 64), np.uint8)
        x[0, 40:56, 20:44] = 1                  # 본체
        x[0, 30:33, 31] = 1                     # 떨어진 돌기 (세로 3칸)
        return x

    def test_a_short_run_at_the_boundary_saturates_without_the_guard(self):
        """잘린 창에서는 3칸만 죽어도 1.0 이 된다 — 가드가 없을 때."""
        x = self._stub_wafer()
        x[0, 30:33, 31] = 2
        assert line_density_max(x, length=7, n_orient=8)[0] == pytest.approx(1.0)

    def test_the_guard_stops_it(self):
        x = self._stub_wafer()
        x[0, 30:33, 31] = 2
        assert line_density_max(x, length=7, n_orient=8, min_dies=7)[0] < 0.999

    def test_an_interior_full_line_still_scores_one(self):
        """가드가 진짜 신호까지 죽이면 안 된다."""
        x = _disc(1, 25.0)
        x[0, 31, 28:35] = 2
        assert line_density_max(x, length=7, n_orient=8, min_dies=7)[0] == pytest.approx(1.0)

    def test_guarded_cells_are_zero_in_the_map(self):
        x = _disc(1, 8.0)                       # 작은 웨이퍼 — 온전 창이 거의 없다
        m = line_density_map(x, length=11, n_orient=8, min_dies=11)
        assert np.all(m == 0.0)

    def test_min_dies_of_one_is_the_old_behaviour(self):
        x = _disc(2, 20.0)
        x[:, 30, 22:28] = 2
        a = line_density_map(x, length=7, n_orient=8)
        b = line_density_map(x, length=7, n_orient=8, min_dies=1)
        assert np.array_equal(a, b)


class TestSquareWindowGuard:
    """분모 붕괴는 창 모양과 무관하다 — 정사각에도 같은 문제가 있다."""

    @staticmethod
    def _isolated_die():
        """본체에서 떨어진 다이 하나. 7x7 창이 여기서 다이 1개로 잘린다."""
        x = np.zeros((1, 64, 64), np.uint8)
        x[0, 40:56, 20:44] = 1
        x[0, 20, 31] = 1
        return x

    def test_square_window_also_saturates_at_the_boundary_without_the_guard(self):
        x = self._isolated_die()
        x[0, 20, 31] = 2
        assert local_fail_density_map(x, k=7).max() == pytest.approx(1.0)

    def test_the_guard_lowers_it(self):
        x = self._isolated_die()
        x[0, 20, 31] = 2
        loose = local_fail_density_map(x, k=7).max()
        tight = local_fail_density_map(x, k=7, min_dies=49).max()
        assert tight < loose
        assert tight == 0.0

    def test_full_square_window_requirement_keeps_interior_signal(self):
        x = _disc(1, 25.0)
        x[0, 29:34, 29:34] = 2
        m = local_fail_density_map(x, k=5, min_dies=25)
        assert m.max() == pytest.approx(1.0)


class TestGuardCanSilenceWholeWafers:
    """조건 7 — 온전 창이 하나도 없는 웨이퍼는 점수가 0 이 되어 정상 취급된다."""

    def test_a_wafer_smaller_than_the_window_scores_zero(self):
        x = np.zeros((1, 64, 64), np.uint8)
        x[0, 30:33, 30:33] = 1
        x[0, 31, 31] = 2
        assert line_density_max(x, length=11, n_orient=8, min_dies=11)[0] == 0.0
