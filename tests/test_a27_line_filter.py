"""방향성 선형 필터 — 단위 테스트.

정사각 창은 선을 조직적으로 희석한다(7x7 창에 길이 7 선이 들어오면 밀도 7/49 = 0.14).
창을 선 모양으로 바꾸면 같은 선이 1.0 이 된다. 여기서 박는 것은 그 성질과,
**방향에 대해 대칭인가**(가로 긁힘과 세로 긁힘을 같게 보는가) 두 가지다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a24_ood_residual import local_fail_density_max  # noqa: E402
from a27_line_filter import (  # noqa: E402
    line_density_map,
    line_density_max,
    line_kernels,
    radius_mask,
)


def _disc(n=1, radius=25.0, size=64):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    m = ((yy - c) ** 2 + (xx - c) ** 2) <= radius ** 2
    x = np.zeros((size, size), np.uint8)
    x[m] = 1
    return np.repeat(x[None], n, axis=0)


class TestKernels:
    def test_there_is_one_kernel_per_orientation(self):
        ks = line_kernels(length=7, n_orient=8)
        assert len(ks) == 8
        assert all(k.shape == (7, 7) for k in ks)

    def test_each_kernel_is_a_line_through_the_centre(self):
        for k in line_kernels(length=7, n_orient=8):
            assert k[3, 3] == 1
            assert 5 <= int(k.sum()) <= 7      # 이산화로 겹치는 칸이 생길 수 있다

    def test_no_orientation_sees_more_cells_than_the_axis_aligned_one(self):
        """방향마다 창 크기가 다르면 그 방향의 긁힘만 조직적으로 희석된다.

        처음 구현이 여기서 걸렸다 — 22.5° 커널이 9칸, 축 방향이 7칸이었다.
        """
        sums = [int(k.sum()) for k in line_kernels(length=7, n_orient=8)]
        assert max(sums) == sums[0] == 7

    def test_orientations_are_distinct(self):
        ks = line_kernels(length=9, n_orient=8)
        flat = {k.tobytes() for k in ks}
        assert len(flat) == 8

    def test_width_two_kernels_are_thicker(self):
        thin = line_kernels(length=7, n_orient=4, width=1)
        thick = line_kernels(length=7, n_orient=4, width=2)
        assert int(thick[0].sum()) > int(thin[0].sum())


class TestLineDensityBeatsSquareOnThinLines:
    def test_a_one_pixel_line_saturates_the_line_filter(self):
        """정사각 창이 희석하는 바로 그 구조."""
        x = _disc(1, 25.0)
        x[0, 31, 26:33] = 2                    # 길이 7 가로 선
        line = line_density_max(x, length=7, n_orient=8)[0]
        square = local_fail_density_max(x, k=7)[0]
        assert line == pytest.approx(1.0)
        assert square < 0.25
        assert line > 3.5 * square

    def test_a_compact_blob_does_not_gain_as_much(self):
        """선 필터가 모든 것을 다 올려 주면 변별력이 없다.

        덩어리는 정사각 창에서도 이미 높으므로 이득이 작아야 한다.
        """
        x = _disc(1, 25.0)
        x[0, 29:34, 29:34] = 2                 # 5x5 덩어리
        line = line_density_max(x, length=7, n_orient=8)[0]
        square = local_fail_density_max(x, k=7)[0]
        assert line / square < 3.5

    @pytest.mark.parametrize("angle_cells", [
        [(31, 26 + t) for t in range(7)],              # 가로
        [(26 + t, 31) for t in range(7)],              # 세로
        [(26 + t, 26 + t) for t in range(7)],          # 대각
        [(26 + t, 31 - (t + 1) // 2) for t in range(7)],   # 기울기 2 (8-연결)
        [(31 - (t + 1) // 2, 26 + t) for t in range(7)],   # 기울기 1/2 (8-연결)
    ])
    def test_it_is_symmetric_across_orientations(self, angle_cells):
        """가로 긁힘과 세로 긁힘을 다르게 보면 방향 편향이 결과에 섞인다.

        긁힘은 **8-연결된 선**이다. 격자를 2칸씩 건너뛰는 점열은 긁힘이 아니라
        흩어진 점이라 여기서 재는 대상이 아니다 — 처음에 그걸 넣었다가 틀렸다.
        """
        x = _disc(1, 26.0)
        for (i, j) in angle_cells:
            x[0, i, j] = 2
        assert line_density_max(x, length=7, n_orient=8)[0] > 0.65


class TestMapProperties:
    def test_map_is_zero_outside_the_die_region(self):
        x = _disc(2, 12.0)
        x[:, 31, 31] = 2
        m = line_density_map(x, length=7, n_orient=8)
        assert np.all(m[x == 0] == 0.0)

    def test_map_max_equals_the_scalar(self):
        x = _disc(3, 22.0)
        for i in range(3):
            x[i, 30, 24 + i:30 + i] = 2
        m = line_density_map(x, length=7, n_orient=8)
        assert np.allclose(m.max((1, 2)), line_density_max(x, length=7, n_orient=8))

    def test_density_stays_a_fraction(self):
        x = _disc(4, 20.0)
        x[:, 30, 25:35] = 2
        m = line_density_map(x, length=9, n_orient=8)
        assert m.min() >= -1e-12 and m.max() <= 1.0 + 1e-12

    def test_a_clean_wafer_scores_zero(self):
        x = _disc(2, 20.0)
        assert np.allclose(line_density_max(x, length=7, n_orient=8), 0.0)


class TestRadiusMask:
    """중심 다이 제외 arm 용. 정상 웨이퍼의 중심 최근접 다이가 58.9% 불량이다."""

    def test_it_excludes_the_innermost_fraction_of_the_radius(self):
        x = _disc(1, 25.0)
        m = radius_mask(x, r_min=1.0 / 32.0)
        assert m[0, 31, 31] == False            # noqa: E712  중심은 빠진다
        assert m[0, 31, 50] == True             # noqa: E712  바깥은 남는다

    def test_non_die_cells_are_always_excluded(self):
        x = _disc(1, 10.0)
        m = radius_mask(x, r_min=0.0)
        assert np.all(m[x == 0] == False)       # noqa: E712

    def test_zero_r_min_keeps_every_die(self):
        x = _disc(1, 20.0)
        assert np.array_equal(radius_mask(x, r_min=0.0), x > 0)
