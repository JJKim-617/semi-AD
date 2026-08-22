"""반경 대역별 밀도 보정 — 단위 테스트.

발상은 "**같은 자리 치고** 얼마나 드문가" 다. 그래서 박아야 할 성질은
- 같은 원시 밀도라도 **대역이 다르면 백분위가 다르다** (아니면 전역 보정과 같다)
- 보정에 **정상만** 들어간다
- 대역 안에서는 **단조**다 (순서를 뒤집으면 보정이 아니라 왜곡이다)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a29_radial_calibration import (  # noqa: E402
    RadialBandReference,
    band_index,
    calibrate_map,
    fit_band_reference,
)


def _disc(n=1, radius=25.0, size=64):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    m = ((yy - c) ** 2 + (xx - c) ** 2) <= radius ** 2
    x = np.zeros((size, size), np.uint8)
    x[m] = 1
    return np.repeat(x[None], n, axis=0)


class TestBandIndex:
    def test_centre_is_band_zero_and_rim_is_the_last_band(self):
        x = _disc(1, 25.0)
        b = band_index(x, n_bands=8)
        assert b[0, 31, 31] == 0
        assert b[0, 31, 31 + 25] == 7

    def test_non_die_cells_get_minus_one(self):
        x = _disc(1, 10.0)
        b = band_index(x, n_bands=8)
        assert np.all(b[x == 0] == -1)

    def test_it_is_invariant_to_wafer_size(self):
        """대역은 웨이퍼 자기 반경으로 정규화한다. 아니면 크기가 점수에 샌다."""
        small = band_index(_disc(1, 12.0), n_bands=8)
        big = band_index(_disc(1, 24.0), n_bands=8)
        assert small.max() == big.max() == 7
        assert (small >= 0).sum() < (big >= 0).sum()


class TestFitBandReference:
    def test_it_refuses_defect_labels(self):
        x = _disc(4, 20.0)
        v = np.zeros_like(x, np.float64)
        with pytest.raises(ValueError, match="one-class"):
            fit_band_reference(v, x, y=np.array([0, 0, 1, 0]), n_bands=8)

    def test_every_band_gets_a_sorted_reference(self):
        rng = np.random.default_rng(0)
        x = _disc(30, 24.0)
        v = rng.random(x.shape) * (x > 0)
        ref = fit_band_reference(v, x, y=np.zeros(30, np.int64), n_bands=8)
        assert isinstance(ref, RadialBandReference)
        assert len(ref.values) == 8
        for arr in ref.values:
            assert len(arr) > 0
            assert np.all(np.diff(arr) >= 0)


class TestCalibrate:
    def _ref_with_two_different_bands(self):
        """대역 0 은 값이 작고, 대역 1 은 값이 크게 나오는 참조."""
        return RadialBandReference(
            values=[np.linspace(0.0, 0.2, 101), np.linspace(0.5, 0.9, 101)],
            n_bands=2)

    def test_same_raw_density_maps_to_different_percentiles_by_band(self):
        """**이 실험의 핵심 성질.** 이게 없으면 전역 보정과 다를 게 없다."""
        ref = self._ref_with_two_different_bands()
        x = np.ones((1, 4, 4), np.uint8)
        bands = np.zeros((1, 4, 4), np.int64)
        bands[0, :, 2:] = 1
        v = np.full((1, 4, 4), 0.6)
        u = calibrate_map(v, bands, ref)
        assert u[0, 0, 0] == pytest.approx(1.0, abs=0.02)     # 대역 0 에선 최상위
        assert 0.1 < u[0, 0, 2] < 0.4                          # 대역 1 에선 하위권

    def test_it_is_monotone_within_a_band(self):
        ref = self._ref_with_two_different_bands()
        bands = np.zeros((1, 1, 5), np.int64)
        v = np.array([[[0.0, 0.05, 0.1, 0.15, 0.2]]])
        u = calibrate_map(v, bands, ref)
        assert np.all(np.diff(u[0, 0]) >= 0)

    def test_cells_outside_the_die_region_are_zero(self):
        ref = self._ref_with_two_different_bands()
        bands = np.full((1, 3, 3), -1, np.int64)
        bands[0, 1, 1] = 0
        u = calibrate_map(np.full((1, 3, 3), 0.15), bands, ref)
        assert u[0, 0, 0] == 0.0
        assert u[0, 1, 1] > 0.0

    def test_percentile_never_reaches_exactly_one_for_an_unseen_larger_value(self):
        """1.0 이 되면 뒤에서 -log(1-u) 같은 변환이 터진다."""
        ref = self._ref_with_two_different_bands()
        bands = np.zeros((1, 1, 1), np.int64)
        u = calibrate_map(np.array([[[99.0]]]), bands, ref)
        assert u[0, 0, 0] < 1.0


class TestEndToEndShape:
    def test_calibrated_map_has_the_same_shape_and_respects_the_die_mask(self):
        rng = np.random.default_rng(1)
        x = _disc(6, 22.0)
        v = rng.random(x.shape) * (x > 0)
        bands = band_index(x, n_bands=8)
        ref = fit_band_reference(v, x, y=np.zeros(6, np.int64), n_bands=8)
        u = calibrate_map(v, bands, ref)
        assert u.shape == x.shape
        assert np.all(u[x == 0] == 0.0)
        assert u.max() <= 1.0
