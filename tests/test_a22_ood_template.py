"""O1 정상 통계 template 잔차 — 단위 테스트.

여기서 고정하는 성질은 세 가지다.

1. **one-class 격리** — template 은 정상만 본다. 결함이 섞이면 예외.
2. **균일 template 이면 구조 점수는 0** — S_llr 이 개수와 분리돼 있다는 것의 정의다.
   이 성질이 깨지면 "구조가 잡혔다"는 주장이 개수 신호의 재포장일 수 있다.
3. **크기 불변** — T-RADIAL 은 같은 패턴을 크게 그린 웨이퍼에 같은 점수를 줘야 한다.
   극좌표가 무너진 지점이라 명시적으로 박는다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a22_ood_template import (
    PositionTemplate,
    RadialTemplate,
    assert_one_class,
    fit_position_template,
    fit_radial_template,
    score_llr,
    score_nll,
)


def _disc(n: int, radius: float, size: int = 64) -> np.ndarray:
    """중앙에 반지름 radius 인 원판 die 영역. 값은 전부 1(정상)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    m = ((yy - c) ** 2 + (xx - c) ** 2) <= radius ** 2
    x = np.zeros((size, size), np.uint8)
    x[m] = 1
    return np.repeat(x[None], n, axis=0)


class TestOneClassGuard:
    def test_rejects_any_defect_label(self):
        y = np.array([0, 0, 3, 0])
        with pytest.raises(ValueError, match="one-class"):
            assert_one_class(y)

    def test_accepts_all_normal(self):
        assert_one_class(np.zeros(10, np.int64)) is None

    def test_fit_position_template_refuses_defect_labels(self):
        x = _disc(4, 20.0)
        with pytest.raises(ValueError, match="one-class"):
            fit_position_template(x, y=np.array([0, 0, 1, 0]))


class TestPositionTemplate:
    def test_probability_is_fail_over_die_with_smoothing(self):
        x = _disc(10, 20.0)
        x[:5, 32, 32] = 2                      # 10장 중 5장에서 중앙이 불량
        t = fit_position_template(x, smoothing=0.0)
        assert isinstance(t, PositionTemplate)
        assert t.p[32, 32] == pytest.approx(0.5)

    def test_cells_never_covered_by_a_die_fall_back_to_global_rate(self):
        """die 가 한 번도 없던 셀에 0 이나 NaN 을 넣으면 큰 웨이퍼에서 터진다."""
        x = _disc(10, 10.0)
        x[:, 32, 32] = 2
        t = fit_position_template(x, smoothing=1.0)
        assert t.die_count[0, 0] == 0
        assert 0.0 < t.p[0, 0] < 1.0
        assert t.p[0, 0] == pytest.approx(t.global_rate, abs=1e-9)

    def test_probability_is_clipped_away_from_zero_and_one(self):
        x = _disc(50, 20.0)                     # 불량이 하나도 없다
        t = fit_position_template(x, smoothing=1.0)
        assert (t.p > 0).all() and (t.p < 1).all()


class TestRadialTemplate:
    def test_bin_probabilities_track_the_radius_of_failures(self):
        """바깥 테두리만 불량이면 바깥 빈의 확률이 안쪽보다 커야 한다."""
        x = _disc(20, 25.0)
        yy, xx = np.mgrid[0:64, 0:64].astype(np.float64)
        r = np.sqrt((yy - 31.5) ** 2 + (xx - 31.5) ** 2)
        rim = (r > 23.0) & (r <= 25.0)
        x[:, rim] = 2
        t = fit_radial_template(x, n_bins=8, smoothing=1.0)
        assert isinstance(t, RadialTemplate)
        assert t.p[-1] > t.p[0]

    def test_is_invariant_to_wafer_size(self):
        """같은 패턴을 두 배로 그린 웨이퍼가 같은 반경 template 을 준다.

        극좌표가 무너진 지점이다. 여기서 크기 자유도를 남기면 같은 실패가 반복된다.
        """
        def rim_wafer(radius, thickness):
            x = _disc(1, radius)
            yy, xx = np.mgrid[0:64, 0:64].astype(np.float64)
            r = np.sqrt((yy - 31.5) ** 2 + (xx - 31.5) ** 2)
            x[0, (r > radius - thickness) & (r <= radius)] = 2
            return x

        small = fit_radial_template(rim_wafer(12.0, 3.0), n_bins=4, smoothing=1.0)
        big = fit_radial_template(rim_wafer(24.0, 6.0), n_bins=4, smoothing=1.0)
        assert np.allclose(small.p, big.p, atol=0.06)


class TestScorers:
    def test_uniform_template_makes_the_structure_score_identically_zero(self):
        """S_llr 이 개수와 분리돼 있다는 것의 정의.

        template 이 균일하면 재척도한 q1 이 곧 q0 이므로 로그가능도비가 0 이다.
        이 성질이 깨지면 구조 점수가 사실은 개수 신호다.
        """
        rng = np.random.default_rng(0)
        x = _disc(8, 22.0)
        for i in range(8):
            m = np.flatnonzero(x[i].ravel() > 0)
            pick = rng.choice(m, size=40 + 10 * i, replace=False)
            x[i].ravel()[pick] = 2
        flat = PositionTemplate(
            p=np.full((64, 64), 0.037), die_count=np.ones((64, 64)), global_rate=0.037)
        s = score_llr(x, flat)
        assert np.allclose(s, 0.0, atol=1e-9)

    def test_uniform_template_nll_is_linear_in_fail_and_die_counts(self):
        """대조군의 정체를 박는다 — 균일 template 의 S_nll 은 개수 통계일 뿐이다."""
        x = _disc(6, 20.0)
        for i in range(6):
            x[i][30, 20 + i] = 2
            if i % 2:
                x[i][31, 25] = 2
        p0 = 0.05
        flat = PositionTemplate(
            p=np.full((64, 64), p0), die_count=np.ones((64, 64)), global_rate=p0)
        s = score_nll(x, flat)
        n_fail = (x == 2).sum((1, 2)).astype(np.float64)
        n_die = (x > 0).sum((1, 2)).astype(np.float64)
        expect = n_fail * np.log((1 - p0) / p0) - n_die * np.log(1 - p0)
        assert np.allclose(s, expect, atol=1e-8)

    def test_structure_score_is_higher_where_the_template_says_failures_are_rare(self):
        """불량 개수가 같아도, template 이 낮은 자리에 놓인 쪽이 더 이상하다."""
        x = _disc(2, 25.0)
        yy, xx = np.mgrid[0:64, 0:64].astype(np.float64)
        r = np.sqrt((yy - 31.5) ** 2 + (xx - 31.5) ** 2)
        inner = np.flatnonzero(((r < 8.0) & (x[0] > 0)).ravel())[:30]
        outer = np.flatnonzero(((r > 22.0) & (r <= 25.0) & (x[0] > 0)).ravel())[:30]
        x[0].ravel()[inner] = 2
        x[1].ravel()[outer] = 2
        p = np.full((64, 64), 0.01)
        p[r > 20.0] = 0.30                      # 정상은 가장자리에서 불량이 잦다
        t = PositionTemplate(p=p, die_count=np.full((64, 64), 100.0), global_rate=0.05)
        s = score_llr(x, t)
        assert s[0] > s[1]

    def test_scores_are_finite_for_all_fail_and_no_fail_wafers(self):
        x = _disc(2, 20.0)
        x[1][x[1] > 0] = 2                      # 전부 불량
        p = np.full((64, 64), 0.05)
        t = PositionTemplate(p=p, die_count=np.full((64, 64), 10.0), global_rate=0.05)
        assert np.isfinite(score_nll(x, t)).all()
        assert np.isfinite(score_llr(x, t)).all()

    def test_radial_template_scoring_matches_position_scoring_shape(self):
        x = _disc(3, 20.0)
        x[:, 32, 32] = 2
        t = fit_radial_template(x, n_bins=8, smoothing=1.0)
        assert score_nll(x, t).shape == (3,)
        assert score_llr(x, t).shape == (3,)
