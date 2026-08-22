"""웨이퍼맵에 정당한 증강들. 명목형 셀 값을 만들어내지 않는 것이 공통 제약이다.

지금 쓰는 증강은 dihedral 하나뿐인데 그것이 이 프로젝트 최대 이득(+0.101)이었다.
그리고 결정규칙 쪽은 오라클 상한이 +0.006 으로 고갈됐으므로, 남은 길은 일반화를 건드리는 것이다.

셀 값은 0=다이 없음, 1=정상, 2=불량 이고 명목형이다.
**어떤 증강도 이 세 값 밖을 만들면 안 되고, 보간은 금지다.**
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a18_augment import (die_dropout, die_noise, random_scale, random_translate)


def wafer(n=4, size=32, seed=0):
    """가운데 원판 위에 불량이 섞인 합성 웨이퍼."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.hypot(yy - (size - 1) / 2, xx - (size - 1) / 2)
    base = np.where(d <= size // 2 - 2, 1, 0).astype(np.uint8)
    out = np.repeat(base[None], n, 0)
    fail = (rng.random(out.shape) < 0.1) & (out > 0)
    out[fail] = 2
    return out


class TestValuesStayNominal:
    @pytest.mark.parametrize("fn,kw", [
        (random_scale, dict(lo=0.5, hi=1.5)),
        (random_translate, dict(max_shift=4)),
        (die_noise, dict(rate=0.05)),
        (die_dropout, dict(rate=0.05)),
    ])
    def test_no_new_cell_values(self, fn, kw):
        x = wafer()
        out = fn(x, rng=np.random.default_rng(0), **kw)
        assert set(np.unique(out)).issubset({0, 1, 2})

    @pytest.mark.parametrize("fn,kw", [
        (random_scale, dict(lo=0.5, hi=1.5)),
        (random_translate, dict(max_shift=4)),
        (die_noise, dict(rate=0.05)),
        (die_dropout, dict(rate=0.05)),
    ])
    def test_shape_and_dtype_survive(self, fn, kw):
        x = wafer()
        out = fn(x, rng=np.random.default_rng(0), **kw)
        assert out.shape == x.shape and out.dtype == np.uint8


class TestRandomScale:
    def test_shrinking_reduces_the_die_area(self):
        x = wafer(n=1, size=64)
        small = random_scale(x, rng=np.random.default_rng(0), lo=0.4, hi=0.4)
        assert (small > 0).sum() < (x > 0).sum()

    def test_growing_increases_the_die_area(self):
        x = wafer(n=1, size=64)
        big = random_scale(x, rng=np.random.default_rng(0), lo=1.6, hi=1.6)
        assert (big > 0).sum() > (x > 0).sum()

    def test_scale_one_is_a_no_op(self):
        x = wafer(n=2, size=32)
        assert np.array_equal(random_scale(x, rng=np.random.default_rng(0), lo=1.0, hi=1.0), x)

    def test_keeps_the_wafer_centred(self):
        """캔버스 안 위치는 이동 증강이 다룬다. 스케일은 중심을 유지해야 축이 분리된다."""
        x = wafer(n=1, size=64)
        out = random_scale(x, rng=np.random.default_rng(0), lo=0.5, hi=0.5)
        ys, xs = np.nonzero(out[0] > 0)
        assert abs(ys.mean() - 31.5) < 2.0 and abs(xs.mean() - 31.5) < 2.0

    def test_defect_ratio_is_roughly_preserved(self):
        """크기만 바꾸는 것이지 패턴을 바꾸는 것이 아니다."""
        x = wafer(n=4, size=64, seed=1)
        out = random_scale(x, rng=np.random.default_rng(0), lo=0.6, hi=0.6)
        before = (x == 2).sum() / (x > 0).sum()
        after = (out == 2).sum() / max((out > 0).sum(), 1)
        assert abs(before - after) < 0.05


class TestRandomTranslate:
    def test_moves_most_wafers_in_a_batch(self):
        """이동량 0 은 정당한 추첨 결과다. 금지하면 증강이 한쪽으로 치우친다.

        따라서 '항상 바뀐다' 가 아니라 '대부분 바뀐다' 를 요구한다.
        max_shift=6 이면 (0,0) 이 나올 확률이 1/169 이므로 32개 중 대부분은 움직여야 한다.
        """
        x = wafer(n=32, size=32)
        out = random_translate(x, rng=np.random.default_rng(1), max_shift=6)
        moved = sum(not np.array_equal(out[i], x[i]) for i in range(len(x)))
        assert moved >= 30, f"32개 중 {moved}개만 움직였다"

    def test_preserves_the_die_count_when_it_fits(self):
        """캔버스에 여유가 있으면 다이가 잘려나가면 안 된다."""
        x = np.zeros((1, 32, 32), np.uint8)
        x[0, 14:18, 14:18] = 1
        out = random_translate(x, rng=np.random.default_rng(2), max_shift=4)
        assert (out > 0).sum() == (x > 0).sum()

    def test_zero_shift_is_a_no_op(self):
        x = wafer(n=2, size=32)
        assert np.array_equal(random_translate(x, rng=np.random.default_rng(0), max_shift=0), x)


class TestDieNoise:
    def test_flips_only_between_pass_and_fail(self):
        """다이가 없는 칸(0)은 건드리면 안 된다. 물리적으로 측정 자체가 없다."""
        x = wafer(n=4, size=32)
        out = die_noise(x, rng=np.random.default_rng(0), rate=0.3)
        assert np.array_equal(out == 0, x == 0)

    def test_rate_zero_is_a_no_op(self):
        x = wafer(n=2, size=32)
        assert np.array_equal(die_noise(x, rng=np.random.default_rng(0), rate=0.0), x)

    def test_higher_rate_flips_more(self):
        x = wafer(n=8, size=32, seed=3)
        lo = (die_noise(x, rng=np.random.default_rng(0), rate=0.02) != x).sum()
        hi = (die_noise(x, rng=np.random.default_rng(0), rate=0.20) != x).sum()
        assert hi > lo

    def test_flip_fraction_matches_the_rate(self):
        x = wafer(n=16, size=64, seed=4)
        rate = 0.1
        out = die_noise(x, rng=np.random.default_rng(0), rate=rate)
        die = x > 0
        got = (out != x).sum() / die.sum()
        assert abs(got - rate) < 0.02


class TestDieDropout:
    def test_only_removes_dies_never_adds(self):
        x = wafer(n=4, size=32)
        out = die_dropout(x, rng=np.random.default_rng(0), rate=0.2)
        assert ((x == 0) & (out != 0)).sum() == 0
        assert (out > 0).sum() < (x > 0).sum()

    def test_rate_zero_is_a_no_op(self):
        x = wafer(n=2, size=32)
        assert np.array_equal(die_dropout(x, rng=np.random.default_rng(0), rate=0.0), x)

    def test_dropped_fraction_matches_the_rate(self):
        x = wafer(n=16, size=64, seed=5)
        out = die_dropout(x, rng=np.random.default_rng(0), rate=0.15)
        die = x > 0
        dropped = (die & (out == 0)).sum() / die.sum()
        assert abs(dropped - 0.15) < 0.02


class TestIndependence:
    def test_each_sample_in_a_batch_gets_its_own_transform(self):
        """배치 전체에 같은 변환을 걸면 다양성이 배치 수만큼으로 줄어든다."""
        x = np.repeat(wafer(n=1, size=32), 8, axis=0)
        out = random_scale(x, rng=np.random.default_rng(0), lo=0.4, hi=1.6)
        areas = {(out[i] > 0).sum() for i in range(len(out))}
        assert len(areas) > 1, "배치 안 표본들이 같은 스케일을 받았다"
