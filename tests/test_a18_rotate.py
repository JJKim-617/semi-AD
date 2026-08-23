"""자유 각도 회전 증강 (E23 후보, 18차).

**왜 이것을 시험하는가.** `docs/research/leakage_mechanism/README.md` 가 남긴 것:
translate 는 운영점이 아니라 판별력을 올리는데, 그럴듯한 남은 이야기는
"캔버스 안 위치가 순수한 잡음 변수이고 translate 가 그것을 지우는 참된 대칭" 이다.
같은 논리를 밀면 **각도도 잡음 변수**다 — Edge-Loc, Loc, Scratch 의 방향은 임의이고
Center, Donut, Edge-Ring, Near-full 은 회전 불변이다.
dihedral 은 그 대칭군의 **8개 원소만** 쓴다. 나머지를 쓰면 어떻게 되는가.

**단 이산 다이 격자에서 자유 회전은 참된 대칭이 아닐 수 있다** — 최근접 재표본이
격자를 성기게 만들어 `die_dropout` 처럼 굴 수 있다. 그것이 이 시험이 지키는 선이다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a18_augment import RECIPES, random_rotate  # noqa: E402


def wafer(n=4, size=32, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.hypot(yy - (size - 1) / 2, xx - (size - 1) / 2)
    base = np.where(d <= size // 2 - 2, 1, 0).astype(np.uint8)
    out = np.repeat(base[None], n, 0)
    fail = (rng.random(out.shape) < 0.1) & (out > 0)
    out[fail] = 2
    return out


def test_no_new_cell_values():
    """명목형 제약. 보간이 섞이면 여기서 깨진다."""
    out = random_rotate(wafer(), rng=np.random.default_rng(0), max_deg=180)
    assert set(np.unique(out)).issubset({0, 1, 2})


def test_shape_and_dtype_survive():
    x = wafer()
    out = random_rotate(x, rng=np.random.default_rng(0), max_deg=180)
    assert out.shape == x.shape and out.dtype == np.uint8


def test_zero_degrees_is_identity():
    """max_deg=0 이면 아무 일도 없어야 한다. 대조군을 만들 때 쓰는 성질이다."""
    x = wafer()
    assert np.array_equal(random_rotate(x, rng=np.random.default_rng(0), max_deg=0), x)


def test_ninety_degrees_matches_rot90_exactly():
    """90도는 격자 위의 정확한 대칭이다. 여기가 안 맞으면 회전 구현이 틀린 것이다.

    dihedral 이 이미 쓰는 변환과 정확히 일치해야 자유 회전을 그 확장이라 부를 수 있다.
    """
    x = wafer(n=1, size=33)          # 홀수 크기 = 격자에 정확한 중심이 있다
    got = random_rotate(x, rng=None, angles=np.array([90.0]))
    assert np.array_equal(got[0], np.rot90(x[0]))


def test_each_sample_gets_its_own_angle():
    """배치 전체에 같은 각을 걸면 다양성이 배치 수만큼으로 줄어든다."""
    x = np.repeat(wafer(n=1, size=33), 8, 0)
    out = random_rotate(x, rng=np.random.default_rng(0), max_deg=180)
    assert len({o.tobytes() for o in out}) > 1


def test_rotation_is_registered_as_a_recipe():
    """학습에서 이름으로 고를 수 있어야 한다."""
    assert "rotate" in RECIPES
    fn, kw = RECIPES["rotate"]
    assert fn is random_rotate


def test_die_count_is_roughly_preserved_at_ninety():
    """90도에서는 다이 수가 정확히 보존된다 — 재표본 손실이 0 이라는 뜻이다."""
    x = wafer(n=1, size=33)
    got = random_rotate(x, rng=None, angles=np.array([90.0]))
    assert int((got > 0).sum()) == int((x > 0).sum())


def test_arbitrary_angle_keeps_most_dies():
    """자유 각도에서도 다이의 대부분은 남아야 한다.

    최근접 재표본이 격자를 성기게 만들면 이 증강은 `die_dropout` 과 같아진다.
    **그러면 증강으로서 쓸모가 없다는 뜻이고, 이 시험이 그 선이다.**
    """
    x = wafer(n=1, size=64)
    got = random_rotate(x, rng=None, angles=np.array([37.0]))
    keep = int((got > 0).sum()) / int((x > 0).sum())
    assert keep > 0.95, f"다이의 {1 - keep:.1%} 가 회전에서 사라졌다"
