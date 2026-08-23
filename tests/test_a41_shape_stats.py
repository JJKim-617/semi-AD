"""결함 형태 통계 — 단위 테스트.

**픽셀 단위 정답을 합성하지 않는다**(기획서 §9.3). 여기서 박는 것은
"연산자가 의도한 기하량을 실제로 재는가" 뿐이고, 검사에는 사람이 손으로 확인할 수 있는
아주 작은 도형만 쓴다.

이 통계들은 **기술 통계이며 어떤 채점기에도 안 들어간다.**
`docs/experiments/candidate/ood_scratch_shift.md` §5 에 그 범위를 실행 전에 박았다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a41_shape_stats import (  # noqa: E402
    component_shape,
    mannwhitney_u,
    mean_fail_neighbors,
    shape_stats,
)


def wafer(rows):
    """문자 그림 -> 64x64 가 아니라 작은 배열. '.'=다이없음 'o'=정상 'x'=불량"""
    m = {".": 0, "o": 1, "x": 2}
    return np.array([[m[c] for c in r] for r in rows], dtype=np.uint8)


# --- 굵기 대리: 불량 다이의 평균 불량 8-이웃 수 ---------------------------------

def test_isolated_fail_has_zero_neighbors():
    x = wafer(["ooooo", "ooooo", "ooxoo", "ooooo", "ooooo"])[None]
    assert mean_fail_neighbors(x)[0] == pytest.approx(0.0)


def test_thin_line_of_three_has_mean_neighbors_four_thirds():
    """가로 3칸 선: 가운데는 이웃 2, 양끝은 각 1 -> 평균 4/3."""
    x = wafer(["ooooo", "ooooo", "oxxxo", "ooooo", "ooooo"])[None]
    assert mean_fail_neighbors(x)[0] == pytest.approx(4.0 / 3.0)


def test_thick_block_has_more_neighbors_than_thin_line():
    """**굵기 대리가 실제로 굵기를 재는지.** 3x3 덩어리 > 1x3 선."""
    line = wafer(["ooooo", "ooooo", "oxxxo", "ooooo", "ooooo"])[None]
    block = wafer(["ooooo", "oxxxo", "oxxxo", "oxxxo", "ooooo"])[None]
    assert mean_fail_neighbors(block)[0] > mean_fail_neighbors(line)[0]


def test_no_fail_gives_nan_not_zero():
    """불량이 없으면 '굵기 0' 이 아니라 **정의되지 않는다.** 0 으로 두면 평균이 오염된다."""
    x = wafer(["ooooo"] * 5)[None]
    assert np.isnan(mean_fail_neighbors(x)[0])


# --- 최대 연결성분의 기하 -------------------------------------------------------

def test_component_size_counts_largest_only():
    x = wafer(["xoooo", "ooooo", "ooxxx", "ooooo", "xoooo"])[None]
    assert component_shape(x)["size"][0] == 3.0


def test_diagonal_chain_is_one_component_under_8_connectivity():
    x = wafer(["xoooo", "oxooo", "ooxoo", "oooxo", "oooox"])[None]
    assert component_shape(x)["size"][0] == 5.0


def test_straight_line_is_more_elongated_than_square_block():
    """**이심률이 '가늘고 긴가' 를 재는지.**"""
    line = wafer(["ooooo", "ooooo", "xxxxx", "ooooo", "ooooo"])[None]
    block = wafer(["ooooo", "oxxxo", "oxxxo", "oxxxo", "ooooo"])[None]
    assert component_shape(line)["elongation"][0] > component_shape(block)["elongation"][0]


def test_square_block_elongation_is_one():
    block = wafer(["ooooo", "oxxxo", "oxxxo", "oxxxo", "ooooo"])[None]
    assert component_shape(block)["elongation"][0] == pytest.approx(1.0, abs=1e-9)


def test_elongation_is_rotation_invariant_under_90_degrees():
    """90도 회전은 격자 정렬을 유지하므로 값이 **정확히** 같아야 한다."""
    line = wafer(["ooooo", "ooooo", "xxxxx", "ooooo", "ooooo"])[None]
    rot = np.rot90(line, 1, axes=(1, 2)).copy()
    assert component_shape(line)["elongation"][0] == pytest.approx(
        component_shape(rot)["elongation"][0])


def test_major_axis_length_tracks_line_length():
    short = wafer(["ooooooo"] * 3 + ["ooxxxoo"] + ["ooooooo"] * 3)[None]
    long_ = wafer(["ooooooo"] * 3 + ["xxxxxxx"] + ["ooooooo"] * 3)[None]
    assert component_shape(long_)["major"][0] > component_shape(short)["major"][0]


def test_empty_component_is_nan_for_shape_but_zero_for_size():
    """크기 0 은 사실이지만 **모양은 정의되지 않는다.**"""
    x = wafer(["ooooo"] * 5)[None]
    r = component_shape(x)
    assert r["size"][0] == 0.0
    assert np.isnan(r["elongation"][0]) and np.isnan(r["major"][0])


def test_shape_of_single_die_component_is_defined_as_one():
    x = wafer(["ooooo", "ooooo", "ooxoo", "ooooo", "ooooo"])[None]
    r = component_shape(x)
    assert r["size"][0] == 1.0
    assert r["elongation"][0] == pytest.approx(1.0)


# --- 통계 묶음 -----------------------------------------------------------------

def test_shape_stats_returns_all_six_keys_with_matching_length():
    x = np.stack([wafer(["ooooo", "ooooo", "oxxxo", "ooooo", "ooooo"]),
                  wafer(["ooooo", "oxxxo", "oxxxo", "oxxxo", "ooooo"])])
    s = shape_stats(x, die_size=np.array([25.0, 25.0]))
    for k in ("die_size", "n_fail", "fail_ratio", "cc_size", "neighbors", "elongation", "major"):
        assert k in s and len(s[k]) == 2


def test_fail_ratio_excludes_no_die_cells():
    """다이가 없는 칸은 분모에서 빠져야 한다(a21 과 같은 관례)."""
    x = wafer(["..o..", ".ooo.", "oxxxo", ".ooo.", "..o.."])[None]
    s = shape_stats(x, die_size=np.array([13.0]))
    assert s["fail_ratio"][0] == pytest.approx(3.0 / 13.0)


# --- Mann-Whitney U ------------------------------------------------------------

def test_mannwhitney_matches_scipy_on_random_data():
    """**직접 구현한 검정을 참조 구현에 대고 박는다.**"""
    from scipy import stats
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 137)
    b = rng.normal(0.4, 1, 211)
    u, p = mannwhitney_u(a, b)
    ref = stats.mannwhitneyu(a, b, alternative="two-sided")
    assert u == pytest.approx(ref.statistic)
    assert p == pytest.approx(ref.pvalue, rel=1e-6)


def test_mannwhitney_handles_ties():
    """동점이 많은 정수 자료. 동점 보정을 빠뜨리면 p 가 틀린다."""
    from scipy import stats
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, 90).astype(float)
    b = rng.integers(0, 5, 120).astype(float)
    u, p = mannwhitney_u(a, b)
    ref = stats.mannwhitneyu(a, b, alternative="two-sided")
    assert u == pytest.approx(ref.statistic)
    assert p == pytest.approx(ref.pvalue, rel=1e-6)


def test_mannwhitney_ignores_nan():
    """굵기 대리는 불량이 없는 웨이퍼에서 nan 이다. 검정이 그걸 빼야 한다."""
    a = np.array([1.0, 2.0, np.nan, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    u, p = mannwhitney_u(a, b)
    u2, p2 = mannwhitney_u(a[~np.isnan(a)], b)
    assert u == pytest.approx(u2) and p == pytest.approx(p2)
