"""3단계 렌더링과 프롬프트 — 단위 테스트.

규약은 `docs/experiments/candidate/ood_stage3_map_handoff.md` §4, §5 에
**코드보다 먼저** 박았다. 여기서 박는 것은 그 §5 의 다섯 조항이다.

1. 척도는 **train-none 에서** 고정한다 (test 통계를 안 쓴다 — 범위 A 유지)
2. 웨이퍼 밖은 순수 검정
3. **최근접 이웃** 확대 (보간 금지 — 명목형 격자다)
4. 원본은 3색 명목 색상표
5. 이미지 순서 고정

그리고 §4 의 위약 arm 이 **다른 웨이퍼의** 맵과 스칼라를 쓴다는 것,
§7 이 **형식 위반과 거부를 오답으로 센다**는 것을 박는다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a48_stage3_render import (  # noqa: E402
    ARM_IMAGE_ORDER,
    CLASS_NAMES,
    build_prompt,
    fit_component_scale,
    parse_response,
    placebo_pairing,
    render_component,
    render_wafer,
)


def wafer(rows):
    m = {".": 0, "o": 1, "x": 2}
    return np.array([[m[c] for c in r] for r in rows], dtype=np.uint8)


# --- §5-1 척도는 train-none 에서 --------------------------------------------------

def test_scale_uses_only_die_cells():
    """웨이퍼 밖(0)의 맵 값은 척도에 들어가면 안 된다 — 거기 값은 뜻이 없다."""
    m = np.array([[[100.0, 0.0], [0.0, 0.0]]])
    die = np.array([[[1, 0], [0, 0]]], np.uint8)
    lo, hi = fit_component_scale(m, die)
    assert lo == pytest.approx(100.0) and hi == pytest.approx(100.0)


def test_scale_is_the_1_and_99_percentile():
    rng = np.random.default_rng(0)
    m = rng.random((10, 8, 8))
    die = np.ones((10, 8, 8), np.uint8)
    lo, hi = fit_component_scale(m, die)
    v = m.ravel()
    assert lo == pytest.approx(np.percentile(v, 1))
    assert hi == pytest.approx(np.percentile(v, 99))


def test_scale_never_collapses_to_a_zero_width_range():
    """상수 맵이면 lo == hi 라 0 으로 나누게 된다. 그걸 막아야 한다."""
    m = np.full((3, 4, 4), 2.0)
    lo, hi = fit_component_scale(m, np.ones((3, 4, 4), np.uint8))
    assert hi > lo


# --- §5-2, §5-3 렌더링 ------------------------------------------------------------

def test_outside_the_wafer_is_pure_black():
    m = np.array([[[5.0, 5.0], [5.0, 5.0]]])
    die = np.array([[[0, 1], [1, 1]]], np.uint8)
    img = render_component(m[0], die[0], 0.0, 10.0, upscale=1)
    assert img[0, 0].tolist() == [0, 0, 0]
    assert img[1, 1].tolist() != [0, 0, 0]


def test_upscale_is_nearest_neighbour_exactly():
    """보간하면 격자가 뭉개진다. 각 칸이 정확히 8x8 블록이어야 한다."""
    m = np.array([[0.0, 10.0], [10.0, 0.0]])
    die = np.ones((2, 2), np.uint8)
    img = render_component(m, die, 0.0, 10.0, upscale=8)
    assert img.shape == (16, 16, 3)
    for i in range(2):
        for j in range(2):
            block = img[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            assert np.array_equal(block, np.broadcast_to(block[0, 0], block.shape))


def test_values_are_clipped_to_the_fixed_scale():
    """test 에 train-none 보다 큰 값이 나와도 척도를 늘리지 않는다(§5-1)."""
    die = np.ones((1, 1), np.uint8)
    hi_img = render_component(np.array([[10.0]]), die, 0.0, 10.0, upscale=1)
    over_img = render_component(np.array([[9999.0]]), die, 0.0, 10.0, upscale=1)
    assert np.array_equal(hi_img, over_img)


def test_component_rendering_is_monotone_in_the_value():
    die = np.ones((1, 1), np.uint8)
    vals = [0.0, 2.5, 5.0, 7.5, 10.0]
    lum = [render_component(np.array([[v]]), die, 0.0, 10.0, 1).astype(int).sum()
           for v in vals]
    assert lum == sorted(lum)


def test_wafer_uses_three_nominal_colours_only():
    x = wafer(["..", "ox"])
    img = render_wafer(x, upscale=1)
    cols = {tuple(img[i, j]) for i in range(2) for j in range(2)}
    assert len(cols) == 3
    assert (0, 0, 0) in cols                       # 다이 없음은 검정


def test_wafer_rendering_has_no_intermediate_shades():
    """연속 색상표를 쓰면 '조금 불량' 같은 없는 범주가 생긴다."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, (16, 16)).astype(np.uint8)
    img = render_wafer(x, upscale=4)
    assert len({tuple(c) for c in img.reshape(-1, 3)}) <= 3


# --- §4 arm 구성 ------------------------------------------------------------------

def test_control_arm_has_one_image_and_experimental_has_four():
    assert len(ARM_IMAGE_ORDER["C0"]) == 1
    assert len(ARM_IMAGE_ORDER["E1"]) == 4
    assert len(ARM_IMAGE_ORDER["P1"]) == 4
    assert len(ARM_IMAGE_ORDER["S1"]) == 1


def test_image_order_is_fixed_original_then_a_then_b_then_c():
    assert ARM_IMAGE_ORDER["E1"] == ["wafer", "A", "B", "C"]
    assert ARM_IMAGE_ORDER["P1"] == ["wafer", "A", "B", "C"]


def test_placebo_pairs_each_wafer_with_a_different_one():
    """위약이 자기 자신과 짝지어지면 실험군과 같아진다."""
    p = placebo_pairing(200, seed=0)
    assert len(p) == 200 and len(set(p.tolist())) == 200
    assert not (p == np.arange(200)).any()


def test_placebo_pairing_is_deterministic():
    assert np.array_equal(placebo_pairing(50, seed=1), placebo_pairing(50, seed=1))
    assert not np.array_equal(placebo_pairing(50, seed=1), placebo_pairing(50, seed=2))


def test_scalar_arm_gets_the_wafers_own_scalars_and_placebo_does_not():
    own = {"n_fail": 100, "log_b_minus_log_c": 0.5}
    other = {"n_fail": 7, "log_b_minus_log_c": -2.0}
    assert "100" in build_prompt("S1", own)
    assert "100" not in build_prompt("P1", other)
    assert "7" in build_prompt("P1", other)


def test_control_prompt_carries_no_scalars():
    p = build_prompt("C0", {"n_fail": 100, "log_b_minus_log_c": 0.5})
    assert "100" not in p


def test_every_prompt_lists_all_eight_classes():
    for arm in ("C0", "E1", "P1", "S1"):
        p = build_prompt(arm, {"n_fail": 1, "log_b_minus_log_c": 0.0})
        for c in CLASS_NAMES:
            assert c in p


# --- §7 응답 파싱: 형식 위반은 오답 -------------------------------------------------

def test_parses_a_bare_class_name():
    assert parse_response("Scratch") == "Scratch"


def test_parsing_is_case_insensitive_and_ignores_surrounding_text():
    assert parse_response("I think the answer is  edge-ring .") == "Edge-Ring"


def test_refusal_returns_none_and_is_therefore_wrong():
    """§7: 형식 위반과 거부는 **오답으로 센다.** 빼고 세면 arm 마다 다른 집합을 재게 된다."""
    assert parse_response("I cannot determine this.") is None
    assert parse_response("") is None


def test_an_answer_naming_two_classes_is_a_format_violation():
    assert parse_response("Either Center or Donut") is None


def test_a_class_name_that_is_not_in_the_eight_is_none():
    assert parse_response("Ring") is None
