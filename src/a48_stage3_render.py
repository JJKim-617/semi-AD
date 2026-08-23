"""3단계(MLLM 8종 분류) 렌더링과 프롬프트 — **규약은 코드보다 먼저 문서에 있다.**

`docs/experiments/candidate/ood_stage3_map_handoff.md` §4(arm 넷), §5(렌더링 규약),
§7(응답 파싱)을 그대로 구현한다. **결과를 보고 여기 상수를 바꾸면 그건 튜닝이다.**

## 자유도를 왜 여기서 다 막는가

정정 11 의 교훈이 "코드보다 먼저 정하라" 였다. 렌더링에는 자유도가 많다 —
색상표, 척도, 보간, 이미지 순서. 그 자유도를 결과 보고 고치면
**우리가 재는 것은 맵의 값어치가 아니라 렌더링 탐색의 값어치**가 된다.

## 척도는 train-none 에서 온다

요소 맵의 표시 범위를 test 통계로 잡으면 **범위 A 가 깨진다**(test 를 본 것이 된다).
`fit_component_scale` 은 train-none 맵의 **다이 칸만** 모아 1/99 백분위를 잡는다.

## 확대는 최근접 이웃이다

웨이퍼맵은 **명목형 격자**다. 보간하면 없는 중간 범주가 생긴다
(기획서 §7 이 임의 각도 회전을 금지하는 것과 같은 이유).
"""

from __future__ import annotations

import numpy as np

CLASS_NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random",
               "Scratch", "Near-full"]

# §5-5 이미지 순서는 고정한다. 반복마다 바꾸지 않는다.
ARM_IMAGE_ORDER = {
    "C0": ["wafer"],
    "E1": ["wafer", "A", "B", "C"],
    "P1": ["wafer", "A", "B", "C"],
    "S1": ["wafer"],
}
ARM_SCALARS = {"C0": False, "E1": True, "P1": True, "S1": True}

# §5-4 원본은 3색 명목 색상표. 다이없음 / 정상 / 불량.
WAFER_COLOURS = np.array([[0, 0, 0], [70, 90, 120], [235, 90, 60]], np.uint8)
UPSCALE = 8
EPS_SCALE = 1e-9


def fit_component_scale(maps, die_mask, lo_pct: float = 1.0,
                        hi_pct: float = 99.0) -> tuple[float, float]:
    """**train-none 맵의 다이 칸만** 모아 표시 범위를 잡는다.

    웨이퍼 밖 칸의 맵 값은 뜻이 없으므로 척도에 넣지 않는다.
    상수 맵이면 폭이 0 이 되어 0 으로 나누게 되므로 최소 폭을 준다.
    """
    m = np.asarray(maps, np.float64)
    d = np.asarray(die_mask) > 0
    v = m[d]
    if v.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(v, lo_pct))
    hi = float(np.percentile(v, hi_pct))
    if hi <= lo:
        hi = lo + EPS_SCALE
    return lo, hi


def _nearest_upscale(img: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return img
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


def render_component(m, die_mask, lo: float, hi: float, upscale: int = UPSCALE):
    """요소 맵 한 장 → RGB uint8. **웨이퍼 밖은 순수 검정**, 다이는 단일 색상표.

    값은 [lo, hi] 로 **자른다** — test 에 더 큰 값이 나와도 척도를 늘리지 않는다.
    """
    m = np.asarray(m, np.float64)
    d = np.asarray(die_mask) > 0
    t = np.clip((m - lo) / max(hi - lo, EPS_SCALE), 0.0, 1.0)
    # 어두운 보라 -> 밝은 노랑. 단조 밝기라 값의 크기가 눈으로 순서대로 읽힌다.
    r = np.clip(60 + 195 * t, 0, 255)
    g = np.clip(20 + 215 * t ** 1.3, 0, 255)
    b = np.clip(110 - 70 * t, 0, 255)
    img = np.stack([r, g, b], -1).astype(np.uint8)
    img[~d] = 0
    return _nearest_upscale(img, upscale)


def render_wafer(x, upscale: int = UPSCALE):
    """원본 웨이퍼맵 → RGB uint8. **3색뿐이다.** 연속 색상표를 쓰지 않는다."""
    a = np.asarray(x, np.int64)
    return _nearest_upscale(WAFER_COLOURS[np.clip(a, 0, 2)], upscale)


def placebo_pairing(n: int, seed: int = 0) -> np.ndarray:
    """위약 arm 의 짝. **자기 자신과 짝지어지면 실험군과 같아지므로** 고정점을 없앤다.

    순환 자리바꿈으로 만든 뒤 섞는다. 고정점이 남을 수 없다.
    """
    if n < 2:
        raise ValueError("위약 짝을 만들려면 표본이 둘 이상이어야 한다")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return perm[(np.argsort(perm) + 1) % n]


def build_prompt(arm: str, scalars: dict) -> str:
    """arm 별 프롬프트. **8종 이름을 전부 열거한다**(범위 B — §9 에 선언했다).

    `scalars` 는 **그 arm 이 보여 줄** 값이다. 위약이면 호출부가 **다른 웨이퍼의**
    스칼라를 넘긴다 — 여기서 자기 것을 다시 읽지 않는다.
    """
    if arm not in ARM_IMAGE_ORDER:
        raise ValueError(f"모르는 arm: {arm}. 가능한 것: {tuple(ARM_IMAGE_ORDER)}")
    lines = [
        "You are given a semiconductor wafer map. Each cell is one die:",
        "black = no die, blue-grey = passing die, orange = failing die.",
        "",
        "The wafer has a defect pattern. Classify it into exactly one of:",
        "  " + ", ".join(CLASS_NAMES),
        "",
    ]
    if len(ARM_IMAGE_ORDER[arm]) > 1:
        lines += [
            "Three additional maps are provided, in this order after the wafer map:",
            "  A - local failing-die density in a 5x5 window, calibrated by radius band",
            "  B - directional line-filter response, window length 11",
            "  C - local failing-die density in a 3x3 window",
            "Brighter means a more unusual value. They are computed from the wafer map",
            "itself using only normal wafers as reference.",
            "",
        ]
    if ARM_SCALARS[arm]:
        lines += [
            "Numeric summaries:",
            "  failing die count: %s" % scalars["n_fail"],
            "  log(B) - log(C): %.4f" % scalars["log_b_minus_log_c"],
            "",
        ]
    lines += [
        "Answer with exactly one class name from the list and nothing else.",
    ]
    return "\n".join(lines)


def parse_response(text) -> str | None:
    """응답에서 클래스 하나를 뽑는다. **못 뽑으면 None 이고 그건 오답으로 센다**(§7).

    빼고 세면 arm 마다 다른 표본을 재게 되므로, 거부도 형식 위반도 오답이다.
    **두 개 이상이 언급되면 형식 위반**이다 — 하나만 답하라고 했다.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    low = text.lower()
    hits = []
    for c in CLASS_NAMES:
        n = c.lower()
        # Edge-Loc 이 Loc 을 포함하므로 긴 이름을 먼저 지운 뒤 센다
        if n in low:
            hits.append(c)
    for long, short in (("Edge-Loc", "Loc"), ("Edge-Ring", "Ring")):
        if long in hits and short in hits and low.count(short.lower()) == low.count(long.lower()):
            hits.remove(short)
    hits = sorted(set(hits))
    return hits[0] if len(hits) == 1 else None
