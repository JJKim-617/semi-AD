"""웨이퍼맵 증강. 명목형 셀 값을 보존하는 것이 공통 제약이다.

셀 값은 0=다이 없음, 1=정상, 2=불량이고 명목형이다. 보간은 존재하지 않는 값을 만들므로
어떤 증강도 nearest 재표본이거나 값 치환이어야 한다.

지금까지 dihedral 하나만 썼고 그것이 이 프로젝트 최대 이득(+0.101)이었다.
그리고 결정규칙 쪽은 오라클 상한이 +0.006 으로 고갈됐다 —
남은 길은 표현과 일반화이고, 증강이 그 축에서 가장 값싼 도구다.

각 함수는 **표본마다 독립적으로** 변환을 뽑는다. 배치 전체에 같은 변환을 걸면
다양성이 배치 수만큼으로 줄어든다.
"""

from __future__ import annotations

import numpy as np


def _nn_resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """nearest 재표본. 색인 매핑만 쓰므로 값이 보존된다."""
    sh, sw = img.shape
    rows = (np.arange(h) * sh // h).clip(0, sh - 1)
    cols = (np.arange(w) * sw // w).clip(0, sw - 1)
    return img[rows[:, None], cols]


def random_scale(x: np.ndarray, rng, lo: float = 0.6, hi: float = 1.4) -> np.ndarray:
    """캔버스는 그대로 두고 웨이퍼만 확대/축소한다. pad 표현 전용.

    이 데이터셋의 크기 분포가 train 과 test 에서 크게 다르다 —
    train 의 58.1% 가 dieSize 400~562 한 구간에 몰려 있는데
    test 는 562~1090 구간이 45.9% 다(train 은 14.2%).
    무작위 스케일은 **학습에서 비어 있는 크기 구간을 합성한다.**

    중심은 유지한다. 캔버스 안 위치를 흔드는 것은 random_translate 의 일이고,
    두 축을 섞으면 어느 쪽이 통했는지 알 수 없다.
    """
    x = np.asarray(x)
    out = np.zeros_like(x)
    n, H, W = x.shape
    for i in range(n):
        s = float(rng.uniform(lo, hi))
        ys, xs = np.nonzero(x[i] > 0)
        if len(ys) == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = x[i, y0:y1, x0:x1]
        nh = int(round(crop.shape[0] * s))
        nw = int(round(crop.shape[1] * s))
        nh, nw = max(1, min(nh, H)), max(1, min(nw, W))
        r = _nn_resize(crop, nh, nw)
        top, left = (H - nh) // 2, (W - nw) // 2
        out[i, top:top + nh, left:left + nw] = r
    return out


def random_translate(x: np.ndarray, rng, max_shift: int = 4) -> np.ndarray:
    """캔버스 안에서 웨이퍼를 옮긴다. pad 표현 전용.

    pad 는 웨이퍼를 중앙에 놓지만 **캔버스 안 위치 자체는 임의**다.
    따라서 이동은 라벨을 보존하는 정당한 대칭이다.
    resize 에는 쓸 수 없다 — 거기서는 웨이퍼가 캔버스를 채운다.
    """
    x = np.asarray(x)
    if max_shift <= 0:
        return x.copy()
    out = np.zeros_like(x)
    n, H, W = x.shape
    for i in range(n):
        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))
        ys0, ys1 = max(0, dy), min(H, H + dy)
        xs0, xs1 = max(0, dx), min(W, W + dx)
        out[i, ys0:ys1, xs0:xs1] = x[i, ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def die_noise(x: np.ndarray, rng, rate: float = 0.01) -> np.ndarray:
    """정상 <-> 불량을 소량 반전한다. 다이가 없는 칸은 건드리지 않는다.

    물리적으로 **측정 잡음**이다. 패턴은 유지한 채 개별 다이를 흔든다.
    none 과 결함의 차이가 "산발적 불량이 구조를 이루는가" 이므로 그 경계를 직접 정규화한다.
    현재 오류의 81.7% 가 그 경계에 있다.

    다이가 없는 칸(0)을 뒤집으면 물리적으로 없는 측정을 만드는 것이라 금지다.
    """
    x = np.asarray(x)
    if rate <= 0:
        return x.copy()
    out = x.copy()
    die = x > 0
    flip = die & (rng.random(x.shape) < rate)
    out[flip] = 3 - out[flip]      # 1 <-> 2
    return out


def die_dropout(x: np.ndarray, rng, rate: float = 0.02) -> np.ndarray:
    """일부 다이를 0(다이 없음)으로 지운다.

    물리적으로 **미측정 다이**이고 실제 웨이퍼에 흔하다.
    모델이 특정 위치의 다이 존재에 의존하지 않게 한다.
    """
    x = np.asarray(x)
    if rate <= 0:
        return x.copy()
    out = x.copy()
    die = x > 0
    out[die & (rng.random(x.shape) < rate)] = 0
    return out


# 학습에서 이름으로 고르기 위한 표. 값은 (함수, 기본 인자).
RECIPES = {
    "scale": (random_scale, dict(lo=0.6, hi=1.4)),
    "translate": (random_translate, dict(max_shift=4)),
    "noise": (die_noise, dict(rate=0.01)),
    "dropout": (die_dropout, dict(rate=0.02)),
}


def apply_recipe(x: np.ndarray, names, rng) -> np.ndarray:
    """이름 목록을 순서대로 적용한다. 빈 목록이면 그대로 돌려준다."""
    out = x
    for name in names:
        if name not in RECIPES:
            raise ValueError(f"모르는 증강: {name}")
        fn, kw = RECIPES[name]
        out = fn(out, rng=rng, **kw)
    return out
