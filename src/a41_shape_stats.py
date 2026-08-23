"""결함 형태의 기술 통계 — **채점기가 아니다.**

## 무엇을 위한 것인가

두 곳에서 쓴다.

1. **12차 사이클**: `val_unseen` 의 Scratch AUROC 이 0.7835 (n=92) 로 낮았다.
   dev 의 Scratch 와 **모양이 실제로 다른지**를 재려면 "긁힘이 가늘고 긴가",
   "굵은가", "이어져 있는가" 를 숫자로 만들어야 한다.
   사전등록 `docs/experiments/candidate/ood_scratch_shift.md` §5.
2. **3단계(MLLM 8종 분류)**: 하류에 넘길 파생 특징 후보다.
   지난 감사에서 **탐지 최선이 분류 최선이 아니었고**(자명한 불량 다이 개수가
   28쌍 중 18에서 1위), Scratch 대 Center 는 개수 0.599 인데
   파생 특징 log(B)-log(C) 가 0.909 였다. 모양 통계는 그 계열이다.

## 범위 선언 — 이것은 C 가 아니다

여기 함수들은 **어떤 arm 의 점수 계산에도 들어가지 않는다.**
결함 웨이퍼에 적용해 분포를 기술하는 데만 쓴다. 라벨은 **표본을 고르는 데만** 쓴다.
채점기에 넣는 순간 범위 C(결함 데이터 사용)가 되므로 그렇게 쓰지 않는다.

## 정의되지 않는 값은 0 이 아니라 nan 이다

불량 다이가 없는 웨이퍼의 "선 굵기" 는 0 이 아니라 **없다.**
0 으로 채우면 평균과 검정이 조용히 오염된다. 그래서 nan 을 돌려주고
`mannwhitney_u` 가 nan 을 뺀다.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


def mean_fail_neighbors(x) -> np.ndarray:
    """불량 다이 하나가 평균 몇 개의 불량 8-이웃을 갖는가. **선 굵기의 대리값.**

    가는 선(1칸 폭)은 이웃이 최대 2 이고, 굵은 덩어리는 최대 8 이다.
    길이에는 거의 안 변하고 굵기에만 변하므로 "긁힘이 굵어졌나" 를 길이와 분리해 본다.

    불량이 없으면 **nan**(0 이 아니다).
    """
    f = (np.asarray(x) == 2).astype(np.float64)
    ker = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    nb = np.stack([ndimage.convolve(f[i], ker, mode="constant", cval=0.0)
                   for i in range(len(f))])
    n = f.reshape(len(f), -1).sum(1)
    tot = (nb * f).reshape(len(f), -1).sum(1)
    out = np.full(len(f), np.nan)
    ok = n > 0
    out[ok] = tot[ok] / n[ok]
    return out


def component_shape(x) -> dict:
    """불량 다이의 **최대 8-연결성분**의 크기와 모양.

    - `size` — 다이 개수. 성분이 없으면 0 (**이건 사실이다**)
    - `elongation` — 관성 텐서 고윳값의 비 sqrt(l_max/l_min). 정사각 덩어리는 1,
      가늘고 길수록 커진다. 성분이 없으면 nan (**모양은 정의되지 않는다**)
    - `major` — 주축 방향 4*sqrt(l_max). 성분의 길이 대리값. 없으면 nan

    **90도 회전과 뒤집기에 정확히 불변**이다(격자 정렬이 유지되므로).
    한 칸짜리 성분은 관성이 0 이라 비가 0/0 이 된다 — 관례로 `elongation=1`,
    `major=1` 로 둔다(가장 안 늘어난 모양).
    """
    x = np.asarray(x)
    b = len(x)
    size = np.zeros(b, np.float64)
    elong = np.full(b, np.nan)
    major = np.full(b, np.nan)
    for i in range(b):
        f = x[i] == 2
        if not f.any():
            continue
        lab, n = ndimage.label(f, structure=CONNECTIVITY_8)
        counts = np.bincount(lab.ravel())[1:]
        k = int(np.argmax(counts)) + 1
        size[i] = float(counts[k - 1])
        r, c = np.nonzero(lab == k)
        if len(r) == 1:
            elong[i], major[i] = 1.0, 1.0
            continue
        rr = r - r.mean()
        cc = c - c.mean()
        m = len(r)
        cov = np.array([[(rr * rr).sum(), (rr * cc).sum()],
                        [(rr * cc).sum(), (cc * cc).sum()]]) / m
        w = np.linalg.eigvalsh(cov)
        lo, hi = float(max(w[0], 0.0)), float(max(w[1], 0.0))
        # 한 줄짜리 선은 lo == 0 이라 비가 발산한다. 다이 한 칸의 관성(1/12)을 바닥으로 둔다.
        floor = 1.0 / 12.0
        elong[i] = float(np.sqrt((hi + floor) / (lo + floor)))
        major[i] = float(4.0 * np.sqrt(hi + floor))
    return {"size": size, "elongation": elong, "major": major}


def shape_stats(x, die_size=None) -> dict:
    """웨이퍼별 형태 통계 묶음. 12차 사이클 §5 가 쓰는 여섯 개 + 크기."""
    x = np.asarray(x)
    b = len(x)
    n_die = (x > 0).reshape(b, -1).sum(1).astype(np.float64)
    n_fail = (x == 2).reshape(b, -1).sum(1).astype(np.float64)
    cs = component_shape(x)
    return {
        "die_size": (np.asarray(die_size, np.float64) if die_size is not None
                     else n_die),
        "n_fail": n_fail,
        "fail_ratio": n_fail / np.maximum(n_die, 1.0),
        "cc_size": cs["size"],
        "neighbors": mean_fail_neighbors(x),
        "elongation": cs["elongation"],
        "major": cs["major"],
    }


def mannwhitney_u(a, b) -> tuple[float, float]:
    """양측 Mann-Whitney U 와 p. **동점 보정과 nan 제거를 포함한다.**

    `scipy.stats.mannwhitneyu` 와 같은 값을 준다는 것을 테스트로 박았다
    (`tests/test_a41_shape_stats.py`). 직접 구현하는 이유는 nan 처리를
    호출부마다 반복하지 않기 위해서다 — 굵기 대리는 불량이 없으면 nan 이다.

    표본이 작으면 정규 근사가 나쁘다. **n<20 인 쪽이 있으면 p 를 믿지 마라.**
    """
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    z = np.concatenate([a, b])
    order = np.argsort(z, kind="mergesort")
    s = z[order]
    ranks = np.empty(len(s), np.float64)
    bounds = np.flatnonzero(np.concatenate([[True], s[1:] != s[:-1], [True]]))
    tie_term = 0.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        ranks[lo:hi] = (lo + hi - 1) / 2.0 + 1.0
        t = hi - lo
        tie_term += t ** 3 - t
    r = np.empty(len(s), np.float64)
    r[order] = ranks
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    mu = n1 * n2 / 2.0
    sd = np.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))))
    if sd == 0:
        return float(u1), 1.0
    zstat = (abs(u1 - mu) - 0.5) / sd            # 연속성 보정
    from scipy import special
    p = float(special.erfc(zstat / np.sqrt(2.0)))
    return float(u1), min(p, 1.0)
