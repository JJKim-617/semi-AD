"""방향성 선형 필터 — 창 모양을 결함 모양에 맞춘다. 학습 없음, CPU 전용.

## 왜

정사각 창 국소 밀도가 현재 바닥이다(7x7, AUPR 0.7147). 그런데 **놓치는 이유가
그림에서 보였다**: 놓친 Scratch 는 선이 창보다 가늘다. 7x7 창에 길이 7 짜리
1픽셀 선이 들어오면 밀도가 7/49 = 0.14 인데, 같은 창의 5x5 덩어리는 20/49 = 0.41 이다.
**정사각 창은 선을 조직적으로 희석한다.**

창을 선 모양으로 바꾸면 같은 선이 1.0 이 된다.
이것은 학습이 아니라 **결함 모양에 대한 사전지식을 창 모양에 넣는 것**이고,
train 자료를 여전히 쓰지 않는다.

## 방향은 최대로 결합한다

어느 방향인지 모르므로 8방향을 다 놓고 최대를 취한다.
평균이 아니라 최대인 이유: 선은 한 방향으로만 뻗으므로 나머지 7방향은 잡음이다.
평균을 내면 다시 희석된다.

## 크기 불변

정사각 창과 같은 이유로 불변이다 — **창 안 다이 개수로 나눈다.**
웨이퍼가 커도 작아도 "이 창 안에서 몇 %가 죽었나" 는 같은 뜻이다.
그래서 dieSize 구간별로 안 무너진다(정사각 창에서 실측 확인됨).
"""

from __future__ import annotations

import numpy as np


def line_kernels(length: int = 7, n_orient: int = 8, width: int = 1) -> list[np.ndarray]:
    """중심을 지나는 선 모양 구조요소를 방향마다 하나씩.

    각도는 [0, pi) 를 균등 분할한다. 선은 방향이 없으므로 pi 이상은 중복이다.

    **t 를 정수 격자에서만 뽑는다.** 처음에 촘촘히(4*length 점) 뽑았더니
    비스듬한 방향에서 계단이 두꺼워져 22.5° 커널이 9칸이 됐다(축 방향은 7칸).
    그러면 방향마다 창이 보는 셀 수가 달라져 **비스듬한 긁힘만 더 희석된다** —
    이 필터를 만든 이유 자체를 무너뜨린다. 정수 t 로 뽑으면 어느 방향이든 7칸이다.
    이산화로 겹치는 칸이 생겨 length 보다 **작아질** 수는 있다.
    """
    if length % 2 == 0:
        raise ValueError("length 는 홀수여야 중심이 정해진다: %d" % length)
    if width < 1:
        raise ValueError("width 는 1 이상: %d" % width)
    half = (length - 1) // 2
    ws = np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    out = []
    for a in range(n_orient):
        th = np.pi * a / n_orient
        k = np.zeros((length, length), np.uint8)
        for t in range(-half, half + 1):
            for w in ws:
                i = int(round(half + t * np.sin(th) + w * np.cos(th)))
                j = int(round(half + t * np.cos(th) - w * np.sin(th)))
                if 0 <= i < length and 0 <= j < length:
                    k[i, j] = 1
        out.append(k)
    return out


def _offsets(k: np.ndarray) -> list[tuple[int, int]]:
    """커널의 1 인 칸을 중심 기준 변위로. 선 커널은 7칸뿐이라 49탭 합성곱이 낭비다."""
    h = (k.shape[0] - 1) // 2
    ii, jj = np.nonzero(k)
    return [(int(i) - h, int(j) - h) for i, j in zip(ii, jj)]


def _shift_sum(a: np.ndarray, offs: list[tuple[int, int]]) -> np.ndarray:
    """변위 합. 커널이 중심 대칭이라 상관과 합성곱이 같다."""
    b, h, w = a.shape
    out = np.zeros_like(a)
    for di, dj in offs:
        i0, i1 = max(0, -di), min(h, h - di)
        j0, j1 = max(0, -dj), min(w, w - dj)
        if i0 < i1 and j0 < j1:
            out[:, i0 + di:i1 + di, j0 + dj:j1 + dj] += a[:, i0:i1, j0:j1]
    return out


def line_density_map(x, length: int = 7, n_orient: int = 8, width: int = 1,
                     valid=None, chunk: int = 2000) -> np.ndarray:
    """다이별 **방향 최대 선 밀도** 맵. 창 안 다이 개수로 나눈다.

    `valid` 를 주면 그 마스크의 다이만 분모와 분자에 넣는다
    (중심 다이 제외 arm 에 쓴다).
    """
    x = np.asarray(x)
    offs = [_offsets(k) for k in line_kernels(length, n_orient, width)]
    out = np.zeros(x.shape, np.float32)
    for a in range(0, len(x), chunk):
        xb = x[a:a + chunk]
        keep = (xb > 0) if valid is None else np.asarray(valid[a:a + chunk])
        die = keep.astype(np.float32)
        fail = ((xb == 2) & keep).astype(np.float32)
        best = np.zeros(xb.shape, np.float32)
        for o in offs:
            num = _shift_sum(fail, o)
            den = _shift_sum(die, o)
            np.maximum(best, num / np.maximum(den, 1e-12), out=best)
        out[a:a + len(xb)] = np.where(keep, best, 0.0)
    return out.astype(np.float64)


def line_density_max(x, length: int = 7, n_orient: int = 8, width: int = 1,
                     valid=None, chunk: int = 2000) -> np.ndarray:
    """위 맵의 최대. 주 arm 은 `length=7, n_orient=8, width=1`."""
    x = np.asarray(x)
    out = np.empty(len(x), np.float64)
    for a in range(0, len(x), chunk):
        v = None if valid is None else valid[a:a + chunk]
        m = line_density_map(x[a:a + chunk], length, n_orient, width, v, chunk)
        out[a:a + len(m)] = m.reshape(len(m), -1).max(1)
    return out


def radius_mask(x, r_min: float = 0.0, r_max: float = 1.0) -> np.ndarray:
    """웨이퍼 **자기 die 반경**으로 정규화한 반경이 [r_min, r_max] 인 다이만 True.

    정상 웨이퍼의 중심 최근접 다이가 58.9% 불량이라(전체 11.6%),
    그 한 칸이 밀도 지도에서 늘 켜진다. `r_min` 으로 빼고 재는 arm 을 만든다.
    """
    x = np.asarray(x)
    b, h, w = x.shape
    die = x > 0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    n = np.maximum(die.sum((1, 2)), 1).astype(np.float64)
    cy = (die * yy).sum((1, 2)) / n
    cx = (die * xx).sum((1, 2)) / n
    r = np.sqrt((yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2)
    rmax = np.maximum(np.where(die, r, -np.inf).max((1, 2)), 1e-9)[:, None, None]
    u = r / rmax
    return die & (u >= r_min) & (u <= r_max)
