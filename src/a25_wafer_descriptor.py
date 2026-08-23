"""위치, 크기, 회전에 불변인 웨이퍼 기술자 (18차).

## 왜

`docs/research/label_ambiguity/README.md` 가 남긴 다음 할 일 1번이다.

> **더 나은 거리.** 위치, 크기에 불변인 기술자로 이웃을 찾아야 한다.
> 해밍 거리는 불량 다이 개수에 끌려간다는 것이 확인됐다(날것 +27.1%p 중
> 절반이 그 교란이었다).

## 무엇을 담나

셋 다 **라벨과 무관한 잡음 변수**를 지운다 — 이번 사이클이 확인한 이 데이터의
대칭(`D4 x 정수 이동`)과 크기 변동에 맞춘 것이다.

1. **반지름 프로파일** — 다이 영역 중심에서의 정규화 반지름 8구간별 불량 비율.
   위치, 크기, **회전 전부에 불변**이다. Center/Donut/Edge-Ring 을 가른다.
2. **각도 집중도** — 구간마다 각도 방향 불량 분포의 푸리에 계수 **크기**
   (1~3차). 크기만 쓰므로 회전에 불변이다. "한쪽에 뭉쳤는가" 를 잡아
   Loc, Edge-Loc 을 Edge-Ring 과 가른다.
3. **퍼짐 정도** — 불량 다이가 자기 무게중심에서 얼마나 흩어져 있는가를
   정규화 반지름으로 잰 값. 작으면 한 덩어리(Loc), 크면 흩어짐(Random, Edge-Ring).
   **칸 단위가 아니라 정규화 길이라 크기에 불변**이다
   (이웃 쌍 개수로 세면 해상도에 끌려간다 — 시험이 그것을 잡았다).

**불량 비율은 기술자에 넣지 않고 `fail_rate` 로 따로 낸다.**
앞선 조사에서 그것이 교란이었으므로 **층으로 통제할 수 있게 분리한다.**
"""

from __future__ import annotations

import numpy as np

N_RADIAL = 8
N_HARMONIC = 3


def fail_rate(x: np.ndarray) -> np.ndarray:
    """다이 칸 중 불량인 비율. **기술자와 분리해 둔다** — 통제 변수로 쓰기 위해서다."""
    x = np.asarray(x)
    die = (x > 0).reshape(len(x), -1).sum(1).astype(np.float64)
    fail = (x == 2).reshape(len(x), -1).sum(1).astype(np.float64)
    return np.where(die > 0, fail / np.maximum(die, 1), np.nan)


def describe(x: np.ndarray) -> np.ndarray:
    """(N,H,W) uint8 {0,1,2} -> (N, D) 기술자. 다이가 없으면 그 행은 전부 NaN."""
    x = np.asarray(x)
    n, h, w = x.shape
    yy, xx = np.mgrid[0:h, 0:w]
    yy = yy.astype(np.float64); xx = xx.astype(np.float64)
    D = N_RADIAL + N_RADIAL * N_HARMONIC + 1
    out = np.full((n, D), np.nan)

    for i in range(n):
        die = x[i] > 0
        if not die.any():
            continue
        fail = x[i] == 2
        cy, cx = yy[die].mean(), xx[die].mean()
        dy, dx = yy - cy, xx - cx
        r = np.hypot(dy, dx)
        # 다이 영역의 최대 반지름으로 정규화한다 -> **크기 불변**.
        rmax = r[die].max()
        if rmax <= 0:
            continue
        rn = r / rmax
        th = np.arctan2(dy, dx)

        prof = np.zeros(N_RADIAL)
        harm = np.zeros((N_RADIAL, N_HARMONIC))
        edges = np.linspace(0, 1.0 + 1e-9, N_RADIAL + 1)
        for b in range(N_RADIAL):
            sel = die & (rn >= edges[b]) & (rn < edges[b + 1])
            nd = int(sel.sum())
            if nd == 0:
                continue
            fsel = fail & sel
            prof[b] = fsel.sum() / nd
            if not fsel.any():
                continue
            # 각도 분포의 푸리에 계수 **크기**만 쓴다 -> **회전 불변**.
            a = th[fsel]
            for k in range(1, N_HARMONIC + 1):
                harm[b, k - 1] = np.abs(np.exp(1j * k * a).mean())

        # 퍼짐: 불량 무게중심으로부터의 RMS 거리를 rmax 로 정규화한다.
        # 칸 개수로 세면 해상도에 끌려가므로 **길이**로 잰다.
        if fail.any():
            fy, fx = yy[fail].mean(), xx[fail].mean()
            spread = float(np.sqrt(((yy[fail] - fy) ** 2
                                    + (xx[fail] - fx) ** 2).mean()) / rmax)
        else:
            spread = 0.0
        out[i] = np.concatenate([prof, harm.ravel(), [spread]])
    return out


def main():
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(description="기술자를 미리 계산해 저장한다")
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64pad.npz")
    p.add_argument("--out", default="result/cls_baseline/wafer_descriptor.npz")
    a = p.parse_args()
    d = np.load(a.cache)
    X = d["X"]
    print(f"{len(X):,}장 기술자 계산")
    Z = describe(X)
    fr = fail_rate(X)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, desc=Z.astype(np.float32),
                        fail_rate=fr.astype(np.float32))
    print(f"[저장] {a.out}  차원 {Z.shape[1]}  NaN 행 {int(np.isnan(Z).any(1).sum())}")


if __name__ == "__main__":
    main()
