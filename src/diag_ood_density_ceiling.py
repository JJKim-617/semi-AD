"""밀도 기반 채점기의 천장은 어디인가 — none 라벨의 정체를 확인한다.

O1 대표 사례를 그려 보니 '헛경보 정상' 과 '잘 잡은 Scratch' 가 눈으로 구분되지 않았다.
둘 다 불량 다이가 30% 안팎으로 흩뿌려져 있다. 그래서 라벨의 뜻을 직접 확인한다.

**WM-811K 의 `none` 은 "불량 다이가 없다" 가 아니라 "알아볼 만한 패턴이 없다" 이다.**
이 구분이 맞다면 밀도(개수/비율) 기반 채점기에는 데이터가 정한 천장이 있고,
그 천장은 방법을 바꿔도 안 올라간다.

같이 확인하는 것: 중앙 다이의 불량률. O1 의 구조 점수가 Center 에서 무작위 아래로
떨어진 이유가 여기 있다고 의심된다.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import fail_ratio  # noqa: E402

NAMES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch",
         "Near-full"]

sp = np.load("data/wm811k/cache/splits_v1.npz")
d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
y = d["y"].astype(np.int64)
te, tr = sp["test"], sp["train"]
X = d["X"]
yte = y[te]
r = fail_ratio(X[te])

print("== 정상과 결함의 불량 다이 비율이 얼마나 겹치는가 (공식 test) ==")
print("test none  분위 [50,75,90,95,99,99.9]:",
      np.round(np.percentile(r[yte == 0], [50, 75, 90, 95, 99, 99.9]), 4))
print("test 결함  분위 [ 1, 5,10,25,50,75  ]:",
      np.round(np.percentile(r[yte != 0], [1, 5, 10, 25, 50, 75]), 4))
print()
for t in (0.10, 0.15, 0.20, 0.30):
    nn = int((r[yte == 0] > t).sum())
    nd = int((r[yte != 0] > t).sum())
    print("  불량률>%0.2f: 정상 %6d장 (%5.2f%%) / 결함 %5d장 (%5.2f%%)"
          % (t, nn, 100 * nn / (yte == 0).sum(), nd, 100 * nd / (yte != 0).sum()))

print("\n== 클래스별 불량 다이 비율 ==")
print("%-11s %7s %9s %9s" % ("클래스", "n", "중앙값", "하위10%"))
for c in range(9):
    m = yte == c
    if m.sum():
        print("%-11s %7d %9.4f %9.4f"
              % (NAMES[c], int(m.sum()), np.median(r[m]), np.percentile(r[m], 10)))

print("\nScratch 중앙값이 none 중앙값의 %.2f 배다. 밀도로는 갈리지 않는다."
      % (np.median(r[yte == 7]) / np.median(r[yte == 0])))

print("\n== 정상 웨이퍼의 중앙 다이는 왜 그렇게 자주 불량인가 ==")
trn = tr[y[tr] == 0]
Xt = X[trn]
die = Xt > 0
yy, xx = np.mgrid[0:64, 0:64].astype(np.float64)
n = die.sum((1, 2))
cy = (die * yy).sum((1, 2)) / n
cx = (die * xx).sum((1, 2)) / n
rad = np.sqrt((yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2)
rad = np.where(die, rad, 1e9)
flat = rad.reshape(len(Xt), -1).argmin(1)
ii, jj = np.unravel_index(flat, (64, 64))
centre = Xt[np.arange(len(Xt)), ii, jj]
print("train-none 중심 최근접 다이의 불량 비율 %.4f (전체 다이 불량률 %.4f)"
      % ((centre == 2).mean(), (Xt == 2).sum() / die.sum()))
for k in (1.5, 3.0, 5.0, 10.0):
    m = die & (rad <= k)
    print("  중심 반경<=%4.1f 다이 불량률 %.4f (다이 %d개)"
          % (k, (Xt[m] == 2).mean(), int(m.sum())))
print("\n중심 다이가 정상 웨이퍼에서 이미 59% 불량이라, Center 결함은 template 이")
print("이미 높은 자리에 불량을 놓는 셈이 된다. 구조 점수가 Center 에서 무작위 아래로")
print("떨어진 것(0.391)이 이것으로 설명된다. 왜 중심 다이가 그런지는 모른다.")
