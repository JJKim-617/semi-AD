"""극좌표가 왜 none 을 결함이라 부르는가 — 웨이퍼 크기 때문인가.

가설. 극좌표는 **반지름으로 정규화**하므로 원본 크기와 무관하게 캔버스를 채운다.
작은 웨이퍼일수록 원본 셀 하나가 극좌표에서 더 넓은 영역을 덮어 블록 artifact 가 커진다.
게다가 중심부는 작은 반지름에도 열이 64개라 원본 몇 셀이 크게 확대된다.

train 웨이퍼는 크고(Scratch median 52x52) test 는 작다(31x31). 그러면 test 의 artifact 가
train 보다 크므로, train 의 artifact 규모에 맞춰진 모델이 test 의 artifact 를 결함으로 읽는다.

**반증 가능한 예측: 극좌표 모델의 none 오검출률은 작은 웨이퍼에서 높아야 한다.
대조군인 pad 모델은 그 기울기가 훨씬 완만해야 한다.**
기울기가 둘 다 평평하면 크기 가설은 틀렸고 다른 이유를 찾아야 한다.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

sp = np.load("data/wm811k/cache/splits_v1.npz")
meta = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
te = sp["test"]
size = meta["die_size"].astype(np.float64)[te]
y = meta["y"].astype(np.int64)[te]

MODELS = [
    ("극좌표 e15b_polar_s0", "result/posthoc/e15b_polar_s0_logits.npz"),
    ("pad     e8_pad",       "result/posthoc/e8_pad_logits.npz"),
    ("resize  e7_ce_long",   "result/posthoc/e7_ce_long_best_logits.npz"),
]

# train 웨이퍼 크기와 비교해 test 가 정말 작은지 먼저 확인한다.
tr_size = meta["die_size"].astype(np.float64)[sp["train"]]
print("dieSize 중앙값  train %.0f   test %.0f   (test/train %.2f)" % (
    np.median(tr_size), np.median(size), np.median(size) / np.median(tr_size)))
print("train 25/50/75 분위 %s" % np.percentile(tr_size, [25, 50, 75]).round(0))
print("test  25/50/75 분위 %s" % np.percentile(size, [25, 50, 75]).round(0))

none_mask = y == 0
edges = np.percentile(size[none_mask], [0, 20, 40, 60, 80, 100])
edges[-1] += 1
print("\nnone 웨이퍼를 크기 5분위로 나눠 오검출률(결함이라 부른 비율)을 본다")
print("%-24s %s" % ("모델", "  ".join("Q%d" % (i + 1) for i in range(5))))
for name, path in MODELS:
    try:
        pred = np.load(path)["test_logits"].argmax(1)
    except FileNotFoundError:
        print("%-24s (로짓 없음: %s)" % (name, path)); continue
    row = []
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        m = none_mask & (size >= lo) & (size < hi)
        row.append((pred[m] != 0).mean())
    slope = row[0] / max(row[-1], 1e-9)
    print("%-24s %s   |  Q1/Q5 = %.1f배" % (
        name, "  ".join("%.3f" % v for v in row), slope))

print("\n(Q1 = 가장 작은 웨이퍼, Q5 = 가장 큰 웨이퍼)")
print("크기 경계 %s" % edges[:-1].round(0))
