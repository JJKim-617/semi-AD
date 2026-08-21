"""극좌표는 중간 크기 웨이퍼에서 무너진다. 그 구간에 학습 데이터가 있는가."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

sp = np.load("data/wm811k/cache/splits_v1.npz")
meta = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
size = meta["die_size"].astype(np.float64)
y = meta["y"].astype(np.int64)
tr, te = sp["train"], sp["test"]

BINS = [0, 400, 562, 776, 1090, 1334, 1600, 10 ** 9]
LBL = ["<400", "400-562", "562-776", "776-1090", "1090-1334", "1334-1600", ">1600"]

print("웨이퍼 크기 구간별 표본 수와, 극좌표 모델의 none 오검출률")
polar = np.load("result/posthoc/e15b_polar_s0_logits.npz")["test_logits"].argmax(1)
pad = np.load("result/posthoc/e8_pad_logits.npz")["test_logits"].argmax(1)
yte = y[te]
ste = size[te]
str_ = size[tr]

print("%-12s %10s %10s %12s %10s %10s" % (
    "크기 구간", "train n", "test n", "train 비중", "극좌표 FP", "pad FP"))
for i in range(len(BINS) - 1):
    lo, hi = BINS[i], BINS[i + 1]
    ntr = int(((str_ >= lo) & (str_ < hi)).sum())
    m = (ste >= lo) & (ste < hi)
    nte = int(m.sum())
    mn = m & (yte == 0)
    fp_p = (polar[mn] != 0).mean() if mn.sum() else float("nan")
    fp_d = (pad[mn] != 0).mean() if mn.sum() else float("nan")
    print("%-12s %10d %10d %11.2f%% %10.3f %10.3f" % (
        LBL[i], ntr, nte, 100 * ntr / len(str_), fp_p, fp_d))

print()
print("train 크기 분포가 얼마나 몰려 있는가")
vals, counts = np.unique(str_, return_counts=True)
order = np.argsort(-counts)[:8]
print("  가장 흔한 dieSize 8개 (train %d장 중)" % len(str_))
for i in order:
    print("    %8.0f  %6d장 (%.1f%%)" % (vals[i], counts[i], 100 * counts[i] / len(str_)))
top2 = counts[np.argsort(-counts)[:2]].sum()
print("  상위 2개가 train 의 %.1f%% 를 차지한다" % (100 * top2 / len(str_)))

m_gap = (ste >= 562) & (ste < 1090)
n_gap_tr = int(((str_ >= 562) & (str_ < 1090)).sum())
print()
print("무너지는 구간(562~1090): train %d장(%.1f%%), test %d장(%.1f%%)" % (
    n_gap_tr, 100 * n_gap_tr / len(str_), int(m_gap.sum()), 100 * m_gap.mean()))
