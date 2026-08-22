import sys

import numpy as np

sys.path.insert(0, "src")

sp = np.load("data/wm811k/cache/splits_v1.npz")
d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
y = d["y"].astype(int)
ds = d["die_size"].astype(float)
te, tr = sp["test"], sp["train"]
trn = tr[y[tr] == 0]
BINS = [0, 400, 562, 776, 1090, 1334, 1600, 10 ** 9]
L = ["<400", "400-562", "562-776", "776-1090", "1090-1334", "1334-1600", ">1600"]
yte = y[te]
isd = (yte != 0)
print("%-12s %9s %8s %9s %13s" % ("구간", "test n", "비중", "결함률", "train-none n"))
for i in range(7):
    m = (ds[te] >= BINS[i]) & (ds[te] < BINS[i + 1])
    mt = (ds[trn] >= BINS[i]) & (ds[trn] < BINS[i + 1])
    if m.sum() == 0:
        continue
    print("%-12s %9d %7.2f%% %9.4f %13d"
          % (L[i], m.sum(), 100 * m.mean(), isd[m].mean(), mt.sum()))

X = d["X"]
big = te[ds[te] >= 1600]
print()
print("'>1600' test 웨이퍼 %d장 (test 의 %.2f%%)" % (len(big), 100 * len(big) / len(te)))
print("  실제 die 개수 분위 [10,50,90]:",
      np.percentile((X[big] > 0).sum((1, 2)), [10, 50, 90]))
print("  dieSize 분위 [10,50,90]:", np.percentile(ds[big], [10, 50, 90]))
print("  train-none 중 dieSize>=1600: %d장 (%.2f%%)"
      % (int((ds[trn] >= 1600).sum()), 100 * (ds[trn] >= 1600).mean()))
print("  '>1600' 결함 수:", int(isd[ds[te] >= 1600].sum()),
      "/ 전체 결함", int(isd.sum()))
