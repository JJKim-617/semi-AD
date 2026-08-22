"""선 필터는 왜 무너졌나 — 포화가 어디서 일어나는지 직접 본다.

주 arm(`line7_8dir`)이 AUPR 0.0804 로 유병률(0.0666) 바로 위였다.
클래스별 AUROC 가 다섯 클래스에서 **정확히 같은 값(0.6490)** 이었고,
점수의 고유값이 13개뿐이었다. 포화다 — **정상 웨이퍼의 70.21% 가 어딘가에
불량 7칸짜리 직선을 갖고 있다.**

## 첫 설명은 양적으로 틀렸다

"창 후보가 많아서(8방향 x 수천 위치) 다중비교로 우연히 걸린다" 고 생각했다.
그런데 계산하면 안 맞는다. 불량률 r=0.116, 다이 800개, 8방향이면
독립 Bernoulli 가정에서 기대 포화 창 수는 8*800*0.116^7 = 0.002 다.
**0.2% 여야 하는데 70% 다.** 그러니 원인은 우연이 아니라
**정상 웨이퍼의 불량이 실제로 뭉쳐 있다**는 것이다.

어디에 뭉쳐 있는가? 가장자리 링이 후보다 — 반경 template 에서 바깥 빈의
불량 확률이 0.34 였고, 링을 따라 놓인 선 창은 연속된 불량을 만나기 쉽다.
**이 스크립트가 그것을 확인한다.**
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import evaluate_ood  # noqa: E402
from a27_line_filter import line_density_map, line_density_max, radius_mask  # noqa: E402

NAMES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch",
         "Near-full"]
sp = np.load("data/wm811k/cache/splits_v1.npz")
d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
y = d["y"].astype(np.int64)
te = sp["test"]
X = d["X"]
rng = np.random.default_rng(0)
sub = np.sort(rng.choice(len(te), 8000, replace=False))
x = np.ascontiguousarray(X[te[sub]])
yy = y[te[sub]]
del X, d
print("표본 %d장 (정상 %d / 결함 %d)" % (len(x), int((yy == 0).sum()), int((yy != 0).sum())))

m = line_density_map(x, length=7, n_orient=8)
h = w = x.shape[1]
die = x > 0
gy, gx = np.mgrid[0:h, 0:w].astype(np.float64)
n = np.maximum(die.sum((1, 2)), 1).astype(np.float64)
cy = (die * gy).sum((1, 2)) / n
cx = (die * gx).sum((1, 2)) / n
r = np.sqrt((gy[None] - cy[:, None, None]) ** 2 + (gx[None] - cx[:, None, None]) ** 2)
rmax = np.maximum(np.where(die, r, -np.inf).max((1, 2)), 1e-9)[:, None, None]
u = r / rmax

flat = m.reshape(len(m), -1)
best = flat.argmax(1)
u_best = u.reshape(len(m), -1)[np.arange(len(m)), best]
sat = flat.max(1) >= 0.999

print("\n== 포화된 창은 어디에 있는가 (정규화 반경) ==")
for name, mask in (("정상 (포화)", (yy == 0) & sat), ("정상 (비포화)", (yy == 0) & ~sat),
                   ("결함 (포화)", (yy != 0) & sat)):
    if mask.sum() == 0:
        continue
    v = u_best[mask]
    print("  %-14s n=%5d  반경 분위 [10,25,50,75,90] = %s"
          % (name, int(mask.sum()), np.round(np.percentile(v, [10, 25, 50, 75, 90]), 3)))
print("\n정상 중 포화 %.2f%%, 그중 반경 0.85 바깥에서 일어난 비율 %.2f%%"
      % (100 * sat[yy == 0].mean(),
         100 * (u_best[(yy == 0) & sat] > 0.85).mean()))

print("\n== 기작 확인: 안쪽만 보면 포화가 줄어드는가 ==")
inner = radius_mask(x, r_min=0.0, r_max=0.85)
s_all = line_density_max(x, length=7, n_orient=8)
s_in = line_density_max(x, length=7, n_orient=8, valid=inner)
isd = (yy != 0).astype(np.int64)
for tag, s in (("선7 전체", s_all), ("선7 안쪽만(r<=0.85)", s_in)):
    e = evaluate_ood(s, isd)
    print("  %-22s 포화 정상 %5.2f%% / 결함 %5.2f%%  AUROC %.4f AUPR %.4f"
          % (tag, 100 * (s[yy == 0] >= 0.999).mean(), 100 * (s[yy != 0] >= 0.999).mean(),
             e["auroc"], e["aupr"]))

print("\n(주의: 이 안쪽-전용 arm 은 실패 원인을 확인하려고 **결과를 보고 만든 것**이다.")
print(" 사전등록된 arm 이 아니므로 채택 근거로 쓰지 않는다. 기작 진단으로만 읽는다.)")
