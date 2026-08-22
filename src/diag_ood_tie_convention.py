"""관례를 바꾸면 어떤 수치가 움직이는가 — 전부 다시 잰다.

AUROC 는 이미 동점 평균랭크라 영향이 없다. **AUPR 만** 바뀐다.
바뀌는 폭이 큰 arm 이 있으면 그 arm 의 결론을 다시 봐야 한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import (  # noqa: E402
    auroc,
    aupr,
    aupr_blocked,
    fail_count,
    fail_ratio,
)
from a24_ood_residual import local_fail_density_max  # noqa: E402

OUT = Path("docs/research/ood_operating_points/evidence")

sp = np.load("data/wm811k/cache/splits_v1.npz")
d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
y = d["y"].astype(np.int64)
te = sp["test"]
X = np.ascontiguousarray(d["X"][te])
del d
yte = y[te]
is_def = (yte != 0).astype(np.int64)

scores = {
    "E0 불량 다이 비율": fail_ratio(X),
    "E0 불량 다이 개수": fail_count(X),
}
for k in (3, 5, 7, 9):
    scores["국소 밀도 max k=%d" % k] = local_fail_density_max(X, k=k, chunk=4000)
cal = np.load("result/ood/o1_radialcal/radial_calibration_scores.npz", allow_pickle=True)
for key in cal.files:
    if key.startswith("주 "):
        scores["반경보정 밀도 k=7 (float32 저장본)"] = cal[key].astype(np.float64)

print("== 관례별 AUPR ==")
print("%-32s %8s %10s %10s %9s %8s" % (
    "arm", "고유값", "기존(순서의존)", "블록(관례)", "차이", "AUROC"))
rows = {}
for name, s in scores.items():
    a_old, a_new = aupr(s, is_def), aupr_blocked(s, is_def)
    rows[name] = {"aupr_old": a_old, "aupr_blocked": a_new,
                  "n_unique": int(len(np.unique(s))), "auroc": auroc(s, is_def)}
    print("%-32s %8d %10.4f %10.4f %+9.4f %8.4f"
          % (name, len(np.unique(s)), a_old, a_new, a_new - a_old, rows[name]["auroc"]))

print("\n== 순서 의존성 직접 확인 (무작위 순열 6회) ==")
rng = np.random.default_rng(0)
for name in ("E0 불량 다이 비율", "국소 밀도 max k=3", "국소 밀도 max k=7"):
    s = scores[name]
    olds, news = [], []
    for i in range(6):
        p = np.random.default_rng(i).permutation(len(s))
        olds.append(aupr(s[p], is_def[p]))
        news.append(aupr_blocked(s[p], is_def[p]))
    print("  %-24s 기존 폭 %.4f   블록 폭 %.2e"
          % (name, max(olds) - min(olds), max(news) - min(news)))

(OUT / "tie_convention.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
print("\n저장 → %s" % (OUT / "tie_convention.json"))
