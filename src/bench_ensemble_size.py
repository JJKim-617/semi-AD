"""앙상블 크기 곡선과 all-8 의 클래스별 내역."""
from __future__ import annotations

import itertools
import sys

import numpy as np

sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate

NC = 9
NAMES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full"]
P = "result/posthoc/%s_logits.npz"
MEMBERS = {
    "e8_pad": P % "e8_pad", "e10_pad_s1": P % "e10_pad_s1", "e10_pad_s2": P % "e10_pad_s2",
    "e7_ce_long": P % "e7_ce_long_best", "e6_focal_long": P % "e6_focal_long_best",
    "e6_focal_long_s1": P % "e6_focal_long_s1_best", "e7_ce_long80": P % "e7_ce_long80_best",
    "e8_res96": P % "e8_res96",
}

probs, ty = {}, None
for tag, path in MEMBERS.items():
    d = np.load(path)
    ty = d["test_y"] if ty is None else ty
    z = d["test_logits"].astype(np.float64)
    e = np.exp(z - z.max(1, keepdims=True))
    probs[tag] = e / e.sum(1, keepdims=True)

tags_all = list(MEMBERS)
print("앙상블 크기 곡선 (모든 조합, 사후 선택 없음)")
print("%3s %5s %10s %10s %10s" % ("k", "조합수", "평균", "최저", "최고"))
for k in range(1, len(tags_all) + 1):
    scores = [float(evaluate(ty, np.mean([probs[t] for t in c], 0).argmax(1), NC)["macro_f1"])
              for c in itertools.combinations(tags_all, k)]
    print("%3d %5d %10.4f %10.4f %10.4f" % (k, len(scores), np.mean(scores), min(scores), max(scores)))

full = np.mean([probs[t] for t in tags_all], 0)
m = evaluate(ty, full.argmax(1), NC)
print()
print("all-8  macro-F1 %.4f  accuracy %.4f" % (m["macro_f1"], m["accuracy"]))
print()
best_single = {i: max(float(evaluate(ty, probs[t].argmax(1), NC)["per_class_f1"][i])
                      for t in tags_all) for i in range(NC)}
print("%-11s %8s %10s %10s" % ("클래스", "support", "all-8 F1", "최고단독"))
for i, n in enumerate(NAMES):
    print("%-11s %8d %10.3f %10.3f" % (n, m["support"][i], m["per_class_f1"][i], best_single[i]))

print()
print("none -> 결함 누출 (all-8 vs 개별 범위)")
pred = full.argmax(1)
print("%-11s %10s %18s" % ("클래스", "all-8", "개별 범위"))
tot = 0
for i, n in enumerate(NAMES):
    if i == 0:
        continue
    e = int(((ty == 0) & (pred == i)).sum())
    rng = [int(((ty == 0) & (probs[t].argmax(1) == i)).sum()) for t in tags_all]
    tot += e
    print("%-11s %10d %18s" % (n, e, "%d ~ %d" % (min(rng), max(rng))))
print("%-11s %10d" % ("합계", tot))
