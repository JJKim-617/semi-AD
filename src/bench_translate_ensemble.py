"""translate 모델이 앙상블을 올리는가. 현재 최고 0.7820 을 넘는지가 핵심 질문이다."""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate
from a6_perclass_offset import cache_logits
from bench_ensemble import leave_one_out_delta

NAMES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full"]
POSTHOC = "result/posthoc"
entries = json.loads(open("docs/experiments/ensemble_members.json", encoding="utf-8").read())

# 없는 로짓만 만든다.
for e in entries:
    p = f"{POSTHOC}/{e['tag']}_logits.npz"
    if not os.path.exists(p):
        if not os.path.exists(e["ckpt"]):
            print(f"  [건너뜀] 체크포인트 없음 {e['tag']}"); continue
        print(f"  [로짓 생성] {e['tag']}")
        cache_logits(e["ckpt"], e["cache"], "data/wm811k/cache/splits_v1.npz",
                     POSTHOC, e["tag"], backbone=e.get("backbone", "resnet18"))

P, ty = {}, None
for e in entries:
    p = f"{POSTHOC}/{e['tag']}_logits.npz"
    if not os.path.exists(p):
        continue
    d = np.load(p)
    ty = d["test_y"] if ty is None else ty
    z = d["test_logits"].astype(np.float64)
    ex = np.exp(z - z.max(1, keepdims=True))
    P[e["tag"]] = ex / ex.sum(1, keepdims=True)

single = {t: float(evaluate(ty, P[t].argmax(1), 9)["macro_f1"]) for t in P}


def ens(tags):
    tags = [t for t in tags if t in P]
    m = evaluate(ty, np.mean([P[t] for t in tags], 0).argmax(1), 9)
    return m, len(tags)


PREV10 = ["e8_pad", "e10_pad_s1", "e10_pad_s2", "e7_ce_long_best", "e6_focal_long_best",
          "e6_focal_long_s1_best", "e7_ce_long80_best", "e8_res96", "e11_size", "e11_shuffle"]
PREV11 = PREV10 + ["e15b_polar_s0"]
TRANS = [t for t in P if t.startswith("e17_translate")]
E17_ALL = [t for t in P if t.startswith("e17_")]

print("\n단독 성능 (E17 계열)")
for t in sorted(E17_ALL, key=lambda x: -single[x]):
    print("  %-22s %.4f" % (t, single[t]))

print("\n앙상블 비교")
for name, tags in [
    ("이전 최고 10개", PREV10),
    ("이전 11개 (극좌표 포함)", PREV11),
    ("10개 + translate", PREV10 + TRANS),
    ("11개 + translate", PREV11 + TRANS),
    ("translate 만", TRANS),
    ("10개 + E17 전부", PREV10 + E17_ALL),
    ("가진 것 전부", list(P)),
]:
    m, n = ens(tags)
    print("  %-26s n=%2d  macro-F1 %.4f  acc %.4f" % (name, n, m["macro_f1"], m["accuracy"]))

# leave-one-out: 이전 10개 + translate 로 제한한 풀에서
POOL = PREV10 + TRANS
scores = {}
for k in (2, 3, 4):
    for c in itertools.combinations(POOL, k):
        scores[c] = float(evaluate(ty, np.mean([P[t] for t in c], 0).argmax(1), 9)["macro_f1"])
print("\nleave-one-out (풀 %d개, k=2~4)" % len(POOL))
rows = []
for t in POOL:
    d = leave_one_out_delta(scores, t)
    rows.append((t, float(np.mean([v["delta"] for v in d.values()])), single[t]))
for t, loo, s in sorted(rows, key=lambda r: -r[1]):
    print("  %-22s 단독 %.4f  LOO %+.4f" % (t, s, loo))

best = ens(PREV10 + TRANS)[0]
print("\n최선 구성의 클래스별")
print("  %-11s %8s %8s" % ("클래스", "support", "F1"))
for i, n in enumerate(NAMES):
    print("  %-11s %8d %8.3f" % (n, best["support"][i], best["per_class_f1"][i]))
pred = np.mean([P[t] for t in PREV10 + TRANS], 0).argmax(1)
print("  none -> 결함 누출 %d / 110,701" % int(((ty == 0) & (pred != 0)).sum()))
