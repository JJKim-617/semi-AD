"""E18 백본 판정 + translate 4 seed 반영 앙상블 재집계."""
from __future__ import annotations
import itertools, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate
from a6_perclass_offset import cache_logits
from bench_ensemble import leave_one_out_delta

NAMES = ["none","Center","Donut","Edge-Loc","Edge-Ring","Loc","Random","Scratch","Near-full"]
POSTHOC = "result/posthoc"
entries = json.loads(open("docs/experiments/ensemble_members.json", encoding="utf-8").read())
for e in entries:
    p = f"{POSTHOC}/{e['tag']}_logits.npz"
    if not os.path.exists(p):
        if not os.path.exists(e["ckpt"]):
            continue
        print(f"  [로짓] {e['tag']}", flush=True)
        cache_logits(e["ckpt"], e["cache"], "data/wm811k/cache/splits_v1.npz",
                     POSTHOC, e["tag"], backbone=e.get("backbone", "resnet18"))

P, ty, ARCH = {}, None, {}
for e in entries:
    p = f"{POSTHOC}/{e['tag']}_logits.npz"
    if not os.path.exists(p): continue
    d = np.load(p); ty = d["test_y"] if ty is None else ty
    z = d["test_logits"].astype(np.float64); ex = np.exp(z - z.max(1, keepdims=True))
    P[e["tag"]] = ex / ex.sum(1, keepdims=True)
    ARCH[e["tag"]] = e.get("architecture", "resnet18")
single = {t: float(evaluate(ty, P[t].argmax(1), 9)["macro_f1"]) for t in P}

def ens(tags):
    tags = [t for t in tags if t in P]
    return evaluate(ty, np.mean([P[t] for t in tags], 0).argmax(1), 9), len(tags)

PREV11 = ["e8_pad","e10_pad_s1","e10_pad_s2","e7_ce_long_best","e6_focal_long_best",
          "e6_focal_long_s1_best","e7_ce_long80_best","e8_res96","e11_size","e11_shuffle",
          "e15b_polar_s0"]
TRANS = sorted(t for t in P if t.startswith("e17_translate"))
BB = sorted(t for t in P if t.startswith("e18_"))

print("\nE18 백본 단독 (resnet18 기준선 3 seed 평균 0.7261)")
byb = defaultdict(list)
for t in BB: byb[ARCH[t]].append(single[t])
for b, v in sorted(byb.items(), key=lambda kv: -np.mean(kv[1])):
    print("  %-16s %s  평균 %.4f  범위 %.4f  대비 %+.4f" % (
        b, " / ".join("%.4f" % x for x in v), np.mean(v), max(v)-min(v), np.mean(v)-0.7261))

print("\n앙상블")
for name, tags in [
    ("이전 11개", PREV11),
    ("11개 + translate 2", PREV11 + TRANS[:2]),
    ("11개 + translate 전부", PREV11 + TRANS),
    ("11개 + translate + 백본", PREV11 + TRANS + BB),
    ("translate 만", TRANS),
    ("translate + 백본", TRANS + BB),
]:
    m, n = ens(tags)
    print("  %-26s n=%2d  macro-F1 %.4f  acc %.4f" % (name, n, m["macro_f1"], m["accuracy"]))

POOL = PREV11 + TRANS + BB
scores = {}
for k in (2, 3):
    for c in itertools.combinations(POOL, k):
        scores[c] = float(evaluate(ty, np.mean([P[t] for t in c], 0).argmax(1), 9)["macro_f1"])
print("\nleave-one-out (풀 %d개, k=2~3) 상위/하위" % len(POOL))
rows = [(t, float(np.mean([v["delta"] for v in leave_one_out_delta(scores, t).values()])), single[t], ARCH[t]) for t in POOL]
rows.sort(key=lambda r: -r[1])
for t, loo, s, a in rows[:6] + [("...", 0, 0, "")] + rows[-4:]:
    if t == "...": print("  ..."); continue
    print("  %-22s %-16s 단독 %.4f  LOO %+.4f" % (t, a, s, loo))

print("\n구조 가짓수별 (크기 3 조합, 백본 포함 풀)")
g = defaultdict(list)
for c in (c for c in scores if len(c) == 3):
    g[len({ARCH[t] for t in c})].append(scores[c])
for n_arch in sorted(g):
    v = g[n_arch]
    print("  구조 %d종  n=%4d  평균 %.4f  최고 %.4f" % (n_arch, len(v), np.mean(v), max(v)))

best_tags = PREV11 + TRANS
m, _ = ens(best_tags)
pred = np.mean([P[t] for t in best_tags if t in P], 0).argmax(1)
print("\n최선(11+translate 전부) 클래스별")
for i, n in enumerate(NAMES):
    print("  %-11s %8d %8.3f" % (n, m["support"][i], m["per_class_f1"][i]))
print("  none -> 결함 누출 %d" % int(((ty == 0) & (pred != 0)).sum()))
