"""TTA 가 앙상블의 보완재인가 대체재인가.

사전 등록한 판정: TTA 앙상블이 같은 구성원의 일반 앙상블을 넘어야 보완재다.
못 넘으면 둘은 같은 분산을 잡는 것이므로 TTA 는 앙상블의 대체재일 뿐이다.

비교는 반드시 **같은 구성원 집합**으로 한다. TTA 로짓이 있는 모델만 쓴다.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate

NAMES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full"]
POSTHOC = "result/posthoc"

entries = json.loads(open("docs/experiments/ensemble_members.json", encoding="utf-8").read())
members = []
for e in entries:
    base = f"{POSTHOC}/{e['tag']}_logits.npz"
    tta = f"{POSTHOC}/{e['tag']}_tta_logits.npz"
    if os.path.exists(base) and os.path.exists(tta):
        members.append(e["tag"])

print("TTA 로짓이 있는 구성원 %d개: %s\n" % (len(members), ", ".join(members)))


def probs(path):
    z = np.load(path)["test_logits"].astype(np.float64)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


ty = np.load(f"{POSTHOC}/{members[0]}_logits.npz")["test_y"]
P = {t: probs(f"{POSTHOC}/{t}_logits.npz") for t in members}
T = {t: probs(f"{POSTHOC}/{t}_tta_logits.npz") for t in members}


def ens(tags, table):
    m = evaluate(ty, np.mean([table[t] for t in tags], 0).argmax(1), 9)
    return m["macro_f1"], m["accuracy"]


print("단독 비교 (짝지은 비교, 학습 분산 없음)")
print("%-24s %9s %9s %9s" % ("run", "기본", "TTA", "차이"))
deltas = []
for t in members:
    a = float(evaluate(ty, P[t].argmax(1), 9)["macro_f1"])
    b = float(evaluate(ty, T[t].argmax(1), 9)["macro_f1"])
    deltas.append(b - a)
    print("%-24s %9.4f %9.4f %+9.4f" % (t, a, b, b - a))
print("  평균 %+.4f   개선 %d/%d" % (np.mean(deltas), sum(d > 0 for d in deltas), len(deltas)))

print("\n앙상블 비교 (같은 구성원 %d개)" % len(members))
pf, pa = ens(members, P)
tf, ta = ens(members, T)
print("  일반 앙상블   macro-F1 %.4f  acc %.4f" % (pf, pa))
print("  TTA  앙상블   macro-F1 %.4f  acc %.4f   (%+.4f)" % (tf, ta, tf - pf))

both = {**{f"{t}": P[t] for t in members}, **{f"{t}_tta": T[t] for t in members}}
bf, ba = ens(list(both), both)
print("  둘 다 합침    macro-F1 %.4f  acc %.4f   (%+.4f)" % (bf, ba, bf - pf))

print("\n크기별 (모든 조합, 일반 vs TTA)")
print("%3s %6s %10s %10s %9s" % ("k", "조합수", "일반 평균", "TTA 평균", "차이"))
for k in range(1, len(members) + 1):
    combos = list(itertools.combinations(members, k))
    a = np.mean([ens(c, P)[0] for c in combos])
    b = np.mean([ens(c, T)[0] for c in combos])
    print("%3d %6d %10.4f %10.4f %+9.4f" % (k, len(combos), a, b, b - a))

print("\nTTA 앙상블의 클래스별 F1")
mt = evaluate(ty, np.mean([T[t] for t in members], 0).argmax(1), 9)
mp = evaluate(ty, np.mean([P[t] for t in members], 0).argmax(1), 9)
print("%-11s %8s %10s %10s %9s" % ("클래스", "support", "일반", "TTA", "차이"))
for i, n in enumerate(NAMES):
    print("%-11s %8d %10.3f %10.3f %+9.3f" % (
        n, mt["support"][i], mp["per_class_f1"][i], mt["per_class_f1"][i],
        mt["per_class_f1"][i] - mp["per_class_f1"][i]))

print("\nnone -> 결함 누출")
pp = np.mean([P[t] for t in members], 0).argmax(1)
tp = np.mean([T[t] for t in members], 0).argmax(1)
tot_p = tot_t = 0
for i, n in enumerate(NAMES):
    if i == 0:
        continue
    a = int(((ty == 0) & (pp == i)).sum()); b = int(((ty == 0) & (tp == i)).sum())
    tot_p += a; tot_t += b
    print("  %-11s %6d -> %6d" % (n, a, b))
print("  %-11s %6d -> %6d" % ("합계", tot_p, tot_t))
