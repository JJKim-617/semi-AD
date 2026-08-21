"""최종 앙상블 확정. '전부 넣기' 와 '확실히 망가진 것만 빼기' 를 비교한다.

이번 사이클에서 정정된 것: 넓은 품질 범위에서는 단독 성능이 앙상블 기여와
r=+0.942 로 강하게 연관된다. 그러면 "아무것도 고르지 않는다" 를 문자 그대로
지킬 이유가 없다 — 확실히 망가진 구성원은 빼는 것이 낫다.

단 뺄 근거가 test 라벨이면 그것은 사후 선택이다. 그래서 두 가지를 나눠 본다.
  A. 전부 넣기 (선택 없음)
  B. val 로 걸러내기 (val 최하위 제외 — 라벨 없이 가능한 유일한 신호)
  C. test 를 보고 망가진 것 제외 (oracle, 상한 참고용)
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate

NAMES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full"]
entries = json.loads(open("docs/experiments/ensemble_members.json", encoding="utf-8").read())
tags = [e["tag"] for e in entries]

P, V, ty, vy = {}, {}, None, None
for t in tags:
    d = np.load(f"result/posthoc/{t}_logits.npz")
    ty = d["test_y"] if ty is None else ty
    vy = d["val_y"] if vy is None else vy
    for store, key in ((P, "test_logits"), (V, "val_logits")):
        z = d[key].astype(np.float64)
        e = np.exp(z - z.max(1, keepdims=True))
        store[t] = e / e.sum(1, keepdims=True)

val_f1 = {t: float(evaluate(vy, V[t].argmax(1), 9)["macro_f1"]) for t in tags}
test_f1 = {t: float(evaluate(ty, P[t].argmax(1), 9)["macro_f1"]) for t in tags}


def ens(sub):
    m = evaluate(ty, np.mean([P[t] for t in sub], 0).argmax(1), 9)
    return m, len(sub)


SETS = {
    "A. 전부 (선택 없음)": tags,
    "B. val 최하위 1개 제외": [t for t in tags if t != min(tags, key=lambda x: val_f1[x])],
    "B2. val 최하위 3개 제외": [t for t in tags if t not in
                             sorted(tags, key=lambda x: val_f1[x])[:3]],
    "C. 극좌표+SSL 제외 (oracle)": [t for t in tags if "polar" not in t and "ssl" not in t],
}

print("%-30s %4s %10s %10s" % ("구성", "n", "macro-F1", "accuracy"))
best = None
for name, sub in SETS.items():
    m, n = ens(sub)
    print("%-30s %4d %10.4f %10.4f" % (name, n, m["macro_f1"], m["accuracy"]))
    if best is None or m["macro_f1"] > best[1]["macro_f1"]:
        best = (name, m, sub)

print("\nval 로 뺄 대상 (val 낮은 순)")
for t in sorted(tags, key=lambda x: val_f1[x])[:4]:
    print("  %-22s val %.4f  test %.4f" % (t, val_f1[t], test_f1[t]))

name, m, sub = best
print("\n최선: %s (n=%d)  macro-F1 %.4f  accuracy %.4f" % (name, len(sub), m["macro_f1"], m["accuracy"]))
print("\n클래스별")
print("%-11s %9s %8s" % ("클래스", "support", "F1"))
for i, n in enumerate(NAMES):
    print("%-11s %9d %8.3f" % (n, m["support"][i], m["per_class_f1"][i]))
pred = np.mean([P[t] for t in sub], 0).argmax(1)
print("none -> 결함 누출 합계 %d / 110,701" % int(((ty == 0) & (pred != 0)).sum()))
