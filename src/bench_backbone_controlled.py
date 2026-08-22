"""구조 다양성 판정에서 품질 교란을 뺀다.

앞선 집계에서 구조 1종이 3종보다 높게 나왔는데, 1종 그룹에 translate 모델 4개가
몰려 있다(전부 resnet18 이고 단독 0.76 대로 가장 세다). 다양성이 아니라 품질을 잰 것이다.

두 가지로 통제한다.
  A. translate 를 빼고 같은 집계를 다시 한다.
  B. 구성원 평균 단독성능이 비슷한 조합끼리만 비교한다.
"""
from __future__ import annotations
import itertools, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate

POSTHOC = "result/posthoc"
entries = json.loads(open("docs/experiments/ensemble_members.json", encoding="utf-8").read())
P, ty, ARCH = {}, None, {}
for e in entries:
    p = f"{POSTHOC}/{e['tag']}_logits.npz"
    if not os.path.exists(p): continue
    d = np.load(p); ty = d["test_y"] if ty is None else ty
    z = d["test_logits"].astype(np.float64); ex = np.exp(z - z.max(1, keepdims=True))
    P[e["tag"]] = ex / ex.sum(1, keepdims=True); ARCH[e["tag"]] = e.get("architecture","resnet18")
single = {t: float(evaluate(ty, P[t].argmax(1), 9)["macro_f1"]) for t in P}

def ens(tags):
    return float(evaluate(ty, np.mean([P[t] for t in tags], 0).argmax(1), 9)["macro_f1"])

# A. translate 와 극좌표(품질 이상치)를 뺀 풀
POOL = [t for t in P
        if not t.startswith("e17_translate") and "polar" not in t
        and (t.startswith("e18_") or t in
             ["e8_pad","e10_pad_s1","e10_pad_s2","e7_ce_long_best","e6_focal_long_best",
              "e6_focal_long_s1_best","e7_ce_long80_best","e8_res96"])]
print("통제 풀 %d개 (translate, 극좌표 제외)" % len(POOL))
print("  resnet18 %d개, 다른 백본 %d개" % (
    sum(1 for t in POOL if ARCH[t]=="resnet18"), sum(1 for t in POOL if ARCH[t]!="resnet18")))

rows = []
for c in itertools.combinations(POOL, 3):
    rows.append((len({ARCH[t] for t in c}), ens(list(c)), float(np.mean([single[t] for t in c]))))

print("\nA. 구조 가짓수별 (크기 3, translate 제외)")
g = defaultdict(list)
for n_a, f1, q in rows: g[n_a].append((f1, q))
for n_a in sorted(g):
    v = g[n_a]
    print("  구조 %d종  n=%4d  앙상블 평균 %.4f  구성원평균 %.4f  이득 %+.4f" % (
        n_a, len(v), np.mean([x[0] for x in v]), np.mean([x[1] for x in v]),
        np.mean([x[0]-x[1] for x in v])))

print("\nB. 구성원 평균 품질을 맞춘 비교 (구성원평균 0.72~0.74 구간만)")
band = [(n_a, f1, q) for n_a, f1, q in rows if 0.72 <= q <= 0.74]
g2 = defaultdict(list)
for n_a, f1, q in band: g2[n_a].append(f1)
for n_a in sorted(g2):
    print("  구조 %d종  n=%4d  앙상블 평균 %.4f" % (n_a, len(g2[n_a]), np.mean(g2[n_a])))

print("\n참고: E14 의 표현 다양성 (같은 방식)")
print("  표현 1종 0.7420 -> 3종 0.7732  (+0.0312, 단조 증가)")
