"""정정된 가설: 앙상블 기여는 '다른가' 가 아니라 '같은 방향으로 더 틀리는가' 가 결정한다.

내 예측이 틀렸다. e12_ssl_s0 은 기준선과 클래스 프로파일이 반대였으므로(Scratch 강, Loc 약)
기여가 양수일 것이라 했는데 -0.0103 이었다.

정정. 이 문제의 오류는 거의 전부 **none -> 결함** 한 방향이다(전체 오류의 83.5%).
확률 평균은 서로 **다른 위치**의 오류는 상쇄하지만 **같은 방향으로 더 큰** 오류는 상쇄하지 못하고
평균을 그 방향으로 끌고 간다. 그러면 기여는 "얼마나 다른가" 가 아니라
"같은 방향 오류를 남들보다 얼마나 더 내는가" 로 결정된다.

**검증 가능한 예측: leave-one-out 기여는 그 모델의 none 누출량과 음의 상관을 가진다.**
클래스 프로파일 차이(기준선과의 L1 거리)보다 none 누출이 더 잘 설명해야 한다.
아니면 이 정정도 틀린 것이다.
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate
from bench_ensemble import leave_one_out_delta

POSTHOC = "result/posthoc"
entries = json.loads(open("docs/experiments/ensemble_members.json", encoding="utf-8").read())
tags = [e["tag"] for e in entries]

probs, ty = {}, None
for t in tags:
    d = np.load(f"{POSTHOC}/{t}_logits.npz")
    ty = d["test_y"] if ty is None else ty
    z = d["test_logits"].astype(np.float64)
    ex = np.exp(z - z.max(1, keepdims=True))
    probs[t] = ex / ex.sum(1, keepdims=True)

single, leak = {}, {}
for t in tags:
    pred = probs[t].argmax(1)
    single[t] = float(evaluate(ty, pred, 9)["macro_f1"])
    leak[t] = int(((ty == 0) & (pred != 0)).sum())

scores = {}
for k in (1, 2, 3):
    for c in itertools.combinations(tags, k):
        scores[c] = float(evaluate(ty, np.mean([probs[t] for t in c], 0).argmax(1), 9)["macro_f1"])

loo = {}
for t in tags:
    d = leave_one_out_delta(scores, t)
    loo[t] = float(np.mean([v["delta"] for kk, v in d.items() if kk in (2, 3)]))

print("%-22s %9s %12s %11s" % ("구성원", "단독", "none 누출", "LOO(k=2,3)"))
for t in sorted(tags, key=lambda x: -loo[x]):
    print("%-22s %9.4f %12d %+11.4f" % (t, single[t], leak[t], loo[t]))


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


L = [leak[t] for t in tags]
S = [single[t] for t in tags]
V = [loo[t] for t in tags]
print("\n상관 (n=%d)" % len(tags))
print("  LOO 대 none 누출      r = %+.3f" % pearson(V, L))
print("  LOO 대 단독 성능      r = %+.3f" % pearson(V, S))
print("  LOO 대 log(none 누출) r = %+.3f" % pearson(V, np.log(np.maximum(L, 1))))

# 누출을 뺀 나머지가 설명하는 부분이 있는지: 누출 비슷한 것끼리 비교
print("\n누출이 비슷한 구간에서 단독 성능이 추가로 설명하는가")
order = sorted(tags, key=lambda t: leak[t])
for t in order:
    print("  %-22s 누출 %6d  단독 %.4f  LOO %+.4f" % (t, leak[t], single[t], loo[t]))
