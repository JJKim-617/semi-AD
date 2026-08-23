"""과거 판정 풀들에서 LOO 판정선을 잡음 구조로 다시 유도한다. 학습 없음.

**순서를 지킨다: 먼저 귀무분포를 재고, 그 다음에 선을 정한다.**
"과거 기각을 통과시키는 값" 을 역산하지 않는다.
"""
import glob
import math
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from bench_loo_criterion import (  # noqa: E402
    leave_one_out_values, null_resolution, null_sd_of_group_mean,
    permutation_p, required_threshold,
)

POST = "result/posthoc"


def load(tags):
    P, ty = {}, None
    for t in tags:
        f = f"{POST}/{t}_logits.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f)
        ty = d["test_y"] if ty is None else ty
        z = d["test_logits"].astype(np.float32)
        e = np.exp(z - z.max(1, keepdims=True))
        P[t] = e / e.sum(1, keepdims=True)
    return P, ty


TR = [f"e17_translate_s{i}" for i in range(4)]
DENS = [f"e20_dens_s{i}" for i in range(3)]
LINE = [f"e21_line_s{i}" for i in range(3)]
SHUF = ["e20_shuf_s0"]
E22 = sorted(os.path.basename(p).replace("_logits.npz", "")
             for p in glob.glob(f"{POST}/e22_*_logits.npz") if "_tta_" not in p)
E24A = [f"e24_pad_s{i}" for i in range(3)]
E24B = [f"e24_padtr_s{i}" for i in range(3)]
CTRL2 = ["e8_pad", "e10_pad_s2"]

CASES = [
    ("E20 밀도", TR + DENS + SHUF + LINE, DENS),
    ("E21 선 필터", TR + DENS + SHUF + LINE, LINE),
    ("E22 (a) 9개", TR + E22, E22),
    ("E22 (b) s0", TR + [t for t in E22 if t.endswith("_s0")],
     [t for t in E22 if t.endswith("_s0")]),
    ("E22 (b) s1", TR + [t for t in E22 if t.endswith("_s1")],
     [t for t in E22 if t.endswith("_s1")]),
    ("E22 (b) s2", TR + [t for t in E22 if t.endswith("_s2")],
     [t for t in E22 if t.endswith("_s2")]),
    ("E24 팔A", CTRL2 + E24A, E24A),
    ("E24 팔B", TR + E24B, E24B),
]

print("=== 1. 검정 해상도 — 그 풀에서 유의가 **가능하기는** 한가 ===")
print("  %-14s %3s %3s %9s %10s %s" % ("판정", "N", "g", "C(N,g)", "최소 p", "alpha=0.05 가능?"))
for name, pool, group in CASES:
    n, g = len(pool), len(group)
    tot, pmin = null_resolution(n, g)
    print("  %-14s %3d %3d %9d %10.4f %s" % (
        name, n, g, tot, pmin, "가능" if pmin <= 0.05 else "**불가능**"))

print()
print("=== 2. 귀무분포를 재고 선을 유도한다 ===")
print("  %-14s %3s %3s %10s %9s %10s %8s" % (
    "판정", "N", "g", "관측 군평균", "귀무 sd", "새 선(p<=.05)", "p"))
rows = []
for name, pool, group in CASES:
    P, ty = load(pool)
    pool = [t for t in pool if t in P]
    group = [t for t in group if t in P]
    if len(group) < 2 or len(pool) <= len(group):
        continue
    loo = leave_one_out_values(P, ty, pool)
    vals = np.array([loo[t] for t in pool])
    gidx = [pool.index(t) for t in group]
    obs = float(vals[gidx].mean())
    sd = null_sd_of_group_mean(vals, len(group))
    thr = required_threshold(vals, len(group), alpha=0.05)
    p = permutation_p(vals, gidx)
    rows.append((name, len(pool), len(group), obs, sd, thr, p))
    ts = "도달불가" if math.isinf(thr) else "%.4f" % thr
    print("  %-14s %3d %3d %10.4f %9.4f %10s %8.3f" % (
        name, len(pool), len(group), obs, sd, ts, p))

print()
print("=== 3. 옛 선 +0.003 은 귀무분포에서 어디였나 ===")
print("  %-14s %10s %12s %s" % ("판정", "옛선/귀무sd", "그때의 p", "옛 선 대 새 선"))
for name, n, g, obs, sd, thr, p in rows:
    z = 0.003 / sd if sd > 0 else float("inf")
    pv = 1 - 0.5 * (1 + math.erf(z / (2 ** 0.5)))
    ratio = "도달불가" if math.isinf(thr) else "%.1f배" % (0.003 / thr)
    print("  %-14s %9.2f배 %12.5f %s" % (name, z, pv, ratio))

print()
print("=== 4. 새 선으로 다시 보면 (참고자료 — 판정을 바꾸지 않는다) ===")
print("  %-14s %10s %10s %8s %s" % ("판정", "관측", "새 선", "p", "새 선 기준"))
for name, n, g, obs, sd, thr, p in rows:
    if math.isinf(thr):
        verd = "**판정 불가** (해상도 없음)"
    else:
        verd = "통과" if p <= 0.05 else ("경계 (p<0.10)" if p <= 0.10 else "미달")
    ts = "도달불가" if math.isinf(thr) else "%.4f" % thr
    print("  %-14s %10.4f %10s %8.3f  %s" % (name, obs, ts, p, verd))

print()
print("=== 5. 지금 자로 무엇을 검출할 수 있나 (검정력) ===")
P, ty = load(TR + E22)
pool = TR + E22
loo = leave_one_out_values(P, ty, pool)
sig13 = float(np.std([loo[t] for t in pool], ddof=1))
P2, ty2 = load(TR + DENS + SHUF + LINE)
pool2 = [t for t in TR + DENS + SHUF + LINE if t in P2]
loo2 = leave_one_out_values(P2, ty2, pool2)
sig11 = float(np.std([loo2[t] for t in pool2], ddof=1))
print("  구성원 LOO 의 sd: 13개 풀 %.4f, 11개 풀 %.4f" % (sig13, sig11))
print("  귀무 sd = sigma/sqrt(g) * sqrt((N-g)/(N-1))  **해석식으로 계산한다**")
print()
print("  %-28s %8s %10s %12s" % ("풀 구성 (대조 4 + 처치 g)", "귀무 sd", "필요(50%)", "필요(80%검정력)"))
for sig, lab in [(sig13, "sigma=%.4f" % sig13)]:
    for g in (3, 6, 9, 12, 20, 40):
        n = 4 + g
        sd = sig / math.sqrt(g) * math.sqrt((n - g) / (n - 1))
        print("  %-28s %8.4f %10.4f %12.4f" % (
            "N=%d, g=%d" % (n, g), sd, 1.645 * sd, (1.645 + 0.84) * sd))
print()
print("  **대조군을 함께 늘리면** (대조 g 개 + 처치 g 개, N=2g)")
print("  %-28s %8s %10s %12s" % ("풀 구성", "귀무 sd", "필요(50%)", "필요(80%검정력)"))
for g in (3, 6, 9, 12, 20):
    n = 2 * g
    sd = sig13 / math.sqrt(g) * math.sqrt((n - g) / (n - 1))
    print("  %-28s %8.4f %10.4f %12.4f" % (
        "N=%d, g=%d" % (n, g), sd, 1.645 * sd, (1.645 + 0.84) * sd))
print()
print("  이 프로젝트에서 관측된 가장 큰 군 평균 LOO 는 +0.0021 (E22 b s1) 이다.")
print("  가장 큰 '진짜 있을 법한' 효과가 그 수준이라면, 위 표에서")
print("  그것을 80%% 로 검출하려면 필요한 g 를 읽으면 된다.")
