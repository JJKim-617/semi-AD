"""E27 판정 — batch 128 대조군이 batch 256 과 다른가.

반증 조건은 실행 전에 `docs/experiments/candidate/batch_control.md` 에 박았다.
**묻는 것은 "어느 쪽이 나은가" 가 아니라 "다른가" 다.**

6 seed 확장(사전 등록 완료):
- **주 통계는 "이진 오류 총합" 하나**다. E25 채택선이 거기 묶여 있다.
  나머지 셋은 보조로 보고하되 **유의성을 주장하지 않는다**(다중 비교 방지).
- 주 통계에 **양측 치환 검정**을 걸고 **p <= 0.05 면 "다르다"**.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate  # noqa: E402
from a6_perclass_offset import cache_logits  # noqa: E402
from bench_loo_criterion import (  # noqa: E402
    null_resolution, permutation_p_two_sided,
)

POST = "result/posthoc"
CACHE = "data/wm811k/cache/wm811k_64pad.npz"
SPL = "data/wm811k/cache/splits_v1.npz"

B256 = [f"e17_translate_s{i}" for i in range(4)]
B128 = sorted((os.path.basename(p).replace("_best_test.json", "")
               for p in glob.glob("result/cls_baseline/e27_b128_s*_best_test.json")),
              key=lambda t: int(t.split("_s")[-1]))

for t in B128:
    if not os.path.exists(f"{POST}/{t}_logits.npz"):
        print(f"  [로짓] {t}", flush=True)
        cache_logits(f"result/cls_baseline/{t}_best.pt", CACHE, SPL, POST, t)

P, ty = {}, None
for t in B256 + B128:
    f = f"{POST}/{t}_logits.npz"
    if not os.path.exists(f):
        continue
    d = np.load(f)
    ty = d["test_y"] if ty is None else ty
    P[t] = d["test_logits"].argmax(1)
B128 = [t for t in B128 if t in P]
if not B128:
    raise SystemExit("E27 로짓이 없다")


def stats(t):
    return (float(evaluate(ty, P[t], 9)["macro_f1"]),
            int(((ty == 0) & (P[t] != 0)).sum()),
            int(((ty != 0) & (P[t] == 0)).sum()))


rows = {t: stats(t) for t in B256 + B128}
print("\n== 개별 seed ==")
print("  %-22s %10s %9s %9s %9s" % ("모델", "macro-F1", "오검출", "미검출", "합"))
for t in B256 + B128:
    f1, fp, fn = rows[t]
    print("  %-22s %10.4f %9d %9d %9d" % (t, f1, fp, fn, fp + fn))


def grp(ts):
    a = np.array([rows[t] for t in ts], dtype=np.float64)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 1] + a[:, 2]


f256, fp256, fn256, tot256 = grp(B256)
f128, fp128, fn128, tot128 = grp(B128)
n, g = len(B256) + len(B128), len(B128)

print("\n== 해상도 (설계 전에 확인한 것) ==")
tot_c, pmin = null_resolution(n, g)
print("  N=%d, g=%d -> C(N,g)=%d, 최소 단측 p=%.4f, **최소 양측 p=%.4f** -> alpha=0.05 %s" % (
    n, g, tot_c, pmin, 2 * pmin, "**가능**" if 2 * pmin <= 0.05 else "불가능"))

print("\n== 사전 등록한 네 통계 (기술 통계) ==")
print("  %-16s %12s %26s %12s %s" % (
    "통계", "batch 256", "그 seed 범위", "batch 128", "범위 안인가"))
inside = {}
for name, a, b in [("단독 macro-F1", f256, f128), ("오검출", fp256, fp128),
                   ("미검출", fn256, fn128), ("**이진 오류 총합**", tot256, tot128)]:
    lo, hi = a.min(), a.max()
    m = b.mean()
    ok = lo <= m <= hi
    inside[name] = ok
    fmt = "%12.4f" if "macro" in name else "%12.0f"
    rng = ("%.4f ~ %.4f" % (lo, hi)) if "macro" in name else ("%.0f ~ %.0f" % (lo, hi))
    print(("  %-16s " + fmt + " %26s " + fmt + " %s") % (
        name, a.mean(), rng, m, "**안**" if ok else "**밖**"))

print("\n== 주 통계 판정 — 이진 오류 총합, 양측 치환 검정 ==")
pooled = np.concatenate([tot256, tot128])
gi = list(range(len(tot256), len(pooled)))
p2 = permutation_p_two_sided(pooled, gi)
print("  batch 256 %.0f  대  batch 128 %.0f   (차이 %+.0f)" % (
    tot256.mean(), tot128.mean(), tot128.mean() - tot256.mean()))
print("  **양측 치환 p = %.4f**  ->  **%s**" % (
    p2, "다르다 (p<=0.05)" if p2 <= 0.05 else "구분되지 않는다 (p>0.05)"))
if 2 * pmin > 0.05:
    print("  (경고: 이 구성에서는 alpha=0.05 자체가 도달 불가능하다)")

print("\n== 보조 통계 (유의성 주장하지 않는다 — 다중 비교) ==")
for name, a, b in [("macro-F1", f256, f128), ("오검출", fp256, fp128),
                   ("미검출", fn256, fn128)]:
    pl = np.concatenate([a, b])
    print("  %-12s 양측 치환 p = %.4f  (참고용)" % (
        name, permutation_p_two_sided(pl, list(range(len(a), len(pl))))))

print("\n== E25 채택선 ==")
print("  batch 128 대조군 %d seed 의 이진 오류 총합 = %.0f" % (len(B128), tot128.mean()))
print("  **E25 채택선 = 이진 오류 총합 < %.0f** (100장 감소)" % (tot128.mean() - 100))
