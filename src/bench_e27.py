"""E27 판정 — batch 128 대조군이 batch 256 과 다른가.

반증 조건은 실행 전에 `docs/experiments/candidate/batch_control.md` 에 박았다.
**묻는 것은 "어느 쪽이 나은가" 가 아니라 "다른가" 다.**
"""
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate  # noqa: E402
from a6_perclass_offset import cache_logits  # noqa: E402
from bench_loo_criterion import null_resolution, permutation_p  # noqa: E402

POST = "result/posthoc"
CACHE = "data/wm811k/cache/wm811k_64pad.npz"
SPL = "data/wm811k/cache/splits_v1.npz"

B256 = [f"e17_translate_s{i}" for i in range(4)]
B128 = [f"e27_b128_s{i}" for i in range(3)]

for t in B128:
    if not os.path.exists(f"result/cls_baseline/{t}_best_test.json"):
        print(f"  [아직] {t}")
        continue
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


print("\n== 개별 seed ==")
print("  %-22s %10s %9s %9s %9s" % ("모델", "macro-F1", "오검출", "미검출", "합"))
rows = {}
for t in B256 + B128:
    f1, fp, fn = stats(t)
    rows[t] = (f1, fp, fn)
    print("  %-22s %10.4f %9d %9d %9d" % (t, f1, fp, fn, fp + fn))


def grp(ts):
    a = np.array([rows[t] for t in ts], dtype=np.float64)
    tot = a[:, 1] + a[:, 2]
    return a[:, 0], a[:, 1], a[:, 2], tot


f256, fp256, fn256, tot256 = grp(B256)
f128, fp128, fn128, tot128 = grp(B128)

print("\n== 사전 등록한 네 통계 ==")
print("  %-16s %12s %26s %12s %s" % (
    "통계", "batch 256", "그 seed 범위", "batch 128", "범위 안인가"))
verdicts = []
for name, a, b in [("단독 macro-F1", f256, f128), ("오검출", fp256, fp128),
                   ("미검출", fn256, fn128), ("**이진 오류 총합**", tot256, tot128)]:
    lo, hi = a.min(), a.max()
    m = b.mean()
    inside = lo <= m <= hi
    verdicts.append(inside)
    fmt = "%12.4f" if "macro" in name else "%12.0f"
    print(("  %-16s " + fmt + " %26s " + fmt + " %s") % (
        name, a.mean(),
        ("%.4f ~ %.4f" % (lo, hi)) if "macro" in name else ("%.0f ~ %.0f" % (lo, hi)),
        m, "**안**" if inside else "**밖**"))

print("\n== 판정 (사전 등록한 규칙) ==")
if all(verdicts):
    print("  네 통계 모두 batch 256 의 seed 범위 안이다.")
    print("  -> **구분되지 않는다.** E25 를 기존 대조군과 비교해도 되고,")
    print("     이진 오류 총합 기준 2,626 이 그대로 선다.")
else:
    out = [n for n, v in zip(["macro-F1", "오검출", "미검출", "이진 총합"], verdicts) if not v]
    print("  범위 밖인 통계: %s" % ", ".join(out))
    print("  -> **다르다.** 그 차이만큼이 E25 판정에서 걷어낼 몫이다.")

print("\n== 이진 오류 총합 기준점을 옳은 대조군에 다시 묶는다 ==")
print("  batch 256 (e17_translate 4 seed): %.0f" % tot256.mean())
print("  **batch 128 (e27 %d seed): %.0f**  <- E25 는 이것과 견줘야 한다" % (
    len(B128), tot128.mean()))
print("  사전 등록한 E25 채택선은 '대조군 대비 100장 이상 감소' 다.")
print("  -> **E25 의 새 채택선: 이진 오류 총합 < %.0f**" % (tot128.mean() - 100))

print("\n== 해상도 (실행 전에 밝힌 한계) ==")
n, g = len(B256) + len(B128), len(B128)
tot_c, pmin = null_resolution(n, g)
print("  N=%d, g=%d -> C(N,g)=%d, 최소 단측 p=%.4f (양측 %.4f)" % (
    n, g, tot_c, pmin, 2 * pmin))
print("  **양측 alpha=0.05 를 만들 수 없다. 유의성을 주장하지 않는다.**")
pooled = np.concatenate([f256, f128])
gi = list(range(len(f256), len(pooled)))
print("  참고: macro-F1 에 대한 단측 치환 p = %.3f" % permutation_p(pooled, gi))
