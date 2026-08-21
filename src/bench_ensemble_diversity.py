"""E14 — 앙상블 이득이 seed 다양성에서 오는가 표현 다양성에서 오는가. 학습 없음."""
from __future__ import annotations

import itertools
import sys

import numpy as np

sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate

NC = 9
P = "result/posthoc/%s_logits.npz"
MEMBERS = {
    "e8_pad":            (P % "e8_pad",            "pad64"),
    "e10_pad_s1":        (P % "e10_pad_s1",        "pad64"),
    "e10_pad_s2":        (P % "e10_pad_s2",        "pad64"),
    "e7_ce_long":        (P % "e7_ce_long_best",   "resize64"),
    "e6_focal_long":     (P % "e6_focal_long_best", "resize64"),
    "e6_focal_long_s1":  (P % "e6_focal_long_s1_best", "resize64"),
    "e7_ce_long80":      (P % "e7_ce_long80_best", "resize64"),
    "e8_res96":          (P % "e8_res96",          "resize96"),
}

probs, singles, ty = {}, {}, None
for tag, (path, _) in MEMBERS.items():
    d = np.load(path)
    if ty is None:
        ty = d["test_y"]
    else:
        assert np.array_equal(ty, d["test_y"]), f"{tag} 의 test 정렬이 다르다"
    z = d["test_logits"].astype(np.float64)
    e = np.exp(z - z.max(1, keepdims=True))
    probs[tag] = e / e.sum(1, keepdims=True)
    singles[tag] = float(evaluate(ty, probs[tag].argmax(1), NC)["macro_f1"])

print("단독 성능")
for tag, (_, rep) in MEMBERS.items():
    print("  %-18s %-9s %.4f" % (tag, rep, singles[tag]))


def ens(tags):
    pred = np.mean([probs[t] for t in tags], 0).argmax(1)
    m = evaluate(ty, pred, NC)
    mean_member = float(np.mean([singles[t] for t in tags]))
    return m["macro_f1"], m["accuracy"], mean_member, m["macro_f1"] - mean_member


COMBOS = [
    ("same-3-pad (대조군)",     ["e8_pad", "e10_pad_s1", "e10_pad_s2"]),
    ("same-3-resize (대조군)",  ["e6_focal_long", "e6_focal_long_s1", "e7_ce_long"]),
    ("diverse-3 (최고끼리)",    ["e8_pad", "e7_ce_long", "e8_res96"]),
    ("diverse-3 (무작위 seed)", ["e10_pad_s1", "e6_focal_long", "e8_res96"]),
    ("diverse-3 (다른 조합)",   ["e10_pad_s2", "e6_focal_long_s1", "e8_res96"]),
    ("all-8",                  list(MEMBERS)),
]
print()
print("%-24s %9s %9s %11s %9s" % ("앙상블", "macro-F1", "acc", "구성원평균", "이득"))
for name, tags in COMBOS:
    f1, acc, mm, gain = ens(tags)
    print("%-24s %9.4f %9.4f %11.4f %+9.4f" % (name, f1, acc, mm, gain))

print()
print("모든 3개 조합을 표현 구성별로 집계 (사후 선택 배제)")
by_kind = {}
for tags in itertools.combinations(MEMBERS, 3):
    reps = {MEMBERS[t][1] for t in tags}
    kind = "표현 %d종" % len(reps)
    f1, _, mm, gain = ens(list(tags))
    by_kind.setdefault(kind, []).append((f1, gain))
for kind in sorted(by_kind):
    v = by_kind[kind]
    f1s = [a for a, _ in v]; gains = [b for _, b in v]
    print("  %-8s n=%3d   macro-F1 평균 %.4f (최고 %.4f)   구성원평균 대비 이득 %+.4f" % (
        kind, len(v), np.mean(f1s), max(f1s), np.mean(gains)))
