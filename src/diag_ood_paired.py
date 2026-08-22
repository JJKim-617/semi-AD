"""짝지은 AUPR 차이로 O1 arm 들을 다시 비교한다 — 반증 조건 6 의 검정 정정.

원래 박아 둔 기준은 "독립 CI 가 겹치지 않아야 이겼다" 였는데, 두 점수가 **같은
118,595장** 위에서 계산되므로 그건 짝지은 비교에 틀린 검정이다. 여기서 올바른 검정을 낸다.

같이 보는 것:
- **template 이 균일 대조군보다 낫는가** (T-* S_nll vs U-NLL). template 이 한 일이 있는가.
- **뒤집은 점수의 값어치** — S_llr 의 Edge-Ring AUROC 가 0.02 다. 이상 점수로는 최악이지만
  뒤집으면 0.98 이라, 하류 8종 분류에는 강한 신호일 수 있다.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import aupr, auroc  # noqa: E402
from a23_ood_template_eval import NAMES, paired_aupr_diff_ci  # noqa: E402

d = np.load("result/ood/o1_template/o1_template_scores.npz", allow_pickle=True)
yte = d["y_test"].astype(np.int64)
is_def = (yte != 0).astype(np.int64)
S = {k: d[k].astype(np.float64) for k in d.files if k not in
     ("y_test", "size_test", "template_pad", "template_resize", "template_radial")}

RATIO = "E0 불량 다이 비율(대조)"
UNIF = "U-NLL 균일 template(대조)"

PAIRS = [
    ("FUSE 비율+T-PAD S_llr", RATIO, "융합이 자명한 기준선을 이기는가 (반증 조건 6)"),
    ("FUSE 비율+T-PAD S_llr", "T-PAD S_llr", "융합이 나머지 구성 요소를 이기는가"),
    ("T-RESIZE S_nll", UNIF, "리사이즈 template 이 균일 대조군보다 나은가"),
    ("T-PAD S_nll", UNIF, "pad template 이 균일 대조군보다 나은가"),
    ("T-RADIAL S_nll", UNIF, "반경 template 이 균일 대조군보다 나은가"),
    ("T-RESIZE S_nll", RATIO, "가장 좋은 단독 template arm 이 바닥을 넘는가"),
]

print("== 짝지은 AUPR 차이 (같은 재표본, 1000회) ==")
print("%-46s %9s %9s %22s %9s" % ("비교", "AUPR a", "AUPR b", "차이 95% CI", "p"))
for a, b, why in PAIRS:
    if a not in S or b not in S:
        print("  건너뜀 (없는 arm): %s vs %s" % (a, b))
        continue
    lo, hi, p = paired_aupr_diff_ci(S[a], S[b], is_def, n_boot=1000, seed=11)
    print("%-46s %9.4f %9.4f  [%+.4f, %+.4f] %9.4f"
          % (a + " vs " + b, aupr(S[a], is_def), aupr(S[b], is_def), lo, hi, p))
    print("    → %s : %s" % (why, "이김" if lo > 0 else ("짐" if hi < 0 else "구분 안 됨")))

print("\n== 뒤집으면 강한 신호인가 (클래스별 AUROC, none 대비) ==")
print("이상 점수로 0.5 보다 한참 낮다는 것은 정보가 없다는 뜻이 아니라 부호가 반대라는 뜻이다.")
for name in ("T-RESIZE S_llr", "T-RADIAL S_llr", "T-PAD S_llr"):
    s = S[name]
    ns = s[yte == 0]
    z = np.zeros(len(ns), np.int64)
    parts = []
    for c in range(1, 9):
        m = yte == c
        a = auroc(np.concatenate([ns, s[m]]),
                  np.concatenate([z, np.ones(int(m.sum()), np.int64)]))
        parts.append("%s %.3f(뒤집으면 %.3f)" % (NAMES[c], a, 1 - a))
    print("\n%s" % name)
    for p_ in parts:
        print("   ", p_)

print("\n== 두 결함 클래스를 서로 가르는가 (Edge-Ring vs Scratch, 결함끼리) ==")
print("탐지가 아니라 하류 8종 분류의 관점이다. 정상은 빼고 결함끼리만 본다.")
er, sc = NAMES.index("Edge-Ring"), NAMES.index("Scratch")
for name in ("T-RESIZE S_llr", "T-RADIAL S_llr", RATIO, "E0 불량 다이 개수(대조)"):
    s = S[name]
    m = (yte == er) | (yte == sc)
    lab = (yte[m] == sc).astype(np.int64)
    print("  %-28s Edge-Ring vs Scratch AUROC %.4f" % (name, auroc(s[m], lab)))
