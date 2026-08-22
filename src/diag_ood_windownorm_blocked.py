"""창 분모 축의 판정을 **주 지표(blocked AUPR)** 로 다시 낸다.

`a31_window_norm_eval.py` 의 짝지은 검정이 순서 의존 AUPR 을 썼다.
후보 문서에 "이 문서의 모든 판정은 blocked AUPR 로 한다" 고 적어 놓고
검정만 다른 관례를 쓴 것이라, 판정에 쓰기 전에 고친다.
동점이 적은 arm 은 결과가 같을 것이고, `k2` 계열(고유값 25개)은 달라질 수 있다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import aupr, aupr_blocked  # noqa: E402
from a23_ood_template_eval import (  # noqa: E402
    paired_aupr_blocked_diff_ci,
    paired_aupr_diff_ci,
)

OUT = Path("result/ood/o1_windownorm")
d = np.load(OUT / "window_norm_scores.npz", allow_pickle=True)
y = d["y_test"].astype(np.int64)
is_def = (y != 0).astype(np.int64)
S = {k: d[k].astype(np.float64) for k in d.files if k not in ("y_test", "size_test")}

CHAMP = "챔피언 반경보정 die k=7"
PRIMARY = "주 k2 k=5"
COMBO = "합성 k2 k=5 + 반경보정"

print("== 조건 1: 같은 k 에서 k2 대 die (blocked, 짝지은 800회) ==")
wins = 0
out = {}
for k in (3, 5, 7, 9):
    a = PRIMARY if k == 5 else "k2 k=%d" % k
    b = "die k=%d" % k
    lo, hi, p = paired_aupr_blocked_diff_ci(S[a], S[b], is_def, n_boot=400, seed=k)
    won = lo > 0 and p < 0.01
    wins += int(won)
    out["k=%d" % k] = {"ci": [lo, hi], "p": p, "won": bool(won)}
    print("  k=%d  k2 %.4f vs die %.4f   [%+.4f, %+.4f] p=%.4f  %s"
          % (k, aupr_blocked(S[a], is_def), aupr_blocked(S[b], is_def), lo, hi, p,
             "이김" if won else ("짐" if hi < 0 else "무승부")))
print("  → k2 가 %d/4 에서 이겼다. 과반(3)이 필요하므로 **분모 축 일반 주장은 기각**." % wins)

print("\n== 조건 2, 7: 챔피언과 합성 (blocked) ==")
for a, b in ((PRIMARY, CHAMP), (COMBO, PRIMARY), (COMBO, CHAMP)):
    lo, hi, p = paired_aupr_blocked_diff_ci(S[a], S[b], is_def, n_boot=400, seed=21)
    lo_o, hi_o, p_o = paired_aupr_diff_ci(S[a], S[b], is_def, n_boot=800, seed=21)
    out[a + " vs " + b] = {"blocked_ci": [lo, hi], "blocked_p": p,
                           "order_ci": [lo_o, hi_o], "order_p": p_o}
    print("  %-34s blocked [%+.4f, %+.4f] p=%.4f  %s"
          % (a + " vs " + b, lo, hi, p,
             "이김" if lo > 0 else ("짐" if hi < 0 else "무승부")))
    print("  %-34s 순서의존 [%+.4f, %+.4f] p=%.4f  (참고)" % ("", lo_o, hi_o, p_o))

print("\n== 두 관례가 얼마나 다른가 ==")
print("%-24s %10s %10s %8s %8s" % ("arm", "AUPR블록", "AUPR순서", "차이", "고유값"))
for name in (COMBO, PRIMARY, CHAMP, "die k=7", "k2 k=3", "die k=3"):
    b_, o_ = aupr_blocked(S[name], is_def), aupr(S[name], is_def)
    print("%-24s %10.4f %10.4f %+8.4f %8d"
          % (name, b_, o_, o_ - b_, len(np.unique(S[name]))))

(OUT / "paired_blocked.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print("\n저장 → %s" % (OUT / "paired_blocked.json"))
