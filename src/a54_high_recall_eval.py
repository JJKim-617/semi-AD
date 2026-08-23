"""고재현율 arm 계열 — 선언된 목적함수(`FPR@95TPR`) 아래의 형식 판정.

반증 조건과 개정은 `candidate/ood_high_recall_arm.md` §3, §5, §9 에 **실행 전에** 박았다.

## 이 스크립트가 하는 것과 안 하는 것

**저장된 점수를 다시 채점한다. 새 arm 을 만들지 않는다.**
H1(fuse4 3 seed), H4(fuse5 앙상블), H2(Tippett), H3(fuse3)는 21, 22, 15차에서 이미 계산됐다.

**§9.6 대로 증거력을 낮춰 읽는다** — 점추정은 이미 본 숫자이고,
**새로 나오는 것은 짝지은 CI, 95% 문턱 동점 수, 형식 판정 셋뿐이다.**

## 봉인

**§9.4 가 조건 6 을 정지시켰다. 통과가 나와도 열지 않는다.**
이 스크립트는 `test_sealed` 를 부르지 않는다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import (  # noqa: E402
    aupr_blocked, auroc, fpr_at_tpr, operating_points, paired_fpr_at_tpr_diff_ci,
)
from a23_ood_template_eval import (  # noqa: E402
    BIN_LBL, BINS, NAMES, _fast_auroc, bootstrap_auroc_ci, paired_aupr_blocked_diff_ci,
)

OUT = Path("result/ood/o2_high_recall")
ENS = Path("result/ood/o2_ensemble/ensemble_scores.npz")
TAIL = Path("result/ood/o1_tail/tail_scores.npz")
TARGET = 0.95
N_BOOT = 300
AUPR_DROP_LIMIT = 0.02        # §3. 실행 전에 정한 문턱.
SEED_RANGE_21 = 0.1835 - 0.1645   # §9.3. 21차 fuse4 의 FPR@95TPR seed 폭 = 0.0190
TIE_LIMIT = 1000              # §5-3
T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


def per_class(score, yy, ci_for=("Scratch", "Center", "Loc")):
    ns = score[yy == 0]
    z = np.zeros(len(ns), np.int64)
    r = {}
    for c in range(1, 9):
        m = yy == c
        if not m.sum():
            continue
        s = np.concatenate([ns, score[m]])
        l = np.concatenate([z, np.ones(int(m.sum()), np.int64)])
        row = {"n": int(m.sum()), "auroc": _fast_auroc(s, l)}
        if NAMES[c] in ci_for:
            row["auroc_ci"] = bootstrap_auroc_ci(s, l, n_boot=200, seed=c)
        r[NAMES[c]] = row
    return r


def per_size(score, isd, size):
    o = {}
    for i, lbl in enumerate(BIN_LBL):
        m = (size >= BINS[i]) & (size < BINS[i + 1])
        if m.sum() and isd[m].sum() not in (0, m.sum()):
            o[lbl] = aupr_blocked(score[m], isd[m])
    return o


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ez = np.load(ENS, allow_pickle=True)
    tz = np.load(TAIL, allow_pickle=True)
    y_dev, size_dev = ez["y_dev"], ez["size_dev"]
    assert np.array_equal(y_dev, tz["y_dev"]), "두 파일이 같은 파티션이어야 한다"
    isd = (y_dev != 0).astype(np.int64)

    arms = {
        "H3 fuse3 (대조)": ez["fuse3_(대조)"],
        "H1 fuse4 s0": ez["fuse4_s0_(참고)"],
        "H1 fuse4 s1": ez["fuse4_s1_(참고)"],
        "H1 fuse4 s2": ez["fuse4_s2_(참고)"],
        "H4 fuse5 앙상블": ez["fuse5_앙상블_(주)"],
        "H2 Tippett 최대": tz["5 Tippett 최대 (이접)"],
    }
    CTRL = "H3 fuse3 (대조)"
    log("test_dev %d (결함 %d) — 저장된 점수 %d개" % (len(y_dev), int(isd.sum()), len(arms)))

    rep = {"target": TARGET, "n_boot": N_BOOT, "aupr_drop_limit": AUPR_DROP_LIMIT,
           "seed_range_21": SEED_RANGE_21, "tie_limit": TIE_LIMIT, "metrics": {}}
    for k, s in arms.items():
        ops = operating_points(s, isd, (0.5, 0.8, TARGET))
        rep["metrics"][k] = {
            "fpr_at_95tpr": fpr_at_tpr(s, isd, TARGET),
            "aupr_blocked": aupr_blocked(s, isd), "auroc": auroc(s, isd),
            "n_unique": int(len(np.unique(s))),
            "n_tied_at_95": ops[2]["n_tied_at_threshold"],
            "fp_at_95": ops[2]["false_positives"],
            "precision_at_95": ops[2]["precision"],
            "operating": ops, "per_class": per_class(s, y_dev),
            "per_size_bin": per_size(s, isd, size_dev)}

    print("\n== 선언된 주 지표: FPR@95TPR (낮을수록 좋다) ==")
    print("%-18s %11s %11s %10s %12s %10s"
          % ("arm", "FPR@95TPR", "AUPR블록", "고유값", "95% 동점", "95% 헛경보"))
    for k, m in rep["metrics"].items():
        print("%-18s %11.4f %11.4f %10d %12d %10d"
              % (k, m["fpr_at_95tpr"], m["aupr_blocked"], m["n_unique"],
                 m["n_tied_at_95"], m["fp_at_95"]))

    base = rep["metrics"][CTRL]
    print("\n== 반증 조건 1 / 9.3: 짝지은 FPR@95TPR 차이 (상한 < 0 이어야 이긴 것) ==")
    rep["paired_fpr"] = {}
    for k in arms:
        if k == CTRL:
            continue
        lo, hi, p = paired_fpr_at_tpr_diff_ci(arms[k], arms[CTRL], isd, TARGET, N_BOOT, 91)
        d = rep["metrics"][k]["fpr_at_95tpr"] - base["fpr_at_95tpr"]
        rep["paired_fpr"][k] = {"diff": float(d), "lo": lo, "hi": hi, "p": p}
        print("  %-18s 차이 %+.4f  CI [%+.4f, %+.4f] p=%.4f  %s"
              % (k, d, lo, hi, p,
                 "이김" if hi < 0 else ("짐" if lo > 0 else "무승부")))

    print("\n== 반증 조건 2: blocked AUPR 하락이 %.2f 를 넘는가 ==" % AUPR_DROP_LIMIT)
    rep["aupr_drop"] = {}
    for k in arms:
        if k == CTRL:
            continue
        drop = base["aupr_blocked"] - rep["metrics"][k]["aupr_blocked"]
        rep["aupr_drop"][k] = float(drop)
        print("  %-18s 하락 %+.4f  %s" % (k, drop, "**한계 초과 — 기각**" if drop > AUPR_DROP_LIMIT
                                        else "통과"))

    print("\n== 반증 조건 3: 95%% 재현율 문턱 동점 (한계 %d장) ==" % TIE_LIMIT)
    for k, m in rep["metrics"].items():
        print("  %-18s %8d  %s" % (k, m["n_tied_at_95"],
                                   "**한계 초과 — 채택 보류**" if m["n_tied_at_95"] > TIE_LIMIT
                                   else "통과"))

    print("\n== 반증 조건 4: Scratch / Center / Loc (CI 가 안 겹치게 내려가면 기각) ==")
    rep["class_check"] = {}
    for k, m in rep["metrics"].items():
        cells, bad = [], []
        for c in ("Center", "Scratch", "Loc"):
            a = m["per_class"][c]
            b = base["per_class"][c]
            cells.append("%s %.4f [%.4f,%.4f]" % (c, a["auroc"], *a["auroc_ci"]))
            if a["auroc_ci"][1] < b["auroc_ci"][0]:
                bad.append(c)
        rep["class_check"][k] = bad
        print("  %-18s %s%s" % (k, "  ".join(cells),
                                "  ** %s 하락 — 기각**" % ",".join(bad) if bad else ""))

    print("\n== 반증 조건 5 / 9.3: 이득 대 21차 seed 폭 %.4f ==" % SEED_RANGE_21)
    h1 = [rep["metrics"]["H1 fuse4 s%d" % s]["fpr_at_95tpr"] for s in (0, 1, 2)]
    rep["h1_worst"] = float(max(h1))
    rep["h1_range"] = float(max(h1) - min(h1))
    print("  H1 세 seed: %s  최악 %.4f (대조 %.4f)  폭 %.4f"
          % (["%.4f" % v for v in h1], max(h1), base["fpr_at_95tpr"], rep["h1_range"]))
    print("  H1 조건 1(최악값도 대조보다 낮은가): %s"
          % ("통과" if max(h1) < base["fpr_at_95tpr"] else "**기각**"))
    for k in ("H4 fuse5 앙상블",):
        gain = abs(rep["paired_fpr"][k]["diff"])
        print("  %-18s 이득 %.4f  대  seed 폭 %.4f  ->  %s"
              % (k, gain, SEED_RANGE_21,
                 "잡음을 넘는다" if gain > SEED_RANGE_21 else "**잡음 안 — 무승부로 적는다**"))
        rep["h4_gain_vs_range"] = {"gain": float(gain), "range": SEED_RANGE_21,
                                   "exceeds": bool(gain > SEED_RANGE_21)}

    print("\n== 병기: 운영 지점 ==")
    for k, m in rep["metrics"].items():
        c = ["%0.3f/%6d" % (r["precision"], r["false_positives"]) for r in m["operating"]]
        print("  %-18s 50%% %s  80%% %s  95%% %s" % (k, c[0], c[1], c[2]))

    print("\n== 병기: dieSize 구간별 blocked AUPR ==")
    print("%-18s " % "arm" + " ".join("%11s" % b for b in BIN_LBL))
    for k, m in rep["metrics"].items():
        print("%-18s " % k + " ".join(
            "%11.4f" % m["per_size_bin"][b] if b in m["per_size_bin"] else "%11s" % "-"
            for b in BIN_LBL))

    print("\n== 병기: 주 지표(blocked AUPR) 짝지은 차이 — 채택 표는 이걸로 정한다 ==")
    rep["paired_aupr"] = {}
    for k in arms:
        if k == CTRL:
            continue
        lo, hi, p = paired_aupr_blocked_diff_ci(arms[k], arms[CTRL], isd, N_BOOT, 92)
        rep["paired_aupr"][k] = {"lo": lo, "hi": hi, "p": p}
        print("  %-18s [%+.4f, %+.4f] p=%.4f  %s"
              % (k, lo, hi, p, "이김" if lo > 0 else ("짐" if hi < 0 else "무승부")))

    (OUT / "high_recall_metrics.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log("저장 완료 → %s  (봉인은 열지 않았다 — §9.4)" % OUT)
