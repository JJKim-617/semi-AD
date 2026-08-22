"""Scratch 에 맞는 창은 정사각이 아니라 선인가 (학습 없음).

E20 진단이 남긴 것: **Scratch 의 국소 밀도 중앙값 0.500 은 none 의 0.400 다음으로 낮다.**
가는 선은 kxk 창을 원리적으로 채우지 못한다(길이 7 짜리 1픽셀 선 = 7/49 = 0.14).
창 모양을 선으로 바꾸면 같은 선이 1.0 이 된다.

**물어야 할 것은 셋이다.**
1. 선 필터가 Scratch 를 정사각 창보다 잘 가르는가.
2. 가른다면 **앙상블이 이미 맞히는 것** 위에서인가 **틀리는 것** 위에서인가.
   후자여야 값어치가 있다.
3. **Scratch 의 precision** — 앙상블이 Scratch 라 부른 것 안에서 참/거짓을 가르는가.

## min_dies 가 없으면 못 쓴다

창 안 다이 개수로 나누므로 가장자리에서 창이 잘리면 연속 3개만 불량이어도 1.0 이 된다.
그대로 쓰면 **정상 웨이퍼의 88% 가 포화**해서 정사각 창보다 한참 나쁘다(AUROC 0.63).
`min_dies=length` 로 온전한 창만 봐야 한다. 이 스크립트는 **둘 다** 찍는다.

비용 때문에 표본을 줄인다. 결함 전부 + 앙상블 오검출 전부 + 무작위 none 20,000장.
none 을 줄였으므로 AUPR 은 왜곡된다 — **AUROC 와 조건부 비교만 읽는다.**
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from a21_density import local_fail_density_map  # noqa: E402
from a22_line_filter import line_density_map  # noqa: E402

NAMES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
         "Random", "Scratch", "Near-full"]
SCRATCH = 7
OUT = "docs/research/local_density/evidence"
T0 = time.time()


def load_ensemble(y_expected=None):
    probs, ty = [], None
    for p in sorted(glob.glob("result/posthoc/*_tta_logits.npz")):
        z = np.load(p)
        ty = z["test_y"] if ty is None else ty
        a = z["test_logits"].astype(np.float64)
        ex = np.exp(a - a.max(1, keepdims=True))
        probs.append(ex / ex.sum(1, keepdims=True))
    if not probs:
        raise SystemExit("TTA 로짓이 없다")
    return np.mean(probs, 0), ty, len(probs)


def main() -> None:
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    te = sp["test"]
    y = d["y"].astype(np.int64)[te]
    P, ty, n_tta = load_ensemble()
    assert np.array_equal(ty, y)
    pred = P.argmax(1)
    print("TTA %d개 앙상블" % n_tta, flush=True)

    rng = np.random.default_rng(0)
    sub = np.sort(np.concatenate([
        np.flatnonzero(y != 0),
        np.flatnonzero((y == 0) & (pred != 0)),
        rng.choice(np.flatnonzero((y == 0) & (pred == 0)), 20000, replace=False)]))
    X = np.ascontiguousarray(d["X"][te][sub])
    ys, ps = y[sub], pred[sub]
    die = X > 0
    print("표본 %d장" % len(sub), flush=True)

    def mx(m):
        return np.where(die, m, -np.inf).reshape(len(m), -1).max(1)

    S = {}
    S["정사각 k=5"] = mx(local_fail_density_map(X, k=5))
    S["정사각 k=7"] = mx(local_fail_density_map(X, k=7))
    print("[%.0fs] 정사각 완료" % (time.time() - T0), flush=True)
    S["선 L=7 (가드 없음)"] = mx(line_density_map(X, length=7, n_orient=8))
    S["선 L=7 온전창"] = mx(line_density_map(X, length=7, n_orient=8, min_dies=7))
    S["선 L=11 온전창"] = mx(line_density_map(X, length=11, n_orient=8, min_dies=11))
    print("[%.0fs] 선 완료" % (time.time() - T0), flush=True)

    isd = (ys != 0).astype(int)
    res = {"n_tta_members": n_tta, "n_sample": int(len(sub)), "binary_auroc": {},
           "saturated_frac_none": {}, "per_class_auroc": {}, "on_errors": {},
           "scratch_precision": {}}

    print("\n== 이진 (none vs 결함) — 표본이라 AUROC 만 읽는다 ==")
    for n, s in S.items():
        a = float(roc_auc_score(isd, s))
        sat = float((s[ys == 0] >= 0.999).mean())
        res["binary_auroc"][n], res["saturated_frac_none"][n] = a, sat
        print("  %-18s AUROC %.4f   none 중 1.0 포화 %.3f" % (n, a, sat))

    print("\n== 클래스별 AUROC (그 클래스 대 none) ==")
    print("  %-18s " % "점수" + " ".join("%9s" % c for c in NAMES[1:]))
    for n, s in S.items():
        row = []
        for c in range(1, 9):
            m = (ys == c) | (ys == 0)
            row.append(float(roc_auc_score((ys[m] == c).astype(int), s[m])))
        res["per_class_auroc"][n] = dict(zip(NAMES[1:], row))
        print("  %-18s " % n + " ".join("%9.4f" % v for v in row))

    FP, TN = (ys == 0) & (ps != 0), (ys == 0) & (ps == 0)
    FN, TP = (ys != 0) & (ps == 0), (ys != 0) & (ps != 0)
    print("\n== 앙상블 오류 위에서 (FP %d TN %d FN %d TP %d) =="
          % (FP.sum(), TN.sum(), FN.sum(), TP.sum()))
    print("  %-18s %10s %10s" % ("점수", "FP 대 TN", "FN 대 TP"))
    for n, s in S.items():
        a1 = float(roc_auc_score(np.r_[np.ones(FP.sum()), np.zeros(TN.sum())],
                                 np.r_[s[FP], s[TN]]))
        a2 = float(roc_auc_score(np.r_[np.ones(FN.sum()), np.zeros(TP.sum())],
                                 np.r_[s[FN], s[TP]]))
        res["on_errors"][n] = {"fp_vs_tn": a1, "fn_vs_tp": a2}
        print("  %-18s %10.4f %10.4f" % (n, a1, a2))
    print("  고치려면 FP 대 TN < 0.5 이고 FN 대 TP > 0.5 여야 한다.")

    call, true = ps == SCRATCH, ys == SCRATCH
    tp_s, fp_s, fn_s = call & true, call & ~true, true & ~call
    print("\n== Scratch precision ==")
    print("  Scratch 라 부른 %d장 = 맞음 %d + 틀림 %d ; 놓친 참 Scratch %d장"
          % (call.sum(), tp_s.sum(), fp_s.sum(), fn_s.sum()))
    print("  틀린 것의 참 라벨:",
          {NAMES[int(c)]: int((ys[fp_s] == c).sum()) for c in np.unique(ys[fp_s])})
    for n, s in S.items():
        a = float(roc_auc_score(np.r_[np.ones(tp_s.sum()), np.zeros(fp_s.sum())],
                                np.r_[s[tp_s], s[fp_s]]))
        res["scratch_precision"][n] = a
        print("  %-18s 참Scratch 대 오탐 AUROC %.4f   중앙 %.3f vs %.3f"
              % (n, a, np.median(s[tp_s]), np.median(s[fp_s])))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "line_vs_square.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\n[%.0fs] 저장 -> %s/line_vs_square.json" % (time.time() - T0, OUT))


if __name__ == "__main__":
    main()
