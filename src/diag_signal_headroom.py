"""보조 신호가 앙상블 위에 보탤 것이 남아 있는가 — 세 단계로 나눠 묻는다 (학습 없음).

## 이 스크립트는 내 앞선 분석의 교란을 잡으려고 썼다

`diag_density_complementarity.py` 는 `FP 대 TN`, `FN 대 TP` 를 재고
"보조 신호가 모델과 같은 편이라 오류를 못 고친다" 고 결론냈다. **그 비교는 교란돼 있다.**

FP 는 **모델이 결함이라 부른** 웨이퍼다. 모델 점수는 국소 밀도와 상관이 있으므로
"모델이 결함이라 부른 것" 을 고르는 순간 이미 밀도가 높은 쪽을 고른 것이다.
TN 과 비교하면 그 선택 효과가 그대로 AUROC 로 나온다. **신호의 성질이 아니라 선택의 성질이다.**

## 그래서 세 단계로 다시 묻는다

1. **선택을 맞춘 비교.** `FP 대 TP`(둘 다 결함이라 불림), `FN 대 TN`(둘 다 none 이라 불림).
   같은 예측을 받은 것끼리 비교하므로 선택 효과가 상쇄된다.
   오검출을 걸러내려면 `FP 대 TP < 0.5`, 놓친 결함을 되살리려면 `FN 대 TN > 0.5` 여야 한다.
2. **모델 확신도로 층화.** 1번에서 신호가 남아도 그것이 모델 점수의 복사본일 수 있다.
   모델 점수 10분위 안에서 다시 재서 **조건부로 남는 정보**만 본다.
3. **실제로 쓰면 오르는가.** 순위 결합 `rank(모델) + w * rank(보조)` 의
   `w` 와 임계값을 **test 라벨로 직접 최적화**한다. 여기서 `w=0` 이 최선이면
   보조 신호는 **결정 규칙 쪽에 보탤 것이 없다.**

1, 2번은 "정보가 있느냐" 이고 3번은 "쓸 수 있느냐" 다. **둘은 다르다.**
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate  # noqa: E402
from a21_density import local_fail_density_map  # noqa: E402
from a22_line_filter import line_density_map  # noqa: E402

OUT = "docs/research/local_density/evidence"


def rank(v):
    r = np.empty(len(v), np.float64)
    r[np.argsort(v, kind="stable")] = np.arange(len(v))
    return r / (len(v) - 1)


def main() -> None:
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    te = sp["test"]
    y = d["y"].astype(int)[te]
    lot = d["lot_name"][te]

    probs, ty = [], None
    for p in sorted(glob.glob("result/posthoc/*_tta_logits.npz")):
        z = np.load(p)
        ty = z["test_y"] if ty is None else ty
        a = z["test_logits"].astype(np.float64)
        ex = np.exp(a - a.max(1, keepdims=True))
        probs.append(ex / ex.sum(1, keepdims=True))
    P = np.mean(probs, 0)
    assert np.array_equal(ty, y)
    pred, d8, s = P.argmax(1), P[:, 1:].argmax(1) + 1, 1.0 - P[:, 0]

    X = np.ascontiguousarray(d["X"][te])
    die = X > 0
    def mx(m):
        return np.where(die, m, -np.inf).reshape(len(m), -1).max(1)
    dens = mx(local_fail_density_map(X, k=5))
    line = mx(line_density_map(X, length=11, n_orient=8, min_dies=11))
    u, inv = np.unique(lot, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(u)).astype(float)
    lotc = (np.bincount(inv, weights=s, minlength=len(u))[inv] - s) / np.maximum(cnt[inv] - 1, 1)

    SIG = {"모델 자신 점수": s, "국소 밀도 k=5": dens, "선 L=11 온전창": line, "lot 문맥": lotc}
    FP, TN = (y == 0) & (pred != 0), (y == 0) & (pred == 0)
    FN, TP = (y != 0) & (pred == 0), (y != 0) & (pred != 0)
    res = {"counts": {"FP": int(FP.sum()), "TN": int(TN.sum()),
                      "FN": int(FN.sum()), "TP": int(TP.sum())}}

    def auc(a, b, v):
        return float(roc_auc_score(np.r_[np.ones(a.sum()), np.zeros(b.sum())],
                                   np.r_[v[a], v[b]]))

    print("FP %d  TN %d  FN %d  TP %d" % (FP.sum(), TN.sum(), FN.sum(), TP.sum()))
    print("\n== 0. 교란된 비교 (기록용. 이 값들을 결론에 쓰면 안 된다) ==")
    print("  %-16s %10s %10s" % ("신호", "FP 대 TN", "FN 대 TP"))
    res["confounded"] = {}
    for n, v in SIG.items():
        a1, a2 = auc(FP, TN, v), auc(FN, TP, v)
        res["confounded"][n] = [a1, a2]
        print("  %-16s %10.4f %10.4f" % (n, a1, a2))
    print("  모델 자신 점수가 0.9999 / 0.0008 인 것이 이 비교가 무엇을 재는지 보여준다 —")
    print("  예측이 곧 점수의 함수라 **정의상** 극단값이 나온다. 신호의 성질이 아니다.")

    print("\n== 1. 선택을 맞춘 비교 ==")
    print("  %-16s %10s %10s" % ("신호", "FP 대 TP", "FN 대 TN"))
    res["matched"] = {}
    for n, v in SIG.items():
        a1, a2 = auc(FP, TP, v), auc(FN, TN, v)
        res["matched"][n] = [a1, a2]
        print("  %-16s %10.4f %10.4f" % (n, a1, a2))
    print("  오검출을 걸러내려면 FP 대 TP < 0.5, 놓친 결함을 되살리려면 FN 대 TN > 0.5")

    print("\n== 2. 모델 확신도 10분위로 층화한 뒤 남는 정보 ==")

    def strat(a, b, v, nb=10):
        m = a | b
        q = np.quantile(s[m], np.linspace(0, 1, nb + 1))
        q[-1] += 1e-9
        vals, ws = [], []
        for i in range(nb):
            sel = m & (s >= q[i]) & (s < q[i + 1])
            pa, pb = (a & sel).sum(), (b & sel).sum()
            if pa < 10 or pb < 10:
                continue
            vals.append(auc(a & sel, b & sel, v))
            ws.append(pa + pb)
        return (float(np.average(vals, weights=ws)) if vals else float("nan")), len(vals)

    print("  %-16s %14s %14s" % ("신호", "FP 대 TP", "FN 대 TN"))
    res["stratified"] = {}
    for n, v in SIG.items():
        if n == "모델 자신 점수":
            continue
        (a1, n1), (a2, n2) = strat(FP, TP, v), strat(FN, TN, v)
        res["stratified"][n] = [a1, a2]
        print("  %-16s %8.4f(%2d층) %8.4f(%2d층)" % (n, a1, n1, a2, n2))
    print("  0.5 = 모델 확신도 위에 보탤 것이 없다")

    print("\n== 3. 실제로 결정 규칙에 넣으면 오르는가 (오라클 w, 오라클 임계값) ==")
    base = float(evaluate(y, pred, 9)["macro_f1"])
    print("  현재 flat  %.4f" % base)
    rs = rank(s)
    res["oracle_combo"] = {}
    for n, v in [("국소 밀도 k=5", dens), ("선 L=11 온전창", line), ("lot 문맥", lotc)]:
        ra = rank(v)
        best = (-1.0, 0.0)
        for w in np.linspace(0, 1.0, 11):
            z2 = rs + w * ra
            for t in np.quantile(z2, np.linspace(0.85, 0.9995, 120)):
                f = float(evaluate(y, np.where(z2 > t, d8, 0), 9)["macro_f1"])
                if f > best[0]:
                    best = (f, float(w))
        res["oracle_combo"][n] = {"best": best[0], "w": best[1], "delta": best[0] - base}
        print("  %-16s 최고 %.4f  (w=%.1f)  flat 대비 %+.4f" % (n, best[0], best[1], best[0] - base))
    print("  **w=0 이 최선이면 그 신호는 결정 규칙에 보탤 것이 없다.**")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "signal_headroom.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\n저장 -> %s/signal_headroom.json" % OUT)


if __name__ == "__main__":
    main()
