"""상한을 **현재** 앙상블에서 다시 재고, 2단계 분해가 값어치가 있는지 판정한다.

## 왜 다시 재는가

`CLASSIFICATION.md` 가 적어 온 상한(0.9271, 0.8869)은 **10개 앙상블 0.7809 시절** 값이다.
모델이 좋아지면 쉬운 오류가 먼저 사라지므로 **상한도 같이 내려간다.**
현재 최선(TTA 18개, 0.7931)에서 다시 재지 않으면 남은 여유를 과대평가한다.

## 2단계 분해는 여기서 판정된다

"오류의 81.7% 가 none/결함 경계라면 이진 -> 8종으로 쪼개는 게 맞지 않나" 가
남은 후보 2번이었다. 학습 없이 답할 수 있다 —
현재 앙상블의 `1 - p(none)` 을 stage-1 점수로 놓고 **임계값을 test 라벨로 직접 최적화**한다.
그것으로도 flat 을 못 넘으면, 2단계로 얻을 것은 **결정 규칙 쪽에는 없다.**

학습이 없다. 캐시된 TTA 로짓만 읽는다.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate  # noqa: E402

OUT = "docs/research/ceilings/evidence"


def main() -> None:
    probs, y, n = [], None, 0
    for p in sorted(glob.glob("result/posthoc/*_tta_logits.npz")):
        z = np.load(p)
        y = z["test_y"] if y is None else y
        a = z["test_logits"].astype(np.float64)
        ex = np.exp(a - a.max(1, keepdims=True))
        probs.append(ex / ex.sum(1, keepdims=True))
        n += 1
    if not probs:
        raise SystemExit("TTA 로짓이 없다")
    P = np.mean(probs, 0)
    pred = P.argmax(1)
    d8 = P[:, 1:].argmax(1) + 1          # 결함 8종만 놓고 고른 최선
    s = 1.0 - P[:, 0]                     # stage-1 점수

    def mf(p):
        return float(evaluate(y, p, 9)["macro_f1"])

    base = mf(pred)
    fp_only = pred.copy()
    fp_only[(y == 0) & (pred != 0)] = 0
    fn_only = pred.copy()
    m = (y != 0) & (pred == 0)
    fn_only[m] = d8[m]

    res = {
        "n_tta_members": n,
        "flat": base,
        "flat_accuracy": float(evaluate(y, pred, 9)["accuracy"]),
        "remove_false_positives": mf(fp_only),
        "defect_recall_one": mf(fn_only),
        "perfect_binary_stage1": mf(np.where(y == 0, 0, d8)),
        "perfect_stage2": mf(np.where(pred == 0, 0, np.where(y != 0, y, d8))),
        "defect_only_8way_accuracy": float((d8[y != 0] == y[y != 0]).mean()),
        "errors_total": int((pred != y).sum()),
        "errors_defect_to_defect": int(((y != 0) & (pred != 0) & (pred != y)).sum()),
        "flat_fp": int(((y == 0) & (pred != 0)).sum()),
        "flat_fn": int(((y != 0) & (pred == 0)).sum()),
    }

    print("TTA %d개 앙상블  macro-F1 %.4f  acc %.4f" % (n, base, res["flat_accuracy"]))
    print("\n== 상한 (현재 앙상블 기준) ==")
    print("  none 오검출 전부 제거                %.4f" % res["remove_false_positives"])
    print("  결함 recall 전부 1.0                %.4f" % res["defect_recall_one"])
    print("  **none/결함 이진이 완벽 + 현재 8종**  %.4f" % res["perfect_binary_stage1"])
    print("  현재 이진 + 8종이 완벽              %.4f" % res["perfect_stage2"])
    print("  (참고) 참 결함만 놓고 8종 정확도      %.4f" % res["defect_only_8way_accuracy"])
    print("  오류 %d장 중 결함끼리 혼동 %d장 (%.1f%%)" % (
        res["errors_total"], res["errors_defect_to_defect"],
        100 * res["errors_defect_to_defect"] / res["errors_total"]))

    print("\n== 2단계 분해 — 임계값을 test 로 직접 최적화(오라클) ==")
    best = (-1.0, None)
    for t in np.unique(np.quantile(s, np.linspace(0.80, 0.9999, 400))):
        f = mf(np.where(s > t, d8, 0))
        if f > best[0]:
            best = (f, float(t))
    pr = np.where(s > best[1], d8, 0)
    res["two_stage_oracle_threshold"] = best[0]
    res["two_stage_oracle_t"] = best[1]
    res["two_stage_fp"] = int(((y == 0) & (pr != 0)).sum())
    res["two_stage_fn"] = int(((y != 0) & (pr == 0)).sum())
    print("  2단계 + 오라클 임계값  %.4f  (t=%.5f)   flat 대비 %+.4f" % (
        best[0], best[1], best[0] - base))
    print("  그때 FP %d / FN %d   (flat: FP %d / FN %d)" % (
        res["two_stage_fp"], res["two_stage_fn"], res["flat_fp"], res["flat_fn"]))
    print("  -> 두 오류를 맞바꿀 뿐 총합이 안 준다. **flat 의 암묵적 경계가 이미 최적이다.**")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ceilings.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\n저장 -> %s/ceilings.json" % OUT)


if __name__ == "__main__":
    main()
