"""per-class F1 비교와 오류 총량 분해 (question 4 근거)."""
import json
from pathlib import Path

import numpy as np

RES = Path("result/cls_baseline")
CLASSES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Loc", "Random", "Scratch", "Near-full"]
EXPS = {"E1": "e1_scratch_best_test.json", "E3": "e3_aug_best_test.json",
        "E4": "e4_cw_best_test.json", "E4b": "e4_aug_cw_best_test.json"}


def prf(C):
    tp = np.diag(C).astype(float)
    rec = tp / np.maximum(C.sum(1), 1)
    prec = tp / np.maximum(C.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return prec, rec, f1


mats = {t: np.array(json.loads((RES / f).read_text())["confusion"]) for t, f in EXPS.items()}

print("== per-class F1 (test) ==")
print(f"{'class':10s} " + "".join(f"{t:>8s}" for t in EXPS))
f1s = {t: prf(C)[2] for t, C in mats.items()}
for i, c in enumerate(CLASSES):
    print(f"{c:10s} " + "".join(f"{f1s[t][i]:8.3f}" for t in EXPS))
print(f"{'macro':10s} " + "".join(f"{f1s[t].mean():8.3f}" for t in EXPS))

print("\n== per-class precision (test) ==")
print(f"{'class':10s} " + "".join(f"{t:>8s}" for t in EXPS))
for i, c in enumerate(CLASSES):
    print(f"{c:10s} " + "".join(f"{prf(mats[t])[0][i]:8.3f}" for t in EXPS))

print("\n== 오류 총량 분해 (결함 8클래스 행 기준) ==")
print(f"{'exp':4s} {'결함오류합':>8s} {'->none(FN)':>10s} {'결함간혼동':>9s} {'none->결함(FP)':>12s} "
      f"{'defect-vs-none 이진 recall':>12s} {'이진 precision':>10s}")
for t, C in mats.items():
    err = C[1:].sum() - np.diag(C)[1:].sum()
    to_none = C[1:, 0].sum()
    inter = err - to_none
    fp = C[0, 1:].sum()
    n_def = C[1:].sum()
    det = n_def - to_none  # 결함을 결함(어느 클래스든)으로
    prec_b = det / (det + fp)
    print(f"{t:4s} {err:8d} {to_none:10d} {inter:9d} {fp:12d} {det/n_def:12.3f} {prec_b:10.3f}")

print("\n== 만약 X 가 개선되면 macro-F1 이 얼마나 오르나 (E4b 기준 시뮬레이션) ==")
C = mats["E4b"].copy().astype(float)
# (a) none 흡수(FN)를 전부 정답으로 되돌리면
Ca = C.copy()
for i in range(1, 9):
    Ca[i, i] += Ca[i, 0]; Ca[i, 0] = 0
# (b) 결함 간 혼동을 전부 정답으로 되돌리면
Cb = C.copy()
for i in range(1, 9):
    for j in range(1, 9):
        if i != j:
            Cb[i, i] += Cb[i, j]; Cb[i, j] = 0
# (c) none 오탐(FP)을 전부 none 으로 되돌리면
Cc = C.copy()
Cc[0, 0] += Cc[0, 1:].sum(); Cc[0, 1:] = 0
for nm, M in (("현재 E4b", C), ("(a) FN->정답", Ca), ("(b) 결함간 혼동 해소", Cb), ("(c) none FP 제거", Cc)):
    print(f"   {nm:20s} macro-F1 {prf(M)[2].mean():.4f}")
