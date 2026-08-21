"""E1/E3/E4/E4b test 혼동 행렬 정독.

result/cls_baseline/*_test.json 의 confusion(행=정답, 열=예측 가정, 검증 포함)을 읽어
클래스별 오분류 목적지, none 흡수 vs 결함 간 혼동을 정량화한다.
"""
import json
from pathlib import Path

import numpy as np

RES = Path("result/cls_baseline")
CLASSES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Loc", "Random", "Scratch", "Near-full"]
EXPS = {
    "E1": "e1_scratch_best_test.json",
    "E3": "e3_aug_best_test.json",
    "E4": "e4_cw_best_test.json",
    "E4b": "e4_aug_cw_best_test.json",
}


def load(fname):
    d = json.loads((RES / fname).read_text())
    C = np.array(d["confusion"], dtype=np.int64)
    return d, C


def check_axis(d, C):
    """confusion 축 방향 검증: 행 합 == support(정답 개수) 인지 확인."""
    sup = np.array([d["per_class_support"][c] if isinstance(d.get("per_class_support"), dict)
                    else 0 for c in CLASSES]) if "per_class_support" in d else None
    row_sums = C.sum(axis=1)
    col_sums = C.sum(axis=0)
    # recall 재계산으로 교차 검증
    rec_row = np.diag(C) / np.maximum(row_sums, 1)
    return row_sums, col_sums, rec_row


def main():
    mats = {}
    for tag, fname in EXPS.items():
        d, C = load(fname)
        mats[tag] = (d, C)
        keys = sorted(d.keys())
        print(f"== {tag} ({fname}) keys: {keys}")
        row_sums, col_sums, rec_row = check_axis(d, C)
        print(f"   row_sums = {row_sums.tolist()}")
        print(f"   recall(행 기준) = {[round(float(r),3) for r in rec_row]}")
        if "per_class_recall" in d:
            pr = d["per_class_recall"]
            if isinstance(pr, dict):
                pr = [pr[str(i)] for i in range(9)]
            print(f"   json recall     = {[round(float(r),3) for r in pr]}")
        print()

    for tag in EXPS:
        d, C = mats[tag]
        n = C.sum()
        print(f"\n########## {tag} — 혼동 행렬 (행=정답, 열=예측) ##########")
        header = "true\\pred".ljust(10) + "".join(c[:8].rjust(9) for c in CLASSES)
        print(header)
        for i, c in enumerate(CLASSES):
            print(c[:9].ljust(10) + "".join(str(int(v)).rjust(9) for v in C[i]))
        print(f"\n-- {tag} 행 정규화(%) --")
        print(header)
        for i, c in enumerate(CLASSES):
            row = C[i] / max(C[i].sum(), 1) * 100
            print(c[:9].ljust(10) + "".join(f"{v:.1f}".rjust(9) for v in row))
        # precision
        prec = np.diag(C) / np.maximum(C.sum(axis=0), 1)
        print(f"\n-- {tag} precision --")
        for i, c in enumerate(CLASSES):
            print(f"   {c:10s} P={prec[i]:.3f}  (예측 수 {int(C.sum(axis=0)[i]):,})")
        # none 흡수 vs 결함 간 혼동
        print(f"\n-- {tag} 결함 클래스 오류 분해 --")
        print(f"{'class':10s} {'support':>8s} {'errors':>7s} {'->none':>7s} {'->결함':>7s} {'none흡수율':>9s}  주요 목적지")
        for i in range(1, 9):
            sup = int(C[i].sum())
            err = sup - int(C[i, i])
            to_none = int(C[i, 0])
            to_def = err - to_none
            dests = [(CLASSES[j], int(C[i, j])) for j in range(9) if j != i and C[i, j] > 0]
            dests.sort(key=lambda t: -t[1])
            dest_str = ", ".join(f"{c} {v}({v/max(sup,1)*100:.0f}%)" for c, v in dests[:4])
            print(f"{CLASSES[i]:10s} {sup:8d} {err:7d} {to_none:7d} {to_def:7d} {to_none/max(err,1)*100:8.1f}%  {dest_str}")
        # none 오탐 (none이 결함으로)
        n_none = int(C[0].sum())
        fp = n_none - int(C[0, 0])
        print(f"\n-- {tag} none({n_none:,}) 의 오탐 {fp:,} ({fp/n_none*100:.2f}%) 목적지 --")
        dests = [(CLASSES[j], int(C[0, j])) for j in range(1, 9)]
        dests.sort(key=lambda t: -t[1])
        print("   " + ", ".join(f"{c} {v}" for c, v in dests))

    # 실험 간 델타: 특정 셀 이동 추적
    print("\n\n########## 실험 간 주요 셀 변화 ##########")
    pairs = [("E1", "E3"), ("E1", "E4"), ("E3", "E4b"), ("E4", "E4b"), ("E1", "E4b")]
    focus = ["Loc", "Scratch", "Center", "Edge-Ring", "Random", "Donut"]
    for a, b in pairs:
        Ca, Cb = mats[a][1], mats[b][1]
        D = Cb - Ca
        print(f"\n-- {a} -> {b} : 결함 행 델타 --")
        for cname in focus:
            i = CLASSES.index(cname)
            moves = [(CLASSES[j], int(D[i, j])) for j in range(9) if D[i, j] != 0]
            moves.sort(key=lambda t: -abs(t[1]))
            print(f"   {cname:10s} " + ", ".join(f"{c} {v:+d}" for c, v in moves))


if __name__ == "__main__":
    main()
