"""신호가 결함 **종류**를 아는가 — 3단계(MLLM 8종 분류) 준비 측정.

반증 조건은 `candidate/ood_class_information.md` 에 실행 전에 박았다.
**분류기를 만들지 않는다.** 결함끼리 두 종을 고르는 28쌍에 대해 신호 하나로 매긴 AUROC 다.
라벨은 평가에만 쓴다.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import fail_count  # noqa: E402
from a22_ood_template import assert_one_class  # noqa: E402
from a23_ood_template_eval import NAMES, _fast_auroc, bootstrap_auroc_ci, ecdf_percentile, fisher  # noqa: E402
from a24_ood_residual import local_fail_count_map, local_fail_count_max  # noqa: E402
from a27_line_filter import line_density_max  # noqa: E402
from a29_radial_calibration import band_index, calibrate_map, fit_band_reference  # noqa: E402

OUT = Path("result/ood/o1_classinfo")
CHUNK, N_BANDS, L_LINE = 4000, 32, 11
STRONG = 0.80
T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    tr, te = sp["train"], sp["test"]
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    trn = tr[y[tr] == 0]
    assert_one_class(y[trn])
    X = d["X"]
    pad_tr = np.ascontiguousarray(X[trn])
    pad_te = np.ascontiguousarray(X[te])
    del X, d
    yte = y[te]
    log("적재 완료")

    v_tr = local_fail_count_map(pad_tr, k=5, chunk=CHUNK).astype(np.float32)
    ref = fit_band_reference(v_tr, pad_tr, y=y[trn], n_bands=N_BANDS, seed=0)
    del v_tr

    def a_score(x):
        out = np.empty(len(x))
        for a in range(0, len(x), CHUNK):
            xb = x[a:a + CHUNK]
            u = calibrate_map(local_fail_count_map(xb, k=5, chunk=CHUNK),
                              band_index(xb, N_BANDS), ref)
            out[a:a + len(xb)] = u.reshape(len(xb), -1).max(1)
        return out

    A_te, A_tr = a_score(pad_te), a_score(pad_tr)
    B_te = line_density_max(pad_te, length=L_LINE, n_orient=8, chunk=CHUNK, min_dies=L_LINE)
    B_tr = line_density_max(pad_tr, length=L_LINE, n_orient=8, chunk=CHUNK, min_dies=L_LINE)
    C_te = local_fail_count_max(pad_te, k=3, chunk=CHUNK)
    C_tr = local_fail_count_max(pad_tr, k=3, chunk=CHUNK)
    log("요소 3종 완료")

    fuse = fisher([ecdf_percentile(t, e) for t, e in
                   ((A_tr, A_te), (B_tr, B_te), (C_tr, C_te))])
    eps = 1e-6
    ratio = np.log(B_te + eps) - np.log(C_te + eps)

    signals = {
        "대조 불량 다이 개수 (A범위)": fail_count(pad_te),
        "A: k2 5x5 + 반경보정 (A범위)": A_te,
        "B: 온전창 선 L=11 (A범위)": B_te,
        "C: k2 3x3 (A범위)": C_te,
        "융합 fuse3 (A범위)": fuse,
        "파생 log(B)-log(C) (B범위)": ratio,
    }
    log("신호 준비 완료")

    defect_cls = list(range(1, 9))
    pairs = list(itertools.combinations(defect_cls, 2))
    table, strong_sets = {}, {}
    for name, s in signals.items():
        rows = {}
        strong = set()
        for c1, c2 in pairs:
            m = (yte == c1) | (yte == c2)
            lab = (yte[m] == c2).astype(np.int64)
            a = _fast_auroc(s[m], lab)
            sep = max(a, 1 - a)          # 부호는 무관하다 — 가르는가만 본다
            key = "%s|%s" % (NAMES[c1], NAMES[c2])
            row = {"auroc": a, "separation": sep,
                   "n1": int((yte == c1).sum()), "n2": int((yte == c2).sum())}
            if min(row["n1"], row["n2"]) < 300:
                lo, hi = bootstrap_auroc_ci(s[m], lab, n_boot=300, seed=c1 * 10 + c2)
                row["auroc_ci"] = [lo, hi]
                row["ci_width"] = hi - lo
            rows[key] = row
            if sep >= STRONG:
                strong.add(key)
        table[name] = rows
        strong_sets[name] = strong
        log("%-30s 강한 쌍 %2d/28  평균 분리도 %.4f"
            % (name, len(strong), np.mean([r["separation"] for r in rows.values()])))

    print("\n== 신호별 요약 ==")
    print("%-30s %10s %12s %12s" % ("신호", "강한 쌍", "평균 분리도", "최대 분리도"))
    for name, rows in table.items():
        seps = [r["separation"] for r in rows.values()]
        print("%-30s %6d/28 %12.4f %12.4f"
              % (name, len(strong_sets[name]), np.mean(seps), np.max(seps)))

    print("\n== 조건 2: 잘 가르는 쌍이 서로 다른가 (강한 쌍 집합의 겹침) ==")
    keys = ["A: k2 5x5 + 반경보정 (A범위)", "B: 온전창 선 L=11 (A범위)", "C: k2 3x3 (A범위)"]
    for i, j in itertools.combinations(range(3), 2):
        a, b = strong_sets[keys[i]], strong_sets[keys[j]]
        inter = len(a & b)
        union = len(a | b) or 1
        print("  %-24s vs %-24s 교집합 %2d / 합집합 %2d (Jaccard %.2f)"
              % (keys[i].split(":")[0], keys[j].split(":")[0], inter, union, inter / union))
    allthree = strong_sets[keys[0]] | strong_sets[keys[1]] | strong_sets[keys[2]]
    print("  세 신호를 합치면 강한 쌍 %d/28, 각각은 %s"
          % (len(allthree), [len(strong_sets[k]) for k in keys]))

    print("\n== 조건 3: 대조군(불량 다이 개수)을 넘는 쌍 수 ==")
    base = table["대조 불량 다이 개수 (A범위)"]
    for name, rows in table.items():
        if name.startswith("대조"):
            continue
        better = sum(1 for k in rows if rows[k]["separation"] > base[k]["separation"])
        print("  %-30s 28쌍 중 %2d 쌍에서 대조군을 넘는다" % (name, better))

    print("\n== 조건 4: Scratch 대 Center ==")
    key = "Center|Scratch"
    for name, rows in table.items():
        r = rows[key]
        print("  %-30s 분리도 %.4f (AUROC %.4f)" % (name, r["separation"], r["auroc"]))

    print("\n== 강한 쌍 목록 (분리도 >= 0.80) ==")
    for name in keys + ["파생 log(B)-log(C) (B범위)", "대조 불량 다이 개수 (A범위)"]:
        rows = table[name]
        best = sorted(rows.items(), key=lambda kv: -kv[1]["separation"])[:6]
        print("  %s" % name)
        for k, r in best:
            note = ""
            if "ci_width" in r:
                note = "  CI %.3f~%.3f%s" % (r["auroc_ci"][0], r["auroc_ci"][1],
                                             "  (CI 폭 > 0.15, 결론 보류)"
                                             if r["ci_width"] > 0.15 else "")
            print("     %-26s %.4f%s" % (k, r["separation"], note))

    (OUT / "class_information.json").write_text(
        json.dumps({"table": table,
                    "strong": {k: sorted(v) for k, v in strong_sets.items()}},
                   ensure_ascii=False, indent=2))
    log("저장 완료 → %s" % OUT)
