"""3-요소 융합 평가. 반증 조건은 `candidate/ood_three_way_fusion.md` 에 실행 전에 박았다.

주 지표 `aupr_blocked`. 융합 보정은 **train none 만** 쓴다(백분위 → Fisher, 동일 가중).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import aupr, aupr_blocked, auroc, fpr_at_tpr, operating_points  # noqa: E402
from a22_ood_template import assert_one_class  # noqa: E402
from a23_ood_template_eval import (  # noqa: E402
    BIN_LBL,
    BINS,
    NAMES,
    _fast_auroc,
    bootstrap_auroc_ci,
    ecdf_percentile,
    fisher,
    paired_aupr_blocked_diff_ci,
)
from a24_ood_residual import local_fail_count_map, local_fail_count_max  # noqa: E402
from a27_line_filter import line_density_max  # noqa: E402
from a29_radial_calibration import band_index, calibrate_map, fit_band_reference  # noqa: E402

OUT = Path("result/ood/o1_fuse3")
CHUNK = 4000
N_BANDS = 32
L_LINE = 11
PRIMARY = "주 fuse3 (k2k5+반경 / 선L11 / k2k3)"
CTRL2 = "대조 2요소 융합 (6차 최선)"
CTRL_NOLINE = "대조 선없음 (k2k5+반경 / k2k3)"
T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def blocked_ci(score, label, n_boot=300, seed=0):
    n = len(label)
    rng = np.random.default_rng(seed)
    v = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        v[b] = aupr_blocked(score[i], label[i])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def per_class_auroc(score, yte, ci_for=("Scratch", "Center", "Loc", "Edge-Ring")):
    ns = score[yte == 0]
    z = np.zeros(len(ns), np.int64)
    rows = {}
    for c in range(1, 9):
        m = yte == c
        s = np.concatenate([ns, score[m]])
        l = np.concatenate([z, np.ones(int(m.sum()), np.int64)])
        row = {"n": int(m.sum()), "auroc": _fast_auroc(s, l)}
        if NAMES[c] in ci_for:
            row["auroc_ci"] = bootstrap_auroc_ci(s, l, n_boot=200, seed=c)
        rows[NAMES[c]] = row
    return rows


def per_size(score, is_def, size):
    out = {}
    for i, lbl in enumerate(BIN_LBL):
        m = (size >= BINS[i]) & (size < BINS[i + 1])
        if m.sum() and is_def[m].sum() not in (0, m.sum()):
            out[lbl] = aupr_blocked(score[m], is_def[m])
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    tr, te = sp["train"], sp["test"]
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    size = d["die_size"].astype(np.float64)
    trn = tr[y[tr] == 0]
    assert_one_class(y[trn])
    X = d["X"]
    pad_tr = np.ascontiguousarray(X[trn])
    pad_te = np.ascontiguousarray(X[te])
    del X, d
    yte = y[te]
    is_def = (yte != 0).astype(np.int64)
    size_te = size[te]
    log("적재 완료")

    v_tr = local_fail_count_map(pad_tr, k=5, chunk=CHUNK).astype(np.float32)
    ref = fit_band_reference(v_tr, pad_tr, y=y[trn], n_bands=N_BANDS, seed=0)
    del v_tr

    def calib(x):
        out = np.empty(len(x))
        for a in range(0, len(x), CHUNK):
            xb = x[a:a + CHUNK]
            u = calibrate_map(local_fail_count_map(xb, k=5, chunk=CHUNK),
                              band_index(xb, N_BANDS), ref)
            out[a:a + len(xb)] = u.reshape(len(xb), -1).max(1)
        return out

    comp = {}
    comp["A_k2k5cal"] = (calib(pad_tr), calib(pad_te))
    log("요소 A (k2 5x5 + 반경보정) 완료")
    comp["B_lineL11"] = (line_density_max(pad_tr, length=L_LINE, n_orient=8, chunk=CHUNK,
                                          min_dies=L_LINE),
                         line_density_max(pad_te, length=L_LINE, n_orient=8, chunk=CHUNK,
                                          min_dies=L_LINE))
    log("요소 B (온전창 선 L=11) 완료")
    comp["C_k2k3"] = (local_fail_count_max(pad_tr, k=3, chunk=CHUNK),
                      local_fail_count_max(pad_te, k=3, chunk=CHUNK))
    log("요소 C (k2 3x3) 완료")

    def fuse(keys):
        us = [ecdf_percentile(comp[k][0], comp[k][1]) for k in keys]
        return fisher(us)

    scores = {
        "요소 A: k2 5x5 + 반경보정": comp["A_k2k5cal"][1],
        "요소 B: 온전창 선 L=11": comp["B_lineL11"][1],
        "요소 C: k2 3x3": comp["C_k2k3"][1],
        CTRL2: fuse(["A_k2k5cal", "B_lineL11"]),
        CTRL_NOLINE: fuse(["A_k2k5cal", "C_k2k3"]),
        PRIMARY: fuse(["A_k2k5cal", "B_lineL11", "C_k2k3"]),
        "탐색 B+C (A 없음)": fuse(["B_lineL11", "C_k2k3"]),
    }
    log("융합 완료")

    report = {}
    for si, (name, s) in enumerate(scores.items()):
        report[name] = {
            "auroc": auroc(s, is_def), "aupr_blocked": aupr_blocked(s, is_def),
            "aupr_order": aupr(s, is_def), "fpr_at_95tpr": fpr_at_tpr(s, is_def, 0.95),
            "n_unique": int(len(np.unique(s))),
            "aupr_blocked_ci95": blocked_ci(s, is_def, 300, 8000 + si),
            "per_class": per_class_auroc(s, yte),
            "per_size_bin": per_size(s, is_def, size_te),
            "operating": operating_points(s, is_def, (0.5, 0.8, 0.95)),
        }
        log("%-34s AUROC %.4f AUPRblk %.4f FPR@95 %.4f"
            % (name, report[name]["auroc"], report[name]["aupr_blocked"],
               report[name]["fpr_at_95tpr"]))

    order = sorted(report.items(), key=lambda kv: -kv[1]["aupr_blocked"])
    print("\n== 전체 (blocked AUPR 내림차순) ==")
    print("%-34s %8s %10s %20s %9s %8s"
          % ("arm", "AUROC", "AUPR블록", "95% CI", "FPR@95", "고유값"))
    for name, m in order:
        print("%-34s %8.4f %10.4f  [%.4f, %.4f] %9.4f %8d"
              % (name, m["auroc"], m["aupr_blocked"], m["aupr_blocked_ci95"][0],
                 m["aupr_blocked_ci95"][1], m["fpr_at_95tpr"], m["n_unique"]))

    print("\n== 조건 1, 6: 짝지은 blocked 비교 ==")
    for a, b in ((PRIMARY, CTRL2), (PRIMARY, CTRL_NOLINE), (CTRL_NOLINE, CTRL2)):
        lo, hi, p = paired_aupr_blocked_diff_ci(scores[a], scores[b], is_def, 300, 31)
        print("  %-48s [%+.4f, %+.4f] p=%.4f  %s"
              % (a + " vs " + b, lo, hi, p,
                 "이김" if lo > 0 else ("짐" if hi < 0 else "무승부")))

    print("\n== 조건 2, 3: 클래스별 CI ==")
    for name, m in order:
        parts = ["%s %.4f [%.4f,%.4f]" % (c, m["per_class"][c]["auroc"],
                                          m["per_class"][c]["auroc_ci"][0],
                                          m["per_class"][c]["auroc_ci"][1])
                 for c in ("Center", "Scratch", "Loc", "Edge-Ring")]
        print("  %-34s %s" % (name, "  ".join(parts)))

    print("\n== 클래스별 AUROC (전부) ==")
    cls = NAMES[1:]
    print("%-34s " % "arm" + " ".join("%10s" % c for c in cls))
    for name, m in order:
        print("%-34s " % name + " ".join("%10.4f" % m["per_class"][c]["auroc"] for c in cls))

    print("\n== 운영 지점 ==")
    for name, m in order:
        cells = ["%0.3f/%6d" % (r["precision"], r["false_positives"]) for r in m["operating"]]
        print("  %-34s 50%% %s  80%% %s  95%% %s" % (name, cells[0], cells[1], cells[2]))

    print("\n== dieSize 구간별 blocked AUPR ==")
    print("%-34s " % "arm" + " ".join("%11s" % b for b in BIN_LBL))
    for name, m in order:
        print("%-34s " % name + " ".join(
            "%11.4f" % m["per_size_bin"][b] if b in m["per_size_bin"] else "%11s" % "-"
            for b in BIN_LBL))

    (OUT / "fuse3_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    np.savez_compressed(OUT / "fuse3_scores.npz",
                        **{k: v.astype(np.float64) for k, v in scores.items()},
                        y_test=yte, size_test=size_te)
    log("저장 완료 → %s" % OUT)
