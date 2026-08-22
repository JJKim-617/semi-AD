"""온전 창 가드 평가. 반증 조건은 `candidate/ood_full_window_guard.md` 에 실행 전에 박았다.

주 지표 `aupr_blocked`. 조건 7 을 위해 **가드 때문에 점수가 0 이 되는 웨이퍼 수**를 센다.
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
from a24_ood_residual import (  # noqa: E402
    local_fail_count_map,
    local_fail_density_max,
)
from a27_line_filter import line_density_max  # noqa: E402
from a29_radial_calibration import band_index, calibrate_map, fit_band_reference  # noqa: E402

OUT = Path("result/ood/o1_fullwindow")
LS = (5, 7, 9, 11, 15)
CHUNK = 4000
N_BANDS = 32
BEST = "대조 현재최선 k2k5+반경보정"
PRIMARY = "주 line7_full"
T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def blocked_ci(score, label, n_boot=300, seed=0):
    n = len(label)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        vals[b] = aupr_blocked(score[i], label[i])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def per_class_auroc(score, yte, ci_for=("Scratch", "Center", "Loc")):
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
    rows = {}
    for i, lbl in enumerate(BIN_LBL):
        m = (size >= BINS[i]) & (size < BINS[i + 1])
        if m.sum() == 0 or is_def[m].sum() in (0, m.sum()):
            continue
        rows[lbl] = aupr_blocked(score[m], is_def[m])
    return rows


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

    scores, tr_scores = {}, {}
    for L in LS:
        scores["line%d 가드없음" % L] = line_density_max(pad_te, length=L, n_orient=8,
                                                     chunk=CHUNK)
        s = line_density_max(pad_te, length=L, n_orient=8, chunk=CHUNK, min_dies=L)
        scores["line%d 온전창" % L] = s
        log("L=%d 완료 (가드 점수 0 인 웨이퍼 %d장, 그중 결함 %d장)"
            % (L, int((s == 0).sum()), int(((s == 0) & (is_def == 1)).sum())))
    scores[PRIMARY] = scores.pop("line7 온전창")

    scores["정사각 die k=7 가드없음"] = local_fail_density_max(pad_te, k=7, chunk=CHUNK)
    scores["정사각 die k=7 온전창"] = local_fail_density_max(pad_te, k=7, chunk=CHUNK,
                                                    min_dies=49)
    log("정사각 가드 완료")

    v_tr = local_fail_count_map(pad_tr, k=5, chunk=CHUNK).astype(np.float32)
    ref = fit_band_reference(v_tr, pad_tr, y=y[trn], n_bands=N_BANDS, seed=0)
    del v_tr

    def best_score(x):
        out = np.empty(len(x))
        for a in range(0, len(x), CHUNK):
            xb = x[a:a + CHUNK]
            u = calibrate_map(local_fail_count_map(xb, k=5, chunk=CHUNK),
                              band_index(xb, N_BANDS), ref)
            out[a:a + len(xb)] = u.reshape(len(xb), -1).max(1)
        return out

    scores[BEST] = best_score(pad_te)
    best_tr = best_score(pad_tr)
    log("현재 최선 재현 완료")

    # 융합 — 보정은 train none 만
    best_line = max((k for k in scores if "온전창" in k and k.startswith("line")),
                    key=lambda k: aupr_blocked(scores[k], is_def), default=None)
    cand = [PRIMARY] + ([best_line] if best_line else [])
    for name in dict.fromkeys(cand):
        L = 7 if name == PRIMARY else int(name.replace("line", "").split()[0])
        tr_line = line_density_max(pad_tr, length=L, n_orient=8, chunk=CHUNK, min_dies=L)
        u1 = ecdf_percentile(best_tr, scores[BEST])
        u2 = ecdf_percentile(tr_line, scores[name])
        scores["융합 최선+%s" % name] = fisher([u1, u2])
    log("융합 완료")

    report = {}
    for si, (name, s) in enumerate(scores.items()):
        report[name] = {
            "auroc": auroc(s, is_def), "aupr_blocked": aupr_blocked(s, is_def),
            "aupr_order": aupr(s, is_def), "fpr_at_95tpr": fpr_at_tpr(s, is_def, 0.95),
            "n_unique": int(len(np.unique(s))), "n_zero": int((s == 0).sum()),
            "n_zero_defect": int(((s == 0) & (is_def == 1)).sum()),
            "aupr_blocked_ci95": blocked_ci(s, is_def, 300, 7000 + si),
            "per_class": per_class_auroc(s, yte),
            "per_size_bin": per_size(s, is_def, size_te),
            "operating": operating_points(s, is_def, (0.5, 0.8, 0.95)),
        }
        log("%-26s AUROC %.4f AUPRblk %.4f FPR@95 %.4f 고유값 %d 0점 %d"
            % (name, report[name]["auroc"], report[name]["aupr_blocked"],
               report[name]["fpr_at_95tpr"], report[name]["n_unique"],
               report[name]["n_zero"]))

    order = sorted(report.items(), key=lambda kv: -kv[1]["aupr_blocked"])
    print("\n== 전체 (blocked AUPR 내림차순) ==")
    print("%-26s %8s %10s %20s %9s %8s %8s"
          % ("arm", "AUROC", "AUPR블록", "95% CI", "FPR@95", "고유값", "0점"))
    for name, m in order:
        print("%-26s %8.4f %10.4f  [%.4f, %.4f] %9.4f %8d %8d"
              % (name, m["auroc"], m["aupr_blocked"], m["aupr_blocked_ci95"][0],
                 m["aupr_blocked_ci95"][1], m["fpr_at_95tpr"], m["n_unique"], m["n_zero"]))

    print("\n== 조건 1: 같은 L 에서 가드 있음 대 없음 (blocked 짝지은) ==")
    wins = 0
    for L in LS:
        a = PRIMARY if L == 7 else "line%d 온전창" % L
        b = "line%d 가드없음" % L
        lo, hi, p = paired_aupr_blocked_diff_ci(scores[a], scores[b], is_def, 300, L)
        won = lo > 0 and p < 0.01
        wins += int(won)
        print("  L=%2d  가드 %.4f vs 없음 %.4f  [%+.4f, %+.4f] p=%.4f  %s"
              % (L, report[a]["aupr_blocked"], report[b]["aupr_blocked"], lo, hi, p,
                 "이김" if won else ("짐" if hi < 0 else "무승부")))
    print("  → 가드가 %d/%d 에서 이겼다" % (wins, len(LS)))
    lo, hi, p = paired_aupr_blocked_diff_ci(
        scores["정사각 die k=7 온전창"], scores["정사각 die k=7 가드없음"], is_def, 300, 42)
    print("  정사각 k=7: [%+.4f, %+.4f] p=%.4f  %s"
          % (lo, hi, p, "이김" if lo > 0 else ("짐" if hi < 0 else "무승부")))

    print("\n== 조건 3, 4: 현재 최선과의 비교 ==")
    for name in [n for n in scores if n != BEST]:
        if "line" not in name and "융합" not in name:
            continue
        lo, hi, p = paired_aupr_blocked_diff_ci(scores[name], scores[BEST], is_def, 300, 5)
        print("  %-30s [%+.4f, %+.4f] p=%.4f  %s"
              % (name, lo, hi, p, "이김" if lo > 0 else ("짐" if hi < 0 else "무승부")))

    print("\n== 클래스별 AUROC ==")
    cls = NAMES[1:]
    print("%-26s " % "arm" + " ".join("%10s" % c for c in cls))
    for name, m in order:
        print("%-26s " % name + " ".join("%10.4f" % m["per_class"][c]["auroc"] for c in cls))

    print("\n== Scratch CI (조건 2 판정) ==")
    for name, m in order:
        r = m["per_class"]["Scratch"]
        print("  %-26s Scratch %.4f [%.4f, %.4f]"
              % (name, r["auroc"], r["auroc_ci"][0], r["auroc_ci"][1]))

    print("\n== 운영 지점 ==")
    for name, m in order[:8]:
        cells = ["%0.3f/%6d" % (r["precision"], r["false_positives"]) for r in m["operating"]]
        print("  %-26s 50%% %s   80%% %s   95%% %s" % (name, cells[0], cells[1], cells[2]))

    print("\n== dieSize 구간별 blocked AUPR ==")
    print("%-26s " % "arm" + " ".join("%11s" % b for b in BIN_LBL))
    for name, m in order[:8]:
        print("%-26s " % name + " ".join(
            "%11.4f" % m["per_size_bin"][b] if b in m["per_size_bin"] else "%11s" % "-"
            for b in BIN_LBL))

    (OUT / "full_window_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    np.savez_compressed(OUT / "full_window_scores.npz",
                        **{k: v.astype(np.float64) for k, v in scores.items()},
                        y_test=yte, size_test=size_te)
    log("저장 완료 → %s" % OUT)
