"""창 분모 축 평가 — `die`(창 안 다이 개수) 대 `k2`(창 넓이).

반증 조건은 `candidate/ood_window_normalization.md` 에 실행 전에 박았다.
**주 지표는 `aupr_blocked`** (동점 블록, 입력 순서 무관). 순서 의존 AUPR 은 병기만 한다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import (  # noqa: E402
    aupr,
    aupr_blocked,
    auroc,
    fpr_at_tpr,
    operating_points,
)
from a22_ood_template import assert_one_class  # noqa: E402
from a23_ood_template_eval import (  # noqa: E402
    BIN_LBL,
    BINS,
    NAMES,
    _fast_auroc,
    bootstrap_auroc_ci,
    paired_aupr_diff_ci,
)
from a24_ood_residual import (  # noqa: E402
    local_fail_count_map,
    local_fail_count_max,
    local_fail_density_map,
    local_fail_density_max,
)
from a29_radial_calibration import (  # noqa: E402
    band_index,
    calibrate_map,
    fit_band_reference,
)

OUT = Path("result/ood/o1_windownorm")
KS = (3, 5, 7, 9)
CHUNK = 4000
N_BANDS = 32
CHAMP = "챔피언 반경보정 die k=7"
PRIMARY = "주 k2 k=5"
T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def blocked_ci(score, label, n_boot=400, seed=0):
    """blocked AUPR 의 부트스트랩 구간. 재표본마다 블록 구조가 바뀌므로 그대로 다시 잰다."""
    score = np.asarray(score, np.float64)
    label = np.asarray(label, np.int64)
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
        rows[lbl] = {"n": int(m.sum()), "aupr_blocked": aupr_blocked(score[m], is_def[m])}
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

    scores = {}
    for k in KS:
        scores["die k=%d" % k] = local_fail_density_max(pad_te, k=k, chunk=CHUNK)
        scores["k2 k=%d" % k] = local_fail_count_max(pad_te, k=k, chunk=CHUNK)
    scores[PRIMARY] = scores.pop("k2 k=5")
    log("분모 축 8 arm 완료")

    # 챔피언 (4차 사이클) — 같은 코드 경로로 다시 만든다
    v_tr = local_fail_density_map(pad_tr, k=7, chunk=CHUNK).astype(np.float32)
    ref = fit_band_reference(v_tr, pad_tr, y=y[trn], n_bands=N_BANDS, seed=0)
    del v_tr

    def cal_score(x, mapper, k):
        out = np.empty(len(x))
        for a in range(0, len(x), CHUNK):
            xb = x[a:a + CHUNK]
            u = calibrate_map(mapper(xb, k=k, chunk=CHUNK), band_index(xb, N_BANDS), ref)
            out[a:a + len(xb)] = u.reshape(len(xb), -1).max(1)
        return out

    scores[CHAMP] = cal_score(pad_te, local_fail_density_map, 7)
    log("챔피언 재현 완료")

    # 직교성 검사 — k2 맵을 같은 반경 대역 보정에 태운다 (참조도 k2 로 다시 적합)
    v2_tr = local_fail_count_map(pad_tr, k=5, chunk=CHUNK).astype(np.float32)
    ref2 = fit_band_reference(v2_tr, pad_tr, y=y[trn], n_bands=N_BANDS, seed=0)
    del v2_tr
    ref_backup, ref = ref, ref2
    scores["합성 k2 k=5 + 반경보정"] = cal_score(pad_te, local_fail_count_map, 5)
    ref = ref_backup
    log("합성 arm 완료")

    report = {}
    for si, (name, s) in enumerate(scores.items()):
        ab = aupr_blocked(s, is_def)
        report[name] = {
            "auroc": auroc(s, is_def), "aupr_blocked": ab, "aupr_order": aupr(s, is_def),
            "fpr_at_95tpr": fpr_at_tpr(s, is_def, 0.95),
            "n_unique": int(len(np.unique(s))),
            "aupr_blocked_ci95": blocked_ci(s, is_def, 400, 6000 + si),
            "per_class": per_class_auroc(s, yte),
            "per_size_bin": per_size(s, is_def, size_te),
            "operating": operating_points(s, is_def, (0.5, 0.8, 0.95)),
        }
        log("%-24s AUROC %.4f AUPRblk %.4f FPR@95 %.4f 고유값 %d"
            % (name, report[name]["auroc"], ab, report[name]["fpr_at_95tpr"],
               report[name]["n_unique"]))

    order = sorted(report.items(), key=lambda kv: -kv[1]["aupr_blocked"])
    print("\n== 전체 (blocked AUPR 내림차순) ==")
    print("%-24s %8s %10s %20s %9s %9s %8s"
          % ("arm", "AUROC", "AUPR블록", "AUPR블록 95% CI", "AUPR순서", "FPR@95", "고유값"))
    for name, m in order:
        print("%-24s %8.4f %10.4f  [%.4f, %.4f] %9.4f %9.4f %8d"
              % (name, m["auroc"], m["aupr_blocked"], m["aupr_blocked_ci95"][0],
                 m["aupr_blocked_ci95"][1], m["aupr_order"], m["fpr_at_95tpr"],
                 m["n_unique"]))

    print("\n== 조건 1: 같은 k 에서 k2 대 die (짝지은 AUPR 차이) ==")
    wins = 0
    paired = {}
    for k in KS:
        a = PRIMARY if k == 5 else "k2 k=%d" % k
        b = "die k=%d" % k
        lo, hi, p = paired_aupr_diff_ci(scores[a], scores[b], is_def, n_boot=800, seed=k)
        paired["k=%d" % k] = {"ci": [lo, hi], "p": p}
        won = lo > 0 and p < 0.01
        wins += int(won)
        print("  k=%d  k2 %.4f vs die %.4f  [%+.4f, %+.4f] p=%.4f  %s"
              % (k, report[a]["aupr_blocked"], report[b]["aupr_blocked"], lo, hi, p,
                 "이김" if won else ("짐" if hi < 0 else "무승부")))
    print("  → k2 가 %d/%d 에서 이겼다 (과반 필요)" % (wins, len(KS)))

    print("\n== 조건 2, 7: 챔피언 및 합성과의 비교 ==")
    for a, b in ((PRIMARY, CHAMP), ("합성 k2 k=5 + 반경보정", PRIMARY),
                 ("합성 k2 k=5 + 반경보정", CHAMP)):
        lo, hi, p = paired_aupr_diff_ci(scores[a], scores[b], is_def, n_boot=800, seed=7)
        print("  %-34s [%+.4f, %+.4f] p=%.4f  %s"
              % (a + " vs " + b, lo, hi, p,
                 "이김" if lo > 0 else ("짐" if hi < 0 else "무승부")))

    print("\n== 클래스별 AUROC ==")
    cls = NAMES[1:]
    print("%-24s " % "arm" + " ".join("%10s" % c for c in cls))
    for name, m in order:
        print("%-24s " % name + " ".join("%10.4f" % m["per_class"][c]["auroc"] for c in cls))

    print("\n== Scratch / Center / Loc CI ==")
    for name, m in order:
        parts = ["%s %.4f [%.4f,%.4f]" % (c, m["per_class"][c]["auroc"],
                                          m["per_class"][c]["auroc_ci"][0],
                                          m["per_class"][c]["auroc_ci"][1])
                 for c in ("Scratch", "Center", "Loc")]
        print("%-24s %s" % (name, "  ".join(parts)))

    print("\n== 운영 지점 (precision / 헛경보) ==")
    print("%-24s %22s %22s %22s" % ("arm", "재현율 50%", "재현율 80%", "재현율 95%"))
    for name, m in order:
        cells = ["%0.3f / %6d (동점 %5d)" % (r["precision"], r["false_positives"],
                                           r["n_tied_at_threshold"]) for r in m["operating"]]
        print("%-24s %22s %22s %22s" % (name, cells[0], cells[1], cells[2]))

    print("\n== dieSize 구간별 blocked AUPR ==")
    print("%-24s " % "arm" + " ".join("%11s" % b for b in BIN_LBL))
    for name, m in order:
        print("%-24s " % name + " ".join(
            "%11.4f" % m["per_size_bin"][b]["aupr_blocked"] if b in m["per_size_bin"]
            else "%11s" % "-" for b in BIN_LBL))

    (OUT / "window_norm_metrics.json").write_text(
        json.dumps({"report": report, "paired_same_k": paired, "k2_wins": wins},
                   ensure_ascii=False, indent=2))
    np.savez_compressed(OUT / "window_norm_scores.npz",
                        **{k: v.astype(np.float64) for k, v in scores.items()},
                        y_test=yte, size_test=size_te)
    log("저장 완료 → %s" % OUT)
