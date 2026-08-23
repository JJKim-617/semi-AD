"""요소 가중 평가. 반증 조건은 `candidate/ood_component_weighting.md` 에 실행 전에 박았다.

**가중은 train-none 에서만 적합한다.** 평가는 `test_dev` 에서만.
`test_sealed` 는 열지 않는다.

주 지표 `aupr_blocked`. 결정론적이라 seed 면제, 대신 부트스트랩 CI.
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
    BIN_LBL, BINS, NAMES, _fast_auroc, bootstrap_auroc_ci, ecdf_percentile, fisher,
    paired_aupr_blocked_diff_ci,
)
from a24_ood_residual import local_fail_count_map, local_fail_count_max  # noqa: E402
from a27_line_filter import line_density_max  # noqa: E402
from a29_radial_calibration import band_index, calibrate_map, fit_band_reference  # noqa: E402
from a38_sealed_holdout import assert_not_sealed, load_partition  # noqa: E402
from a44_component_weighting import fit_component_weights, probit, weighted_score  # noqa: E402

OUT = Path("result/ood/o1_weighting")
CHUNK, N_BANDS, L_LINE = 4000, 32, 11
KEYS = ("A", "B", "C")
T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


def blocked_ci(s, l, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    v = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(l), len(l))
        v[b] = aupr_blocked(s[i], l[i])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def per_class(score, yy, ci_for=("Scratch", "Center", "Loc", "Edge-Ring")):
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
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    size = d["die_size"].astype(np.float64)
    X = d["X"]
    trn = sp["train"][y[sp["train"]] == 0]
    assert_one_class(y[trn])
    assert_not_sealed(trn, "train-none")
    dev = load_partition("test_dev")
    assert_not_sealed(dev, "test_dev")
    pad_tr = np.ascontiguousarray(X[trn])
    pad_dev = np.ascontiguousarray(X[dev])
    del X, d
    y_dev, size_dev = y[dev], size[dev]
    isd = (y_dev != 0).astype(np.int64)
    log("train-none %d / test_dev %d (결함 %d)" % (len(trn), len(dev), int(isd.sum())))

    v_tr = local_fail_count_map(pad_tr, k=5, chunk=CHUNK).astype(np.float32)
    ref = fit_band_reference(v_tr, pad_tr, y=y[trn], n_bands=N_BANDS, seed=0)
    del v_tr

    def a_sc(x):
        o = np.empty(len(x))
        for i in range(0, len(x), CHUNK):
            xb = x[i:i + CHUNK]
            u = calibrate_map(local_fail_count_map(xb, k=5, chunk=CHUNK),
                              band_index(xb, N_BANDS), ref)
            o[i:i + len(xb)] = u.reshape(len(xb), -1).max(1)
        return o

    def comps(x):
        return {"A": a_sc(x),
                "B": line_density_max(x, length=L_LINE, n_orient=8, chunk=CHUNK,
                                      min_dies=L_LINE),
                "C": local_fail_count_max(x, k=3, chunk=CHUNK)}

    c_tr, c_dev = comps(pad_tr), comps(pad_dev)
    log("요소 완료")

    # 백분위 -> probit. **참조는 train-none 하나.**
    u_tr = {k: ecdf_percentile(c_tr[k], c_tr[k]) for k in KEYS}
    u_dev = {k: ecdf_percentile(c_tr[k], c_dev[k]) for k in KEYS}
    z_tr = np.column_stack([probit(u_tr[k]) for k in KEYS])
    z_dev = np.column_stack([probit(u_dev[k]) for k in KEYS])

    fit = fit_component_weights(z_tr, y=y[trn])       # 라벨 가드 포함
    w = fit["weights"]
    log("train-none 상관행렬:")
    for i, k in enumerate(KEYS):
        log("   %s  " % k + "  ".join("%6.3f" % fit["corr"][i, j] for j in range(3)))
    log("가중 w ∝ S^-1 1 : " + "  ".join("%s %+.4f" % (k, w[i]) for i, k in enumerate(KEYS)))

    scores = {
        "채택 fuse3 (동일 가중 Fisher)": fisher([u_dev[k] for k in KEYS]),
        "W0 대조 (동일 가중 probit 합)": weighted_score(z_dev, np.full(3, 1 / 3)),
        "W1 주 (S^-1 가중 probit 합)": weighted_score(z_dev, w),
        "(탐색) W2 Mahalanobis 한쪽 꼬리":
            np.einsum("ij,jk,ik->i", z_dev, np.linalg.pinv(fit["cov"]), z_dev)
            * np.sign(weighted_score(z_dev, w)),
    }
    for k in KEYS:
        scores["요소 %s 단독" % k] = c_dev[k]

    rep = {"corr": fit["corr"].tolist(), "cov": fit["cov"].tolist(),
           "weights": {k: float(w[i]) for i, k in enumerate(KEYS)},
           "n_train_none": fit["n"], "metrics": {}}
    for si, (name, s) in enumerate(scores.items()):
        rep["metrics"][name] = {
            "auroc": auroc(s, isd), "aupr_blocked": aupr_blocked(s, isd),
            "aupr_order": aupr(s, isd), "fpr_at_95tpr": fpr_at_tpr(s, isd, 0.95),
            "n_unique": int(len(np.unique(s))),
            "aupr_blocked_ci95": blocked_ci(s, isd, 300, 7000 + si),
            "per_class": per_class(s, y_dev), "per_size_bin": per_size(s, isd, size_dev),
            "operating": operating_points(s, isd, (0.5, 0.8, 0.95)),
        }

    order = sorted(rep["metrics"].items(), key=lambda kv: -kv[1]["aupr_blocked"])
    print("\n%-34s %8s %10s %22s %9s %8s"
          % ("arm", "AUROC", "AUPR블록", "95% CI", "FPR@95", "고유값"))
    for n, m in order:
        print("%-34s %8.4f %10.4f  [%.4f, %.4f] %9.4f %8d"
              % (n, m["auroc"], m["aupr_blocked"], m["aupr_blocked_ci95"][0],
                 m["aupr_blocked_ci95"][1], m["fpr_at_95tpr"], m["n_unique"]))

    print("\n== 반증 조건 1, 2: 짝지은 blocked 비교 ==")
    rep["paired"] = {}
    P, W0, F3 = "W1 주 (S^-1 가중 probit 합)", "W0 대조 (동일 가중 probit 합)", "채택 fuse3 (동일 가중 Fisher)"
    for a, b in ((P, F3), (P, W0), (W0, F3)):
        lo, hi, pv = paired_aupr_blocked_diff_ci(scores[a], scores[b], isd, 300, 41)
        rep["paired"]["%s vs %s" % (a, b)] = {"lo": lo, "hi": hi, "p": pv}
        print("  %-70s [%+.4f, %+.4f] p=%.4f  %s"
              % (a + " vs " + b, lo, hi, pv,
                 "이김" if lo > 0 else ("짐" if hi < 0 else "무승부")))

    print("\n== 반증 조건 3: 클래스별 ==")
    cls = NAMES[1:]
    print("%-34s " % "arm" + " ".join("%10s" % c for c in cls))
    for n, m in order:
        print("%-34s " % n + " ".join("%10.4f" % m["per_class"][c]["auroc"] for c in cls))
    print("\n  (CI 있는 넷)")
    for n, m in order:
        print("  %-34s " % n + "  ".join(
            "%s %.4f [%.4f,%.4f]" % (c, m["per_class"][c]["auroc"],
                                     m["per_class"][c]["auroc_ci"][0],
                                     m["per_class"][c]["auroc_ci"][1])
            for c in ("Center", "Scratch", "Loc", "Edge-Ring")))

    print("\n== 운영 지점 ==")
    for n, m in order:
        c = ["%0.3f/%6d" % (r["precision"], r["false_positives"]) for r in m["operating"]]
        print("  %-34s 50%% %s  80%% %s  95%% %s" % (n, c[0], c[1], c[2]))

    print("\n== 반증 조건 5: dieSize 구간별 blocked AUPR ==")
    print("%-34s " % "arm" + " ".join("%11s" % b for b in BIN_LBL))
    for n, m in order:
        print("%-34s " % n + " ".join(
            "%11.4f" % m["per_size_bin"][b] if b in m["per_size_bin"] else "%11s" % "-"
            for b in BIN_LBL))

    (OUT / "weighting_metrics.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    np.savez_compressed(OUT / "weighting_scores.npz", y_dev=y_dev, size_dev=size_dev,
                        **{k: v.astype(np.float64) for k, v in scores.items()})
    log("저장 완료 → %s" % OUT)
