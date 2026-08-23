"""fuse4 — O2(epoch 3) 를 세 요소에 더한다. `candidate/ood_o2_fuse4.md` 참조.

**`epoch=3` 은 `test_dev` 성능으로 골랐다**(20차 사이클). 그건 튜닝이고,
그래서 봉인이 있다. 반증 조건을 다 통과해야만 `test_sealed` 를 한 번 연다.
이 스크립트는 **봉인을 열지 않는다.**
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from _runtime import setup  # noqa: E402

setup("feat")
import torch  # noqa: E402

from a21_ood_metrics import aupr, aupr_blocked, auroc, fpr_at_tpr, operating_points  # noqa: E402
from a23_ood_template_eval import (  # noqa: E402
    BIN_LBL, BINS, NAMES, _fast_auroc, bootstrap_auroc_ci, ecdf_percentile, fisher,
    paired_aupr_blocked_diff_ci,
)
from a24_ood_residual import local_fail_count_map, local_fail_count_max  # noqa: E402
from a27_line_filter import line_density_max  # noqa: E402
from a29_radial_calibration import band_index, calibrate_map, fit_band_reference  # noqa: E402
from a38_sealed_holdout import assert_not_sealed, load_partition  # noqa: E402
import a42_ood_ssl_knn as M  # noqa: E402
from a43_ood_ssl_eval import gather_bank, score_partition, select_train_none, train_encoder  # noqa: E402

OUT = Path("result/ood/o2_fuse4")
CHUNK, N_BANDS, L_LINE = 4000, 32, 11
EPOCHS = 3                     # 20차 사이클이 test_dev 에서 찾은 봉우리. 튜닝임을 문서에 적었다.
T0 = time.time()


def log(m):
    print("[%7.1fs] %s" % (time.time() - T0, m), flush=True)


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


def full(score, yy, size, si):
    isd = (yy != 0).astype(np.int64)
    return {"auroc": auroc(score, isd), "aupr_blocked": aupr_blocked(score, isd),
            "aupr_order": aupr(score, isd), "fpr_at_95tpr": fpr_at_tpr(score, isd, 0.95),
            "n_unique": int(len(np.unique(score))),
            "aupr_blocked_ci95": blocked_ci(score, isd, 300, 5000 + si),
            "per_class": per_class(score, yy), "per_size_bin": per_size(score, isd, size),
            "operating": operating_points(score, isd, (0.5, 0.8, 0.95))}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--threads", type=int, default=28)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    seeds = [int(s) for s in a.seeds.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    sp = np.load("data/wm811k/cache/splits_v1.npz")
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    size = d["die_size"].astype(np.float64)
    X = d["X"]
    trn = select_train_none(y, sp["train"])
    assert_not_sealed(trn, "train-none")
    dev = load_partition("test_dev")
    assert_not_sealed(dev, "test_dev")
    pad_tr = np.ascontiguousarray(X[trn])
    pad_dev = np.ascontiguousarray(X[dev])
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
    u3 = [ecdf_percentile(c_tr[k], c_dev[k]) for k in ("A", "B", "C")]
    scores = {"fuse3 (대조)": fisher(u3)}
    log("fuse3 준비  blocked AUPR %.4f" % aupr_blocked(scores["fuse3 (대조)"], isd))

    meta = {}
    for seed in seeds:
        t1 = time.time()
        torch.manual_seed(seed)
        enc = M.PatchEncoder()
        hist = train_encoder(enc, X, trn, seed, EPOCHS, "cpu")
        enc.eval()
        bank, cnt, tot = gather_bank(enc, X, trn, M.BANK_SIZE, seed, "cpu")
        s_tr, fb1 = score_partition(enc, X, trn, bank, "cpu")
        s_dev, fb2 = score_partition(enc, X, dev, bank, "cpu")
        u_o2 = ecdf_percentile(s_tr, s_dev)
        scores["O2 단독 s%d" % seed] = s_dev
        scores["fuse4 s%d" % seed] = fisher(u3 + [u_o2])
        meta["s%d" % seed] = {"epochs": EPOCHS, "final_masked_ce": hist[-1]["masked_ce"],
                              "fallback_train": fb1, "fallback_dev": fb2,
                              "threads": a.threads, "seconds": round(time.time() - t1, 1)}
        log("seed %d 완료 (%.0fs)  O2 단독 %.4f  fuse4 %.4f"
            % (seed, time.time() - t1, aupr_blocked(s_dev, isd),
               aupr_blocked(scores["fuse4 s%d" % seed], isd)))

    rep = {"meta": meta, "epochs": EPOCHS, "metrics": {}}
    for si, (k, s) in enumerate(scores.items()):
        rep["metrics"][k] = full(s, y_dev, size_dev, si)

    print("\n%-18s %8s %10s %22s %9s %9s"
          % ("arm", "AUROC", "AUPR블록", "95% CI", "FPR@95", "고유값"))
    for k, m in rep["metrics"].items():
        print("%-18s %8.4f %10.4f  [%.4f, %.4f] %9.4f %9d"
              % (k, m["auroc"], m["aupr_blocked"], m["aupr_blocked_ci95"][0],
                 m["aupr_blocked_ci95"][1], m["fpr_at_95tpr"], m["n_unique"]))

    print("\n== 반증 조건 1: fuse3 대비 짝지은 blocked 차이 (세 seed 전부 이겨야 한다) ==")
    rep["paired"] = {}
    for seed in seeds:
        k = "fuse4 s%d" % seed
        lo, hi, pv = paired_aupr_blocked_diff_ci(scores[k], scores["fuse3 (대조)"],
                                                 isd, 300, 61 + seed)
        rep["paired"][k] = {"lo": lo, "hi": hi, "p": pv}
        print("  %-18s [%+.4f, %+.4f] p=%.4f  %s"
              % (k, lo, hi, pv, "이김" if lo > 0 else ("짐" if hi < 0 else "무승부")))
    vals = [rep["metrics"]["fuse4 s%d" % s]["aupr_blocked"] for s in seeds]
    rep["fuse4_seed_range"] = float(max(vals) - min(vals))
    rep["fuse4_minus_fuse3_mean"] = float(np.mean(vals) - rep["metrics"]["fuse3 (대조)"]["aupr_blocked"])
    print("\n== 반증 조건 6: seed 폭 %.4f  대  평균 이득 %.4f  -> %s =="
          % (rep["fuse4_seed_range"], rep["fuse4_minus_fuse3_mean"],
             "이득이 더 크다" if rep["fuse4_minus_fuse3_mean"] > rep["fuse4_seed_range"]
             else "**seed 폭이 더 크다 — 무승부로 적는다**"))

    print("\n== 반증 조건 2: 클래스별 ==")
    for k, m in rep["metrics"].items():
        print("  %-18s " % k + "  ".join(
            "%s %.4f [%.4f,%.4f]" % (c, m["per_class"][c]["auroc"],
                                     m["per_class"][c]["auroc_ci"][0],
                                     m["per_class"][c]["auroc_ci"][1])
            for c in ("Center", "Scratch", "Loc", "Edge-Ring")))

    print("\n== 반증 조건 4: dieSize 구간별 ==")
    print("%-18s " % "arm" + " ".join("%11s" % b for b in BIN_LBL))
    for k, m in rep["metrics"].items():
        print("%-18s " % k + " ".join(
            "%11.4f" % m["per_size_bin"][b] if b in m["per_size_bin"] else "%11s" % "-"
            for b in BIN_LBL))

    print("\n== 운영 지점 ==")
    for k, m in rep["metrics"].items():
        c = ["%0.3f/%6d" % (r["precision"], r["false_positives"]) for r in m["operating"]]
        print("  %-18s 50%% %s  80%% %s  95%% %s" % (k, c[0], c[1], c[2]))

    (OUT / "fuse4_metrics.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    np.savez_compressed(OUT / "fuse4_scores.npz", y_dev=y_dev, size_dev=size_dev,
                        **{k.replace(" ", "_"): v.astype(np.float64) for k, v in scores.items()})
    log("저장 완료 → %s" % OUT)
