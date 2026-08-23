"""크기 비례 창 — 창이 다이 단위로는 크기 불변인데 웨이퍼 비율로는 아니다.

반증 조건은 `candidate/ood_size_scaled_window.md` 에 실행 전에 박았다.

## 기작 가설

요소 A, C 는 **다이 개수로 고정된 창**(5x5, 3x3) 안의 불량 비율이다.
그러면 5x5 창이 덮는 **웨이퍼 비율**이 다이 518개 웨이퍼에서는 4.8%,
2,393개 웨이퍼에서는 **1.0%** 다. **큰 웨이퍼에서 창이 상대적으로 너무 작다.**

18차가 실측한 "큰 웨이퍼 정상이 8~84배 더 자주 걸린다" 를 이것으로 설명할 수 있다.
**설명이지 측정이 아니므로 검정한다.**

## arm 셋

- **S1(주)**: 크기 비례 창 + **크기 층별 백분위 보정**
- **S0(대조)**: 크기 비례 창 + 기존 통합 보정
- **fuse3(대조)**: 현행

**S0 이 없으면 창의 기여와 보정의 기여를 못 가른다** — 14차가 잡아낸 실수가 그것이다.

`die_size` 는 정상 웨이퍼에서도 아는 값이라 **범위 A** 다. 결함 정보를 안 쓴다.
`test_dev` 만 쓴다. 결정론적이라 seed 면제.
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

OUT = Path("result/ood/o1_size_window")
CHUNK, N_BANDS, L_LINE = 4000, 32, 11
REF_DIE = 518.0          # train-none 다이 수 중앙값. 실행 전에 정해져 있던 수다.
K_MIN, K_MAX = 3, 15
T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


def scaled_k(die_size, k0: int) -> np.ndarray:
    """`k = round_odd(k0 * sqrt(die_size / 518))`, [3, 15] 로 자른다.

    면적이 다이 수에 비례하므로 **길이는 sqrt** 다. 지수를 맞추면 손잡이가 생기고
    그건 튜닝이라 `sqrt` 하나로 못 박았다(사전등록 §7).
    """
    r = k0 * np.sqrt(np.asarray(die_size, np.float64) / REF_DIE)
    k = 2 * np.round((r - 1) / 2.0) + 1          # 가장 가까운 홀수
    return np.clip(k, K_MIN, K_MAX).astype(np.int64)


def grouped_count_max(x, ks) -> np.ndarray:
    """웨이퍼마다 다른 `k` 로 k^2 밀도 최대값. **같은 `k` 끼리 묶어 한 번에 센다.**"""
    out = np.empty(len(x), np.float64)
    for k in np.unique(ks):
        idx = np.flatnonzero(ks == k)
        for a in range(0, len(idx), CHUNK):
            b = idx[a:a + CHUNK]
            out[b] = local_fail_count_max(np.ascontiguousarray(x[b]), k=int(k), chunk=CHUNK)
    return out


def grouped_maps(x, ks):
    """웨이퍼마다 다른 `k` 로 밀도 **맵**. 반경 보정에 필요하다."""
    out = np.empty(x.shape, np.float32)
    for k in np.unique(ks):
        idx = np.flatnonzero(ks == k)
        for a in range(0, len(idx), CHUNK):
            b = idx[a:a + CHUNK]
            out[b] = local_fail_count_map(np.ascontiguousarray(x[b]), k=int(k),
                                          chunk=CHUNK).astype(np.float32)
    return out


def size_stratum(die_size) -> np.ndarray:
    return np.searchsorted(np.array(BINS[1:-1], np.float64),
                           np.asarray(die_size, np.float64), side="right")


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
    y_dev, size_dev, size_tr = y[dev], size[dev], size[trn]
    isd = (y_dev != 0).astype(np.int64)
    log("train-none %d / test_dev %d (결함 %d)" % (len(trn), len(dev), int(isd.sum())))

    kA_tr, kA_dev = scaled_k(size_tr, 5), scaled_k(size_dev, 5)
    kC_tr, kC_dev = scaled_k(size_tr, 3), scaled_k(size_dev, 3)
    log("요소 A 의 창 크기 분포 (train-none): " + ", ".join(
        "k=%d %d장" % (k, int((kA_tr == k).sum())) for k in np.unique(kA_tr)))
    log("요소 A 의 창 크기 분포 (test_dev):   " + ", ".join(
        "k=%d %d장" % (k, int((kA_dev == k).sum())) for k in np.unique(kA_dev)))

    # --- 현행 fuse3 ----------------------------------------------------------------
    v_tr = local_fail_count_map(pad_tr, k=5, chunk=CHUNK).astype(np.float32)
    ref0 = fit_band_reference(v_tr, pad_tr, y=y[trn], n_bands=N_BANDS, seed=0)
    del v_tr

    def a_fixed(x):
        o = np.empty(len(x))
        for i in range(0, len(x), CHUNK):
            xb = x[i:i + CHUNK]
            u = calibrate_map(local_fail_count_map(xb, k=5, chunk=CHUNK),
                              band_index(xb, N_BANDS), ref0)
            o[i:i + len(xb)] = u.reshape(len(xb), -1).max(1)
        return o

    B_tr = line_density_max(pad_tr, length=L_LINE, n_orient=8, chunk=CHUNK, min_dies=L_LINE)
    B_dev = line_density_max(pad_dev, length=L_LINE, n_orient=8, chunk=CHUNK, min_dies=L_LINE)
    A0_tr, A0_dev = a_fixed(pad_tr), a_fixed(pad_dev)
    C0_tr = local_fail_count_max(pad_tr, k=3, chunk=CHUNK)
    C0_dev = local_fail_count_max(pad_dev, k=3, chunk=CHUNK)
    fuse3 = fisher([ecdf_percentile(A0_tr, A0_dev), ecdf_percentile(B_tr, B_dev),
                    ecdf_percentile(C0_tr, C0_dev)])
    log("현행 fuse3 blocked AUPR %.4f" % aupr_blocked(fuse3, isd))

    # --- 크기 비례 창 ---------------------------------------------------------------
    mA_tr = grouped_maps(pad_tr, kA_tr)
    refS = fit_band_reference(mA_tr, pad_tr, y=y[trn], n_bands=N_BANDS, seed=0)
    del mA_tr

    def a_scaled(x, ks):
        o = np.empty(len(x))
        for k in np.unique(ks):
            idx = np.flatnonzero(ks == k)
            for a in range(0, len(idx), CHUNK):
                b = idx[a:a + CHUNK]
                xb = np.ascontiguousarray(x[b])
                u = calibrate_map(local_fail_count_map(xb, k=int(k), chunk=CHUNK),
                                  band_index(xb, N_BANDS), refS)
                o[b] = u.reshape(len(b), -1).max(1)
        return o

    AS_tr, AS_dev = a_scaled(pad_tr, kA_tr), a_scaled(pad_dev, kA_dev)
    CS_tr, CS_dev = grouped_count_max(pad_tr, kC_tr), grouped_count_max(pad_dev, kC_dev)
    log("크기 비례 창 완료")

    # S0: 통합 보정
    s0 = fisher([ecdf_percentile(AS_tr, AS_dev), ecdf_percentile(B_tr, B_dev),
                 ecdf_percentile(CS_tr, CS_dev)])

    # S1: 크기 층별 보정 — 층마다 train-none 참조를 따로 만든다
    st_tr, st_dev = size_stratum(size_tr), size_stratum(size_dev)
    us = []
    for comp_tr, comp_dev in ((AS_tr, AS_dev), (B_tr, B_dev), (CS_tr, CS_dev)):
        u = np.empty(len(comp_dev))
        for s in np.unique(st_dev):
            mt, md = st_tr == s, st_dev == s
            ref_vals = comp_tr[mt] if mt.sum() >= 10 else comp_tr   # 너무 적으면 통합으로 되돌린다
            u[md] = ecdf_percentile(ref_vals, comp_dev[md])
        us.append(u)
    s1 = fisher(us)
    log("층별 보정 완료")

    scores = {"fuse3 (현행)": fuse3, "S0 창만 (통합 보정)": s0, "S1 창+층별 보정 (주)": s1}
    rep = {"ref_die": REF_DIE, "metrics": {},
           "k_hist_train": {int(k): int((kA_tr == k).sum()) for k in np.unique(kA_tr)},
           "k_hist_dev": {int(k): int((kA_dev == k).sum()) for k in np.unique(kA_dev)}}
    for si, (k, s) in enumerate(scores.items()):
        rep["metrics"][k] = {
            "auroc": auroc(s, isd), "aupr_blocked": aupr_blocked(s, isd),
            "aupr_order": aupr(s, isd), "fpr_at_95tpr": fpr_at_tpr(s, isd, 0.95),
            "n_unique": int(len(np.unique(s))),
            "aupr_blocked_ci95": blocked_ci(s, isd, 300, 3000 + si),
            "per_class": per_class(s, y_dev), "per_size_bin": per_size(s, isd, size_dev),
            "operating": operating_points(s, isd, (0.5, 0.8, 0.95))}

    print("\n%-24s %8s %10s %22s %9s %9s"
          % ("arm", "AUROC", "AUPR블록", "95% CI", "FPR@95", "고유값"))
    for k, m in rep["metrics"].items():
        print("%-24s %8.4f %10.4f  [%.4f, %.4f] %9.4f %9d"
              % (k, m["auroc"], m["aupr_blocked"], m["aupr_blocked_ci95"][0],
                 m["aupr_blocked_ci95"][1], m["fpr_at_95tpr"], m["n_unique"]))

    print("\n== 반증 조건 1, 3: 짝지은 blocked 차이 ==")
    rep["paired"] = {}
    for a, b in (("S1 창+층별 보정 (주)", "fuse3 (현행)"),
                 ("S0 창만 (통합 보정)", "fuse3 (현행)"),
                 ("S1 창+층별 보정 (주)", "S0 창만 (통합 보정)")):
        lo, hi, pv = paired_aupr_blocked_diff_ci(scores[a], scores[b], isd, 300, 81)
        rep["paired"]["%s vs %s" % (a, b)] = {"lo": lo, "hi": hi, "p": pv}
        print("  %-52s [%+.4f, %+.4f] p=%.4f  %s"
              % (a + " vs " + b, lo, hi, pv,
                 "이김" if lo > 0 else ("짐" if hi < 0 else "무승부")))

    print("\n== 반증 조건 2: dieSize 구간별 (이 arm 의 존재 이유) ==")
    print("%-24s " % "arm" + " ".join("%11s" % b for b in BIN_LBL))
    for k, m in rep["metrics"].items():
        print("%-24s " % k + " ".join(
            "%11.4f" % m["per_size_bin"][b] if b in m["per_size_bin"] else "%11s" % "-"
            for b in BIN_LBL))

    print("\n== 반증 조건 4: `>1600` 정상의 95% 문턱 초과율 ==")
    norm = y_dev == 0
    rep["fa_gt1600"] = {}
    for k, s in scores.items():
        t = rep["metrics"][k]["operating"][2]["threshold"]
        m = norm & (size_dev >= 1600)
        rate = float((s[m] >= t).mean())
        rep["fa_gt1600"][k] = rate
        print("  %-24s %.4f  (정상 %d장)" % (k, rate, int(m.sum())))

    print("\n== 반증 조건 5: 클래스별 ==")
    for k, m in rep["metrics"].items():
        print("  %-24s " % k + "  ".join(
            "%s %.4f [%.4f,%.4f]" % (c, m["per_class"][c]["auroc"],
                                     m["per_class"][c]["auroc_ci"][0],
                                     m["per_class"][c]["auroc_ci"][1])
            for c in ("Center", "Scratch", "Loc", "Edge-Ring")))

    print("\n== 운영 지점 ==")
    for k, m in rep["metrics"].items():
        c = ["%0.3f/%6d" % (r["precision"], r["false_positives"]) for r in m["operating"]]
        print("  %-24s 50%% %s  80%% %s  95%% %s" % (k, c[0], c[1], c[2]))

    (OUT / "size_window_metrics.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log("저장 완료 → %s" % OUT)
