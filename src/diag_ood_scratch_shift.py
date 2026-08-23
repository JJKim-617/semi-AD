"""Scratch 경고등 — `val_unseen` 에서만 AUROC 0.7835 (n=92) 인 이유를 가른다.

반증 조건과 판정 규칙은 `docs/experiments/candidate/ood_scratch_shift.md` 에
**실행 전에** 박았다. 여기서는 그 §2~§5 를 그대로 잰다.

## 봉인 규약

`val_unseen` 을 **이 프로세스에서 정확히 한 번** 연다(사전등록 §6-1).
그래서 스크립트를 나누지 않고 전부 여기서 끝낸다. `test_sealed` 는 열지 않는다.
**이 사이클은 val 로 어떤 arm 도 고르거나 기각하거나 튜닝하지 않는다.**

## 왜 2x2 인가

클래스별 AUROC 은 양성(그 클래스)과 음성(그 파티션의 정상) 둘 다에 의존한다.
val 은 둘이 **동시에** 바뀌었으므로 한 번 재서는 원인을 못 가른다.
점수 함수는 양쪽에서 같다 — 백분위 참조와 반경 대역 참조가 **train-none 하나**뿐이다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import subsample_auroc_null  # noqa: E402
from a22_ood_template import assert_one_class  # noqa: E402
from a23_ood_template_eval import BINS, NAMES, _fast_auroc, ecdf_percentile, fisher  # noqa: E402
from a24_ood_residual import local_fail_count_map, local_fail_count_max  # noqa: E402
from a27_line_filter import line_density_max  # noqa: E402
from a29_radial_calibration import band_index, calibrate_map, fit_band_reference  # noqa: E402
from a38_sealed_holdout import assert_not_sealed, load_partition  # noqa: E402
from a41_shape_stats import mannwhitney_u, shape_stats  # noqa: E402

OUT = Path("result/ood/o1_scratch_shift")
EVID = Path("docs/research/ood_scratch_shift/evidence")
CHUNK, N_BANDS, L_LINE = 4000, 32, 11
N_REP = 2000
REASON = ("사전등록 candidate/ood_scratch_shift.md §6 — Scratch 경고등 원인 진단 1회. "
          "arm 선택·기각·튜닝에 쓰지 않는다")
STATS = ("die_size", "n_fail", "fail_ratio", "cc_size", "neighbors", "elongation", "major")
BONF = 6                      # 사전등록 §7-S4 가 정한 검정 개수(die_size 는 문맥용)
T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


def cross_auroc(pos, neg) -> float:
    """양성과 음성을 **다른 파티션에서** 가져와 재는 AUROC."""
    s = np.concatenate([np.asarray(neg, np.float64), np.asarray(pos, np.float64)])
    l = np.concatenate([np.zeros(len(neg), np.int64), np.ones(len(pos), np.int64)])
    return _fast_auroc(s, l)


def size_bin(v):
    return np.searchsorted(np.array(BINS[1:-1], np.float64), np.asarray(v, np.float64),
                           side="right")


def size_matched_null(pos_dev, size_dev, size_val, neg_dev, n, n_rep=N_REP, seed=0):
    """**dev Scratch 를 val Scratch 의 크기 구성으로 다시 뽑은** 귀무분포.

    "val 이 낮은 것은 웨이퍼가 커서다" 를 직접 검정한다. 크기 구간별 비율을 val 에 맞춰
    dev 에서 n 장을 뽑는다. 어떤 구간이 dev 에 모자라면 그 구간은 있는 만큼만 쓰고
    부족분을 전체에서 채운다(그 사실을 같이 돌려준다).
    """
    bd, bv = size_bin(size_dev), size_bin(size_val)
    nb = len(BINS) - 1
    quota = np.array([int(round((bv == k).sum() / len(bv) * n)) for k in range(nb)])
    quota[int(np.argmax(quota))] += n - quota.sum()
    pools = {k: np.flatnonzero(bd == k) for k in range(nb)}
    short = {int(k): int(quota[k] - len(pools[k])) for k in range(nb)
             if quota[k] > len(pools[k])}
    rng = np.random.default_rng(seed)
    lab = np.concatenate([np.zeros(len(neg_dev), np.int64), np.ones(n, np.int64)])
    vals = np.empty(n_rep)
    allp = np.arange(len(pos_dev))
    for i in range(n_rep):
        take = []
        deficit = 0
        for k in range(nb):
            q = int(quota[k])
            if q <= 0:
                continue
            if q <= len(pools[k]):
                take.append(rng.choice(pools[k], q, replace=False))
            else:
                take.append(pools[k])
                deficit += q - len(pools[k])
        idx = np.concatenate(take) if take else np.array([], int)
        if deficit:
            rest = np.setdiff1d(allp, idx)
            idx = np.concatenate([idx, rng.choice(rest, deficit, replace=False)])
        vals[i] = _fast_auroc(np.concatenate([neg_dev, pos_dev[idx]]), lab)
    return {"quota": quota.tolist(), "short": short, "mean": float(vals.mean()),
            "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    EVID.mkdir(parents=True, exist_ok=True)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    size = d["die_size"].astype(np.float64)
    X = d["X"]

    trn = sp["train"][y[sp["train"]] == 0]
    assert_one_class(y[trn])
    assert_not_sealed(trn, "train-none")
    pad_tr = np.ascontiguousarray(X[trn])

    dev = load_partition("test_dev")
    val = load_partition("val_unseen", unseal=True, reason=REASON)   # <-- 딱 한 번
    pad_dev = np.ascontiguousarray(X[dev])
    pad_val = np.ascontiguousarray(X[val])
    del X, d
    y_dev, y_val = y[dev], y[val]
    log("적재: train-none %d, dev %d, val %d" % (len(trn), len(dev), len(val)))

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

    def comps(x):
        return {"A": a_score(x),
                "B": line_density_max(x, length=L_LINE, n_orient=8, chunk=CHUNK,
                                      min_dies=L_LINE),
                "C": local_fail_count_max(x, k=3, chunk=CHUNK)}

    c_tr = comps(pad_tr)
    log("train-none 요소 완료")
    c_dev = comps(pad_dev)
    log("dev 요소 완료")
    c_val = comps(pad_val)
    log("val 요소 완료")

    def arms(c):
        u = {k: ecdf_percentile(c_tr[k], c[k]) for k in ("A", "B", "C")}
        return {"fuse3": fisher([u["A"], u["B"], u["C"]]),
                "A_k2k5cal": c["A"], "B_lineL11": c["B"], "C_k2k3": c["C"]}

    s_dev, s_val = arms(c_dev), arms(c_val)
    report = {"reason": REASON, "n": {"dev": len(dev), "val": len(val)}}

    # --- §2  2x2 교차 분해 (전 클래스) ------------------------------------------
    log("=== 2x2 교차 분해 ===")
    cross = {}
    for arm in ("fuse3", "A_k2k5cal", "B_lineL11", "C_k2k3"):
        nd, nv = s_dev[arm][y_dev == 0], s_val[arm][y_val == 0]
        cross[arm] = {}
        for c in range(1, 9):
            pd_, pv = s_dev[arm][y_dev == c], s_val[arm][y_val == c]
            if len(pd_) == 0 or len(pv) == 0:
                continue
            cross[arm][NAMES[c]] = {
                "n_dev": int(len(pd_)), "n_val": int(len(pv)),
                "devpos_devneg": cross_auroc(pd_, nd),
                "devpos_valneg": cross_auroc(pd_, nv),
                "valpos_devneg": cross_auroc(pv, nd),
                "valpos_valneg": cross_auroc(pv, nv),
            }
    report["cross"] = cross
    print("\n%-11s %-9s %5s %5s | %8s %8s %8s %8s"
          % ("arm", "클래스", "n_dev", "n_val", "dev/dev", "dev/val", "val/dev", "val/val"))
    for arm in cross:
        for cname, r in cross[arm].items():
            print("%-11s %-9s %5d %5d | %8.4f %8.4f %8.4f %8.4f"
                  % (arm, cname, r["n_dev"], r["n_val"], r["devpos_devneg"],
                     r["devpos_valneg"], r["valpos_devneg"], r["valpos_valneg"]))

    # --- §3  표본 잡음 귀무 -------------------------------------------------------
    log("=== 표본 잡음 귀무 (dev 양성을 val 의 n 으로 부표집) ===")
    nulls = {}
    for arm in ("fuse3", "B_lineL11"):
        nd = s_dev[arm][y_dev == 0]
        nulls[arm] = {}
        for c in range(1, 9):
            pd_ = s_dev[arm][y_dev == c]
            nv = int((y_val == c).sum())
            if nv == 0 or nv > len(pd_):
                continue
            obs = cross[arm][NAMES[c]]["valpos_devneg"]
            r = subsample_auroc_null(pd_, nd, n=nv, n_rep=N_REP, seed=c, observed=obs)
            r.pop("values")
            r["observed_valpos_valneg"] = cross[arm][NAMES[c]]["valpos_valneg"]
            nulls[arm][NAMES[c]] = r
    report["subsample_null"] = nulls
    print("\n%-11s %-9s %5s %8s %22s %9s %9s"
          % ("arm", "클래스", "n", "귀무평균", "귀무 95% 구간", "val/dev", "분위"))
    for arm in nulls:
        for cname, r in nulls[arm].items():
            print("%-11s %-9s %5d %8.4f  [%.4f, %.4f]      %9.4f %9.4f"
                  % (arm, cname, r["n"], r["mean"], r["ci95"][0], r["ci95"][1],
                     r["observed"], r["frac_below_observed"]))

    # --- 크기 맞춘 귀무 (Scratch 만) -----------------------------------------------
    log("=== 크기 맞춘 귀무 (Scratch) ===")
    smn = {}
    for arm in ("fuse3", "B_lineL11"):
        pd_ = s_dev[arm][y_dev == 7]
        smn[arm] = size_matched_null(pd_, size[dev][y_dev == 7], size[val][y_val == 7],
                                     s_dev[arm][y_dev == 0], n=int((y_val == 7).sum()),
                                     seed=7)
        r = smn[arm]
        print("  %-11s 크기맞춤 귀무 평균 %.4f  [%.4f, %.4f]  quota %s  부족 %s"
              % (arm, r["mean"], r["ci95"][0], r["ci95"][1], r["quota"], r["short"]))
    report["size_matched_null"] = smn

    # --- §5  형태 통계 -------------------------------------------------------------
    log("=== 형태 통계 (Scratch: dev vs val) ===")
    sd = shape_stats(pad_dev[y_dev == 7], size[dev][y_dev == 7])
    sv = shape_stats(pad_val[y_val == 7], size[val][y_val == 7])
    shape = {}
    print("\n%-11s %10s %10s %10s %10s %10s %-8s"
          % ("통계", "dev중앙", "val중앙", "dev평균", "val평균", "p", "판정"))
    for k in STATS:
        u, p = mannwhitney_u(sd[k], sv[k])
        sig = (p < 0.05 / BONF) if k != "die_size" else (p < 0.05)
        shape[k] = {"dev_median": float(np.nanmedian(sd[k])),
                    "val_median": float(np.nanmedian(sv[k])),
                    "dev_mean": float(np.nanmean(sd[k])),
                    "val_mean": float(np.nanmean(sv[k])),
                    "u": u, "p": p, "significant_bonferroni": bool(sig)}
        print("%-11s %10.4f %10.4f %10.4f %10.4f %10.2e %-8s"
              % (k, shape[k]["dev_median"], shape[k]["val_median"],
                 shape[k]["dev_mean"], shape[k]["val_mean"], p,
                 "다르다" if sig else "차이없음"))
    report["shape_scratch"] = shape

    # 대조: 정상끼리도 같은 통계가 다른가 (형태 차이가 Scratch 특유인지 보려면 필요하다)
    log("=== 대조: 정상끼리 같은 통계 ===")
    rs = np.random.default_rng(0)
    nd_i = rs.choice(np.flatnonzero(y_dev == 0), 3000, replace=False)
    nv_i = rs.choice(np.flatnonzero(y_val == 0), 3000, replace=False)
    snd = shape_stats(pad_dev[nd_i], size[dev][nd_i])
    snv = shape_stats(pad_val[nv_i], size[val][nv_i])
    ctrl = {}
    for k in STATS:
        u, p = mannwhitney_u(snd[k], snv[k])
        ctrl[k] = {"dev_median": float(np.nanmedian(snd[k])),
                   "val_median": float(np.nanmedian(snv[k])), "p": p}
        print("  %-11s dev %10.4f  val %10.4f  p %.2e" % (k, ctrl[k]["dev_median"],
                                                          ctrl[k]["val_median"], p))
    report["shape_none_control"] = ctrl

    (OUT / "scratch_shift.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (EVID / "scratch_shift.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    log("저장 완료 → %s" % OUT)
