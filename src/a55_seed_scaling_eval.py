"""seed 16개 — 세 사이클 연속 부딪힌 벽이 측정 정밀도인지 확인한다.

반증 조건과 판정 규칙은 `candidate/ood_seed_scaling.md` §5 에 **실행 전에** 박았다.
**seed 수 16 은 §3 에서 비용을 재고 고정했다. 결과를 보고 바꾸지 않는다.**

## 두 종류의 불확실성

- **U-eval** — 평가 집합. 웨이퍼 복원 재표본, 두 arm 에 같은 재표본.
- **U-seed** — seed 뽑기. **16개 seed 를 복원 재표본해 앙상블을 다시 만든다.**
  21, 22차는 이걸 못 재서 **단일 seed 의 폭**을 대용으로 썼는데,
  **폭은 N 이 늘수록 커지므로 N 이 다른 비교에 못 쓴다**(§5.2).

## CPU 고정

**GPU 0번이 비어도 안 쓴다.** 19차에서 학습이 스레드 수·장치에 의존하는 것을 실측했고,
장치를 바꾸면 seed 0~2 가 22차 값과 달라져 **비교가 깨진다.** 비교 가능성 문제다.

**seed 마다 결과를 즉시 저장한다** — 중간에 끊겨도 끝난 수만큼은 남고,
그때는 §3 대로 **끝난 수를 적고 판정을 유보한다.**
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

from a21_ood_metrics import (  # noqa: E402
    aupr_blocked, auroc, fpr_at_tpr, operating_points, paired_fpr_at_tpr_diff_ci,
)
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

OUT = Path("result/ood/o2_seed_scaling")
CHUNK, N_BANDS, L_LINE, EPOCHS = 4000, 32, 11, 3
N_SEEDS = 16                  # §3 에서 비용을 재고 고정. 결과 보고 안 바꾼다.
N_BOOT = 300
T0 = time.time()


def log(m):
    print("[%7.1fs] %s" % (time.time() - T0, m), flush=True)


def per_class(score, yy, ci_for=("Scratch", "Center", "Loc")):
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
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--threads", type=int, default=28)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
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
    log("train-none %d / test_dev %d (결함 %d) — seed %d개 예정, CPU %d 스레드"
        % (len(trn), len(dev), int(isd.sum()), a.seeds, a.threads))

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
    fuse3 = fisher(u3)
    base_aupr = aupr_blocked(fuse3, isd)
    base_fpr = fpr_at_tpr(fuse3, isd, 0.95)
    log("fuse3 준비  blocked AUPR %.4f  FPR@95 %.4f" % (base_aupr, base_fpr))

    us, single = [], []
    for s in range(a.seeds):
        t1 = time.time()
        torch.manual_seed(s)
        enc = M.PatchEncoder()
        train_encoder(enc, X, trn, s, EPOCHS, "cpu")
        enc.eval()
        bank, _, _ = gather_bank(enc, X, trn, M.BANK_SIZE, s, "cpu")
        s_tr, _ = score_partition(enc, X, trn, bank, "cpu")
        s_dev, _ = score_partition(enc, X, dev, bank, "cpu")
        u = ecdf_percentile(s_tr, s_dev)
        us.append(u)
        f4 = fisher(u3 + [u])
        single.append({"seed": s, "fuse4_aupr": aupr_blocked(f4, isd),
                       "fuse4_fpr95": fpr_at_tpr(f4, isd, 0.95),
                       "o2_alone_aupr": aupr_blocked(s_dev, isd),
                       "seconds": round(time.time() - t1, 1)})
        # **매 seed 마다 저장한다** — 끊겨도 끝난 수만큼은 남는다(§3)
        np.savez_compressed(OUT / "seed_u.npz", u=np.array(us, np.float32),
                            y_dev=y_dev, size_dev=size_dev, fuse3=fuse3)
        (OUT / "single_seeds.json").write_text(json.dumps(single, ensure_ascii=False, indent=2))
        log("seed %2d 완료 (%.0fs)  fuse4 %.4f  FPR@95 %.4f"
            % (s, time.time() - t1, single[-1]["fuse4_aupr"], single[-1]["fuse4_fpr95"]))

    n = len(us)
    U = np.array(us)
    log("=== seed %d개 완료. 집계 시작 ===" % n)

    # §7 잡음이 실제로 줄어드는가 — 앞에서부터 k 개 앙상블
    curve = []
    for k in range(1, n + 1):
        f5k = fisher(u3 + [U[:k].mean(0)])
        curve.append({"k": k, "aupr_blocked": aupr_blocked(f5k, isd),
                      "fpr_at_95tpr": fpr_at_tpr(f5k, isd, 0.95)})
    fuse5 = fisher(u3 + [U.mean(0)])

    sa = np.array([r["fuse4_aupr"] for r in single])
    sf = np.array([r["fuse4_fpr95"] for r in single])
    rep = {"n_seeds": n, "pinned_seeds": a.seeds, "threads": a.threads,
           "device": "cpu", "epochs": EPOCHS,
           "fuse3": {"aupr_blocked": base_aupr, "fpr_at_95tpr": base_fpr},
           "single_seeds": single, "curve": curve,
           "single_seed_spread": {
               "aupr_sd": float(sa.std(ddof=1)), "aupr_range": float(sa.max() - sa.min()),
               "fpr_sd": float(sf.std(ddof=1)), "fpr_range": float(sf.max() - sf.min()),
               "aupr_sem": float(sa.std(ddof=1) / np.sqrt(n))}}

    # --- U-eval: 평가 집합 재표본 -------------------------------------------------
    lo, hi, pv = paired_aupr_blocked_diff_ci(fuse5, fuse3, isd, N_BOOT, 101)
    rep["u_eval_aupr"] = {"lo": lo, "hi": hi, "p": pv,
                          "gain": float(aupr_blocked(fuse5, isd) - base_aupr)}
    flo, fhi, fp = paired_fpr_at_tpr_diff_ci(fuse5, fuse3, isd, 0.95, N_BOOT, 102)
    rep["u_eval_fpr"] = {"lo": flo, "hi": fhi, "p": fp}

    # --- U-seed: seed 재표본 ------------------------------------------------------
    rng = np.random.default_rng(2026)
    gains = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, n, n)
        gains[b] = aupr_blocked(fisher(u3 + [U[pick].mean(0)]), isd) - base_aupr
    rep["u_seed_aupr"] = {"lo": float(np.percentile(gains, 2.5)),
                          "hi": float(np.percentile(gains, 97.5)),
                          "mean": float(gains.mean())}

    m5 = {"aupr_blocked": aupr_blocked(fuse5, isd), "auroc": auroc(fuse5, isd),
          "fpr_at_95tpr": fpr_at_tpr(fuse5, isd, 0.95),
          "n_unique": int(len(np.unique(fuse5))),
          "per_class": per_class(fuse5, y_dev),
          "per_size_bin": per_size(fuse5, isd, size_dev),
          "operating": operating_points(fuse5, isd, (0.5, 0.8, 0.95))}
    rep["fuse5_n"] = m5
    rep["fuse3_per_class"] = per_class(fuse3, y_dev)
    rep["fuse3_per_size_bin"] = per_size(fuse3, isd, size_dev)

    print("\n== seed 별 fuse4 ==")
    print("%5s %12s %12s %10s" % ("seed", "AUPR블록", "FPR@95", "초"))
    for r in single:
        print("%5d %12.4f %12.4f %10.0f"
              % (r["seed"], r["fuse4_aupr"], r["fuse4_fpr95"], r["seconds"]))
    sp_ = rep["single_seed_spread"]
    print("  표준편차 %.4f | 폭 %.4f | 평균의 표준오차 %.4f"
          % (sp_["aupr_sd"], sp_["aupr_range"], sp_["aupr_sem"]))

    print("\n== §7 앙상블 크기 곡선 ==")
    print("%5s %12s %12s" % ("k", "AUPR블록", "FPR@95"))
    for c in curve:
        print("%5d %12.4f %12.4f" % (c["k"], c["aupr_blocked"], c["fpr_at_95tpr"]))

    print("\n== 판정 ==")
    print("  fuse3            AUPR블록 %.4f  FPR@95 %.4f" % (base_aupr, base_fpr))
    print("  fuse5_%-2d         AUPR블록 %.4f  FPR@95 %.4f  고유값 %d"
          % (n, m5["aupr_blocked"], m5["fpr_at_95tpr"], m5["n_unique"]))
    print("  조건 1 U-eval  [%+.5f, %+.5f] p=%.4f  -> %s"
          % (lo, hi, pv, "통과 (하한 > 0)" if lo > 0 else "**미충족**"))
    r2 = rep["u_seed_aupr"]
    print("  조건 2 U-seed  [%+.5f, %+.5f]        -> %s"
          % (r2["lo"], r2["hi"], "통과 (하한 > 0)" if r2["lo"] > 0 else "**미충족**"))
    print("  (병기) FPR@95 U-eval [%+.5f, %+.5f] p=%.4f" % (flo, fhi, fp))
    print("  (참고) 옛 규칙 '이득 > seed 폭': 이득 %.5f 대 폭 %.5f -> %s"
          % (rep["u_eval_aupr"]["gain"], sp_["aupr_range"],
             "넘음" if rep["u_eval_aupr"]["gain"] > sp_["aupr_range"] else "못 넘음"))

    print("\n== 조건 3 클래스별 ==")
    for c in ("Center", "Scratch", "Loc"):
        b = rep["fuse3_per_class"][c]
        v = m5["per_class"][c]
        bad = v["auroc_ci"][1] < b["auroc_ci"][0]
        print("  %-9s fuse3 %.4f [%.4f,%.4f] -> fuse5 %.4f [%.4f,%.4f] %s"
              % (c, b["auroc"], *b["auroc_ci"], v["auroc"], *v["auroc_ci"],
                 "**하락 — 기각**" if bad else ""))

    print("\n== 조건 5 dieSize 구간별 ==")
    print("%-10s " % "arm" + " ".join("%11s" % b for b in BIN_LBL))
    for nm, mm in (("fuse3", rep["fuse3_per_size_bin"]), ("fuse5_%d" % n, m5["per_size_bin"])):
        print("%-10s " % nm + " ".join(
            "%11.4f" % mm[b] if b in mm else "%11s" % "-" for b in BIN_LBL))

    print("\n== 운영 지점 (fuse5_%d) ==" % n)
    print("  " + "  ".join("%.0f%% %0.3f/%6d" % (r["target_recall"] * 100, r["precision"],
                                                 r["false_positives"]) for r in m5["operating"]))

    (OUT / "seed_scaling.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log("저장 완료 → %s  (봉인은 열지 않았다 — §5.4)" % OUT)
