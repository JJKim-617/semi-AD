"""seed 앙상블 — 백분위를 평균해 초기화 잡음을 지운다.

반증 조건은 `candidate/ood_seed_ensemble.md` 에 실행 전에 박았다.
**seed 집합 {0,1,2} 는 17차에서 결과를 보기 전에 정한 것이고 바꾸지 않는다.**
`test_dev` 만 쓴다. **봉인은 조건을 전부 통과할 때만 연다 — 이 스크립트는 안 연다.**

## 왜 점수가 아니라 백분위를 평균하나

seed 마다 kNN 거리의 척도가 다르다(17차 실측: 평균 0.0021~0.0884).
**점수를 그냥 평균하면 척도가 큰 seed 가 지배한다.**
백분위는 각 seed 의 **train-none 경험분포**에서 나오므로 seed 마다 [0,1] 로 정규화된다.
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
from a51_fuse4_eval import EPOCHS, blocked_ci, per_class, per_size  # noqa: E402

OUT = Path("result/ood/o2_ensemble")
CHUNK, N_BANDS, L_LINE = 4000, 32, 11
SEEDS = (0, 1, 2)             # 17차에서 결과 보기 전에 정한 집합. 바꾸지 않는다.
T0 = time.time()


def log(m):
    print("[%7.1fs] %s" % (time.time() - T0, m), flush=True)


def full(score, yy, size, si):
    isd = (yy != 0).astype(np.int64)
    return {"auroc": auroc(score, isd), "aupr_blocked": aupr_blocked(score, isd),
            "aupr_order": aupr(score, isd), "fpr_at_95tpr": fpr_at_tpr(score, isd, 0.95),
            "n_unique": int(len(np.unique(score))),
            "aupr_blocked_ci95": blocked_ci(score, isd, 300, 4000 + si),
            "per_class": per_class(score, yy), "per_size_bin": per_size(score, isd, size),
            "operating": operating_points(score, isd, (0.5, 0.8, 0.95))}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
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

    us = []
    for seed in SEEDS:
        t1 = time.time()
        torch.manual_seed(seed)
        enc = M.PatchEncoder()
        train_encoder(enc, X, trn, seed, EPOCHS, "cpu")
        enc.eval()
        bank, _, _ = gather_bank(enc, X, trn, M.BANK_SIZE, seed, "cpu")
        s_tr, _ = score_partition(enc, X, trn, bank, "cpu")
        s_dev, _ = score_partition(enc, X, dev, bank, "cpu")
        u = ecdf_percentile(s_tr, s_dev)
        us.append(u)
        scores["fuse4 s%d (참고)" % seed] = fisher(u3 + [u])
        log("seed %d 완료 (%.0fs)  fuse4 %.4f"
            % (seed, time.time() - t1, aupr_blocked(scores["fuse4 s%d (참고)" % seed], isd)))

    u_ens = np.mean(us, axis=0)          # **백분위의 평균.** 점수의 평균이 아니다(§3).
    scores["O2 앙상블 단독 (참고)"] = u_ens
    scores["fuse5 앙상블 (주)"] = fisher(u3 + [u_ens])

    rep = {"seeds": list(SEEDS), "epochs": EPOCHS, "threads": a.threads, "metrics": {}}
    for si, (k, s) in enumerate(scores.items()):
        rep["metrics"][k] = full(s, y_dev, size_dev, si)

    print("\n%-22s %8s %10s %22s %9s %9s"
          % ("arm", "AUROC", "AUPR블록", "95% CI", "FPR@95", "고유값"))
    for k, m in rep["metrics"].items():
        print("%-22s %8.4f %10.4f  [%.4f, %.4f] %9.4f %9d"
              % (k, m["auroc"], m["aupr_blocked"], m["aupr_blocked_ci95"][0],
                 m["aupr_blocked_ci95"][1], m["fpr_at_95tpr"], m["n_unique"]))

    P, F = "fuse5 앙상블 (주)", "fuse3 (대조)"
    lo, hi, pv = paired_aupr_blocked_diff_ci(scores[P], scores[F], isd, 300, 71)
    gain = rep["metrics"][P]["aupr_blocked"] - rep["metrics"][F]["aupr_blocked"]
    rep["paired_vs_fuse3"] = {"lo": lo, "hi": hi, "p": pv, "gain": float(gain)}
    print("\n== 반증 조건 1: 짝지은 차이 [%+.4f, %+.4f] p=%.4f  %s =="
          % (lo, hi, pv, "이김" if lo > 0 else ("짐" if hi < 0 else "무승부")))
    print("== 반증 조건 2: 이득 %+.4f  대  21차 seed 폭 0.0145  ->  %s =="
          % (gain, "잡음을 넘는다" if gain > 0.0145 else "**잡음 안 — 실질 채택 후보 아님**"))

    print("\n== 반증 조건 3: 클래스별 ==")
    for k, m in rep["metrics"].items():
        print("  %-22s " % k + "  ".join(
            "%s %.4f [%.4f,%.4f]" % (c, m["per_class"][c]["auroc"],
                                     m["per_class"][c]["auroc_ci"][0],
                                     m["per_class"][c]["auroc_ci"][1])
            for c in ("Center", "Scratch", "Loc", "Edge-Ring")))

    print("\n== 반증 조건 5: dieSize 구간별 ==")
    print("%-22s " % "arm" + " ".join("%11s" % b for b in BIN_LBL))
    for k, m in rep["metrics"].items():
        print("%-22s " % k + " ".join(
            "%11.4f" % m["per_size_bin"][b] if b in m["per_size_bin"] else "%11s" % "-"
            for b in BIN_LBL))

    print("\n== 운영 지점 ==")
    for k, m in rep["metrics"].items():
        c = ["%0.3f/%6d" % (r["precision"], r["false_positives"]) for r in m["operating"]]
        print("  %-22s 50%% %s  80%% %s  95%% %s" % (k, c[0], c[1], c[2]))

    (OUT / "ensemble_metrics.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    np.savez_compressed(OUT / "ensemble_scores.npz", y_dev=y_dev, size_dev=size_dev,
                        u_ens=u_ens,
                        **{k.replace(" ", "_"): v.astype(np.float64) for k, v in scores.items()})
    log("저장 완료 → %s" % OUT)
