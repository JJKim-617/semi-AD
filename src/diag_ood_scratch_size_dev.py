"""Scratch 가 **큰 웨이퍼에서** 어려운가 — `test_dev` 안에서만 확인한다.

## 왜 이걸 하는가

12차 사이클이 판정 규칙 S3 을 발동시켰다: `val_unseen` 의 Scratch 는 **모양이 다르다.**
가장 큰 차이가 크기였다 — **die_size 중앙값 dev 904 대 val 2393**, 그리고
**val Scratch 92장 중 66장(72%)이 `>1600` 구간인데 dev Scratch 517장 중 6장뿐**이다.
그래서 크기 맞춘 귀무를 만들려 해도 60장이 모자라 만들 수 없었다.

**이 스크립트는 탐색이다.** 사전등록 §5 에 없다 — 결과를 보고 생긴 질문이다.
그래서 여기서 나오는 것으로 arm 을 고치거나 채택 표를 바꾸지 않는다.
**val 을 다시 열지 않는다.** `test_dev` 안에서 크기 의존성이 보이는지만 본다.

`docs/reports/OOD.md` 는 `>1600` 의 낮은 수치를 **"약점이 아니라 표본 부족"** 이라 적었다.
그 문장이 맞는지 다른 각도에서 본다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import aupr_blocked  # noqa: E402
from a22_ood_template import assert_one_class  # noqa: E402
from a23_ood_template_eval import BIN_LBL, BINS, NAMES, _fast_auroc, ecdf_percentile, fisher  # noqa: E402
from a24_ood_residual import local_fail_count_map, local_fail_count_max  # noqa: E402
from a27_line_filter import line_density_max  # noqa: E402
from a29_radial_calibration import band_index, calibrate_map, fit_band_reference  # noqa: E402
from a38_sealed_holdout import assert_not_sealed, load_partition  # noqa: E402
from a41_shape_stats import shape_stats  # noqa: E402

OUT = Path("result/ood/o1_scratch_shift")
EVID = Path("docs/research/ood_scratch_shift/evidence")
CHUNK, N_BANDS, L_LINE = 4000, 32, 11
T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


def spearman(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ok = ~(np.isnan(a) | np.isnan(b))
    ra = np.argsort(np.argsort(a[ok])).astype(np.float64)
    rb = np.argsort(np.argsort(b[ok])).astype(np.float64)
    if len(ra) < 3:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


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
    assert_not_sealed(dev, "test_dev")
    pad_dev = np.ascontiguousarray(X[dev])
    del X, d
    y_dev, size_dev = y[dev], size[dev]
    log("적재: train-none %d, dev %d" % (len(trn), len(dev)))

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

    c_tr, c_dev = comps(pad_tr), comps(pad_dev)
    log("요소 완료")
    u = {k: ecdf_percentile(c_tr[k], c_dev[k]) for k in ("A", "B", "C")}
    fuse = fisher([u["A"], u["B"], u["C"]])
    arms = {"fuse3": fuse, "B_lineL11": c_dev["B"], "A_k2k5cal": c_dev["A"],
            "C_k2k3": c_dev["C"]}

    rep = {"n_dev": int(len(dev))}
    bidx = np.searchsorted(np.array(BINS[1:-1], np.float64), size_dev, side="right")

    # --- 1. dev 안에서 크기 구간별 클래스 AUROC (음성은 그 구간의 정상만) -------------
    log("=== dev 크기 구간별 AUROC (음성 = 같은 구간의 dev 정상) ===")
    print("%-11s %-10s " % ("arm", "클래스") + " ".join("%13s" % b for b in BIN_LBL))
    bysize = {}
    for arm in ("fuse3", "B_lineL11"):
        bysize[arm] = {}
        for c in (7, 1, 5, 4):                      # Scratch, Center, Loc, Edge-Ring
            row, cells = {}, []
            for k, lbl in enumerate(BIN_LBL):
                m = bidx == k
                pos, neg = arms[arm][m & (y_dev == c)], arms[arm][m & (y_dev == 0)]
                if len(pos) < 5 or len(neg) < 50:
                    cells.append("%13s" % ("n=%d" % len(pos)))
                    row[lbl] = {"n_pos": int(len(pos)), "auroc": None}
                    continue
                a = _fast_auroc(np.concatenate([neg, pos]),
                                np.concatenate([np.zeros(len(neg), int),
                                                np.ones(len(pos), int)]))
                cells.append("%9.4f(%3d)" % (a, len(pos)))
                row[lbl] = {"n_pos": int(len(pos)), "auroc": float(a)}
            bysize[arm][NAMES[c]] = row
            print("%-11s %-10s " % (arm, NAMES[c]) + " ".join(cells))
    rep["by_size_bin"] = bysize

    # --- 2. Scratch 안에서 점수 순위와 형태의 상관 -----------------------------------
    log("=== dev Scratch 안: 점수 순위 대 형태 (Spearman) ===")
    m = y_dev == 7
    st = shape_stats(pad_dev[m], size_dev[m])
    corr = {}
    print("%-11s " % "arm" + " ".join("%12s" % k for k in st))
    for arm in ("fuse3", "B_lineL11", "A_k2k5cal", "C_k2k3"):
        s = arms[arm][m]
        corr[arm] = {k: spearman(s, st[k]) for k in st}
        print("%-11s " % arm + " ".join("%12.4f" % corr[arm][k] for k in st))
    rep["scratch_shape_corr"] = corr
    print("  (음수 = 그 통계가 클수록 이상 점수가 낮다 = 놓치기 쉽다)")

    # --- 3. 큰 웨이퍼 전반: 결함 전체의 크기 구간별 blocked AUPR ----------------------
    log("=== dev 크기 구간별 blocked AUPR (결함 전체) ===")
    isd = (y_dev != 0).astype(np.int64)
    persize = {}
    for arm in ("fuse3", "B_lineL11"):
        cells = []
        persize[arm] = {}
        for k, lbl in enumerate(BIN_LBL):
            mm = bidx == k
            if mm.sum() and isd[mm].sum() not in (0, mm.sum()):
                v = aupr_blocked(arms[arm][mm], isd[mm])
                persize[arm][lbl] = {"aupr_blocked": float(v),
                                     "n": int(mm.sum()), "n_def": int(isd[mm].sum())}
                cells.append("%8.4f(%4d)" % (v, isd[mm].sum()))
            else:
                cells.append("%14s" % "-")
        print("%-11s " % arm + " ".join(cells))
    rep["by_size_aupr"] = persize

    # --- 4. 크기 구간별 Scratch 장수 (dev 가 큰 웨이퍼를 얼마나 못 봤나) --------------
    log("=== dev 클래스 x 크기 구간 장수 ===")
    print("%-10s " % "클래스" + " ".join("%11s" % b for b in BIN_LBL))
    counts = {}
    for c in range(0, 9):
        row = [int(((bidx == k) & (y_dev == c)).sum()) for k in range(len(BIN_LBL))]
        counts[NAMES[c]] = row
        print("%-10s " % NAMES[c] + " ".join("%11d" % v for v in row))
    rep["counts_class_by_size"] = counts

    (OUT / "scratch_size_dev.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    (EVID / "scratch_size_dev.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log("저장 완료")
