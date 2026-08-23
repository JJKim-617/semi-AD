"""크기 구간별 **헛경보율** — 결함 라벨 없이 잴 수 있는 절반.

반증 조건은 `candidate/ood_large_wafer_coverage.md` 에 실행 전에 박았다.

## 왜 이것만 재는가

`>1600` 구간의 낮은 AUPR 은 **놓침 탓인지 헛경보 탓인지** 안 갈렸다(정정 13).
재현율은 그 구간 결함이 `test_dev` 에 29장뿐이라 **못 잰다.**
그러나 **헛경보는 정상만 있으면 잰다** — 그 구간에 정상이 446장 있다.

**문턱을 정할 때만 라벨을 쓰고, 구간별 초과 비율에는 라벨이 안 들어간다.**
`test_sealed` 는 열지 않는다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import operating_points  # noqa: E402
from a22_ood_template import assert_one_class  # noqa: E402
from a23_ood_template_eval import BIN_LBL, BINS, ecdf_percentile, fisher  # noqa: E402
from a24_ood_residual import local_fail_count_map, local_fail_count_max  # noqa: E402
from a27_line_filter import line_density_max  # noqa: E402
from a29_radial_calibration import band_index, calibrate_map, fit_band_reference  # noqa: E402
from a38_sealed_holdout import assert_not_sealed, load_partition  # noqa: E402

OUT = Path("result/ood/o1_size_fa")
EVID = Path("docs/research/ood_scratch_shift/evidence")
CHUNK, N_BANDS, L_LINE = 4000, 32, 11
T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


def wilson(k, n, z=1.96):
    """이항 비율의 Wilson 구간. 446장에서 정규 근사는 꼬리에서 나쁘다."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


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
    dev = load_partition("test_dev")
    assert_not_sealed(dev, "test_dev")
    pad_tr = np.ascontiguousarray(X[trn])
    pad_dev = np.ascontiguousarray(X[dev])
    del X, d
    y_dev, size_dev, size_tr = y[dev], size[dev], size[trn]
    isd = (y_dev != 0).astype(np.int64)
    log("train-none %d / test_dev %d" % (len(trn), len(dev)))

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
    fuse_dev = fisher([ecdf_percentile(c_tr[k], c_dev[k]) for k in ("A", "B", "C")])
    fuse_tr = fisher([ecdf_percentile(c_tr[k], c_tr[k]) for k in ("A", "B", "C")])
    log("요소 완료  dev fuse3 준비")

    ops = operating_points(fuse_dev, isd, (0.5, 0.8, 0.95))
    bd = np.searchsorted(np.array(BINS[1:-1], np.float64), size_dev, side="right")
    bt = np.searchsorted(np.array(BINS[1:-1], np.float64), size_tr, side="right")
    norm = y_dev == 0

    rep = {"thresholds": [{"recall": o["target_recall"], "threshold": o["threshold"]}
                          for o in ops], "bins": {}}
    for o in ops:
        t = o["threshold"]
        print("\n== 재현율 %.0f%% 문턱 (전체 헛경보 %d장) — 크기 구간별 정상의 초과 비율 =="
              % (o["target_recall"] * 100, o["false_positives"]))
        print("%-12s %8s %10s %22s | %8s %10s"
              % ("구간", "dev정상", "초과비율", "Wilson 95% CI", "train n", "train 비율"))
        for k, lbl in enumerate(BIN_LBL):
            m = norm & (bd == k)
            n = int(m.sum())
            if n == 0:
                continue
            kk = int((fuse_dev[m] >= t).sum())
            lo, hi = wilson(kk, n)
            mt = bt == k
            nt = int(mt.sum())
            rt = float((fuse_tr[mt] >= t).mean()) if nt else float("nan")
            print("%-12s %8d %10.4f  [%.4f, %.4f] | %8d %10.4f"
                  % (lbl, n, kk / n, lo, hi, nt, rt))
            rep["bins"].setdefault("%.2f" % o["target_recall"], {})[lbl] = {
                "n_dev_normal": n, "fa_rate": kk / n, "ci95": [lo, hi],
                "n_train_none": nt, "train_none_rate": rt,
                "n_dev_defect": int(((y_dev != 0) & (bd == k)).sum()),
            }

    print("\n== 참고: 구간별 dev 결함 장수 (재현율은 이 자료로 못 잰다) ==")
    for k, lbl in enumerate(BIN_LBL):
        n = int(((y_dev != 0) & (bd == k)).sum())
        if int((bd == k).sum()):
            print("  %-12s 결함 %5d" % (lbl, n))

    (OUT / "size_falsealarm.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    (EVID / "size_falsealarm.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log("저장 완료 → %s" % OUT)
