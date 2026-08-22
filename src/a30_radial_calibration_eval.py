"""반경 대역별 밀도 보정 평가. 반증 조건은 `candidate/ood_radial_calibration.md` 에 실행 전에 박았다.

arm 세 개만 잰다 — 주 arm 하나와 **대조군 둘**.
전역 보정 대조군이 없으면 "보정" 의 효과와 "반경 대역" 의 효과를 못 가른다.
2차 사이클에서 대조군을 빠뜨려 결론이 뒤집힌 적이 있다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import evaluate_ood  # noqa: E402
from a22_ood_template import assert_one_class  # noqa: E402
from a23_ood_template_eval import (  # noqa: E402
    BIN_LBL,
    NAMES,
    bootstrap_aupr_ci,
    paired_aupr_diff_ci,
    per_class,
    per_size_bin,
)
from a24_ood_residual import local_fail_density_map, local_fail_density_max  # noqa: E402
from a29_radial_calibration import (  # noqa: E402
    band_index,
    calibrate_map,
    fit_band_reference,
    fit_global_reference,
)

OUT = Path("result/ood/o1_radialcal")
K = 7
N_BANDS = 32
CHUNK = 4000
PRIMARY = "주 radial_cal_k7 (32대역)"
CTRL_RAW = "대조 보정없음 k=7"
CTRL_GLOBAL = "대조 전역보정 k=7"
T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def score_calibrated(x, ref, banded: bool):
    out = np.empty(len(x), np.float64)
    for a in range(0, len(x), CHUNK):
        xb = x[a:a + CHUNK]
        v = local_fail_density_map(xb, k=K, chunk=CHUNK)
        bands = band_index(xb, N_BANDS) if banded else np.where(xb > 0, 0, -1)
        u = calibrate_map(v, bands, ref)
        out[a:a + len(xb)] = u.reshape(len(xb), -1).max(1)
    return out


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
    log("적재 완료. train none %d / test %d" % (len(trn), len(te)))

    v_tr = local_fail_density_map(pad_tr, k=K, chunk=CHUNK).astype(np.float32)
    log("train-none 밀도 맵 완료")
    ref_band = fit_band_reference(v_tr, pad_tr, y=y[trn], n_bands=N_BANDS, seed=0)
    ref_glob = fit_global_reference(v_tr, pad_tr, y=y[trn], seed=0)
    log("참조 분포 적합 (정상 %d장만). 대역별 표본 수 %s"
        % (len(trn), [len(a) for a in ref_band.values][:6]))
    print("  대역별 정상 밀도 중앙값:",
          " ".join("%.3f" % np.median(a) for a in ref_band.values), flush=True)
    del v_tr

    scores = {}
    scores[CTRL_RAW] = local_fail_density_max(pad_te, k=K, chunk=CHUNK)
    log("대조군(보정 없음) 완료")
    scores[CTRL_GLOBAL] = score_calibrated(pad_te, ref_glob, banded=False)
    log("대조군(전역 보정) 완료")
    scores[PRIMARY] = score_calibrated(pad_te, ref_band, banded=True)
    log("주 arm 완료")

    report = {}
    for si, (name, s) in enumerate(scores.items()):
        m = evaluate_ood(s, is_def)
        m["aupr_ci95"] = bootstrap_aupr_ci(s, is_def, n_boot=600, seed=5000 + si)
        m["per_class"] = per_class(s, yte)
        m["per_size_bin"] = per_size_bin(s, is_def, size_te)
        report[name] = m
        log("%-26s AUROC %.4f AUPR %.4f FPR@95 %.4f"
            % (name, m["auroc"], m["aupr"], m["fpr_at_95tpr"]))

    print("\n== 전체 ==")
    print("%-26s %8s %8s %20s %10s" % ("arm", "AUROC", "AUPR", "AUPR 95% CI", "FPR@95TPR"))
    for name in (CTRL_RAW, CTRL_GLOBAL, PRIMARY):
        m = report[name]
        print("%-26s %8.4f %8.4f  [%.4f, %.4f] %10.4f"
              % (name, m["auroc"], m["aupr"], m["aupr_ci95"][0], m["aupr_ci95"][1],
                 m["fpr_at_95tpr"]))

    print("\n== 짝지은 AUPR 차이 (1000회) ==")
    paired = {}
    for a, b in ((PRIMARY, CTRL_RAW), (PRIMARY, CTRL_GLOBAL), (CTRL_GLOBAL, CTRL_RAW)):
        lo, hi, p = paired_aupr_diff_ci(scores[a], scores[b], is_def, n_boot=1000, seed=99)
        paired["%s vs %s" % (a, b)] = {"ci": [lo, hi], "p": p}
        print("  %-46s [%+.4f, %+.4f] p=%.4f  %s"
              % (a + " vs " + b, lo, hi, p,
                 "이김" if lo > 0 else ("짐" if hi < 0 else "구분 안 됨")))

    print("\n== 클래스별 AUROC ==")
    cls = NAMES[1:]
    print("%-26s " % "arm" + " ".join("%10s" % c for c in cls))
    for name in (CTRL_RAW, CTRL_GLOBAL, PRIMARY):
        m = report[name]
        print("%-26s " % name + " ".join(
            "%10.4f" % m["per_class"][c]["auroc"] if c in m["per_class"] else "%10s" % "-"
            for c in cls))

    print("\n== Scratch / Center / Loc 부트스트랩 CI ==")
    for name in (CTRL_RAW, CTRL_GLOBAL, PRIMARY):
        parts = []
        for c in ("Scratch", "Center", "Loc"):
            r = report[name]["per_class"].get(c, {})
            if "auroc_ci" in r:
                parts.append("%s %.4f [%.4f,%.4f]"
                             % (c, r["auroc"], r["auroc_ci"][0], r["auroc_ci"][1]))
        print("%-26s %s" % (name, "  ".join(parts)))

    print("\n== dieSize 구간별 AUPR ==")
    print("%-26s " % "arm" + " ".join("%11s" % b for b in BIN_LBL))
    for name in (CTRL_RAW, CTRL_GLOBAL, PRIMARY):
        m = report[name]
        print("%-26s " % name + " ".join(
            "%11.4f" % m["per_size_bin"][b]["aupr"] if b in m["per_size_bin"]
            else "%11s" % "-" for b in BIN_LBL))

    (OUT / "radial_calibration_metrics.json").write_text(
        json.dumps({"report": report, "paired": paired, "primary": PRIMARY,
                    "band_median": [float(np.median(a)) for a in ref_band.values]},
                   ensure_ascii=False, indent=2))
    np.savez_compressed(OUT / "radial_calibration_scores.npz",
                        **{k: v.astype(np.float32) for k, v in scores.items()},
                        y_test=yte, size_test=size_te)
    log("저장 완료 → %s" % OUT)
