"""방향성 선형 필터 평가. 반증 조건은 `candidate/ood_shape_matched_filters.md` 에 실행 전에 박았다.

**주 arm 은 하나로 미리 고정했다**(`line7_8dir`). 나머지는 탐색용이며 판정에 쓰지 않는다.
여러 변형 중 test 성능으로 최고를 고르면 그건 test 로 고른 것이라 누수다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import evaluate_ood  # noqa: E402
from a23_ood_template_eval import (  # noqa: E402
    BIN_LBL,
    NAMES,
    bootstrap_aupr_ci,
    ecdf_percentile,
    fisher,
    paired_aupr_diff_ci,
    per_class,
    per_size_bin,
)
from a24_ood_residual import local_fail_density_max  # noqa: E402
from a27_line_filter import line_density_max, radius_mask  # noqa: E402

OUT = Path("result/ood/o1_line")
PRIMARY = "주 line7_8dir"
CONTROL = "대조 정사각 k=7"
T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def build_scores(x, tag):
    s = {}
    s["대조 정사각 k=3"] = local_fail_density_max(x, k=3)
    s["대조 정사각 k=7"] = local_fail_density_max(x, k=7)
    log("%s 정사각 대조군 완료" % tag)
    s[PRIMARY] = line_density_max(x, length=7, n_orient=8, width=1)
    log("%s 주 arm 완료" % tag)
    s["탐색 line5_8dir"] = line_density_max(x, length=5, n_orient=8)
    s["탐색 line9_8dir"] = line_density_max(x, length=9, n_orient=8)
    s["탐색 line11_8dir"] = line_density_max(x, length=11, n_orient=8)
    s["탐색 line7_w2"] = line_density_max(x, length=7, n_orient=8, width=2)
    log("%s 선 변형 완료" % tag)
    keep = radius_mask(x, r_min=1.0 / 32.0)
    s["탐색 중심제외 정사각 k=7"] = local_fail_density_max(
        np.where(keep, x, 0).astype(np.uint8), k=7)
    s["탐색 중심제외 line7"] = line_density_max(x, length=7, n_orient=8, valid=keep)
    log("%s 중심 제외 완료" % tag)
    for k in (3, 5, 9):
        if "sq%d" % k not in s:
            s["_sq%d" % k] = local_fail_density_max(x, k=k)
    return s


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    tr, te = sp["train"], sp["test"]
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    size = d["die_size"].astype(np.float64)
    trn = tr[y[tr] == 0]
    X = d["X"]
    pad_tr = np.ascontiguousarray(X[trn])
    pad_te = np.ascontiguousarray(X[te])
    del X, d
    yte = y[te]
    is_def = (yte != 0).astype(np.int64)
    size_te = size[te]
    log("적재 완료. train none %d / test %d" % (len(trn), len(te)))

    s_te = build_scores(pad_te, "test")
    s_tr = build_scores(pad_tr, "train-none")

    # 다중 척도 결합 — 보정은 train none 만 쓴다
    us_te, us_tr = [], []
    for k in (3, 5, 7, 9):
        key = "대조 정사각 k=%d" % k if k in (3, 7) else "_sq%d" % k
        us_te.append(ecdf_percentile(s_tr[key], s_te[key]))
    s_te["탐색 다중척도 k3,5,7,9"] = fisher(us_te)
    log("다중척도 결합 완료")

    scores = {k: v for k, v in s_te.items() if not k.startswith("_")}
    report = {}
    for si, (name, s) in enumerate(scores.items()):
        m = evaluate_ood(s, is_def)
        m["aupr_ci95"] = bootstrap_aupr_ci(s, is_def, n_boot=400, seed=4000 + si)
        m["per_class"] = per_class(s, yte)
        m["per_size_bin"] = per_size_bin(s, is_def, size_te)
        report[name] = m
        log("%-26s AUROC %.4f AUPR %.4f FPR@95 %.4f"
            % (name, m["auroc"], m["aupr"], m["fpr_at_95tpr"]))

    print("\n== 전체 (AUPR 내림차순) ==")
    print("%-26s %8s %8s %20s %10s" % ("arm", "AUROC", "AUPR", "AUPR 95% CI", "FPR@95TPR"))
    for name, m in sorted(report.items(), key=lambda kv: -kv[1]["aupr"]):
        print("%-26s %8.4f %8.4f  [%.4f, %.4f] %10.4f"
              % (name, m["auroc"], m["aupr"], m["aupr_ci95"][0], m["aupr_ci95"][1],
                 m["fpr_at_95tpr"]))

    print("\n== 대조 정사각 k=7 대비 짝지은 AUPR 차이 (1000회) ==")
    paired = {}
    for name in scores:
        if name == CONTROL:
            continue
        lo, hi, p = paired_aupr_diff_ci(scores[name], scores[CONTROL], is_def,
                                        n_boot=1000, seed=77)
        paired[name] = {"ci": [lo, hi], "p": p}
        print("  %-26s [%+.4f, %+.4f] p=%.4f  %s"
              % (name, lo, hi, p, "이김" if lo > 0 else ("짐" if hi < 0 else "구분 안 됨")))

    print("\n== 클래스별 AUROC ==")
    cls = NAMES[1:]
    print("%-26s " % "arm" + " ".join("%10s" % c for c in cls))
    for name, m in sorted(report.items(), key=lambda kv: -kv[1]["aupr"]):
        print("%-26s " % name + " ".join(
            "%10.4f" % m["per_class"][c]["auroc"] if c in m["per_class"] else "%10s" % "-"
            for c in cls))

    print("\n== Scratch / Center / Loc 부트스트랩 CI ==")
    for name, m in report.items():
        parts = []
        for c in ("Scratch", "Center", "Loc"):
            r = m["per_class"].get(c, {})
            if "auroc_ci" in r:
                parts.append("%s %.4f [%.4f,%.4f]"
                             % (c, r["auroc"], r["auroc_ci"][0], r["auroc_ci"][1]))
        print("%-26s %s" % (name, "  ".join(parts)))

    print("\n== dieSize 구간별 AUPR ==")
    print("%-26s " % "arm" + " ".join("%11s" % b for b in BIN_LBL))
    for name, m in sorted(report.items(), key=lambda kv: -kv[1]["aupr"]):
        print("%-26s " % name + " ".join(
            "%11.4f" % m["per_size_bin"][b]["aupr"] if b in m["per_size_bin"]
            else "%11s" % "-" for b in BIN_LBL))

    (OUT / "line_filter_metrics.json").write_text(
        json.dumps({"report": report, "paired_vs_square7": paired, "primary": PRIMARY},
                   ensure_ascii=False, indent=2))
    np.savez_compressed(OUT / "line_filter_scores.npz",
                        **{k: v.astype(np.float32) for k, v in scores.items()},
                        y_test=yte, size_test=size_te)
    log("저장 완료 → %s" % OUT)
