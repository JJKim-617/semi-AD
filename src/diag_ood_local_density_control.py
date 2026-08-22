"""빠진 대조군 — 이웃 평균의 최대는 사실 그냥 **국소 불량 밀도**가 아닌가.

## 왜 이걸 급히 재는가

뭉개기 실험에서 `T-RADIAL nll/smooth5_max` 가 AUPR 0.3856 → 0.6808 로 뛰었다.
믿기 전에 교란을 찾아야 한다(작업 원칙 7).

**대수적으로 확인되는 것**: `pool_smoothed_max` 는 창 안 잔차의 합을 창 안 다이 개수로
나눈다. template 이 균일 상수 p 면 셀 잔차는 불량이면 -log(p), 정상 다이면 -log(1-p) 라
창 평균은 `f * (-log p) + (1-f) * (-log(1-p))`, 즉 **국소 불량 비율 f 의 아핀 함수**다.
아핀 변환은 순위를 안 바꾸므로 **균일 template + smooth_k_max = 국소 불량 밀도의 최대**와
AUROC/AUPR 이 정확히 같다.

그러면 0.6808 은 template 이 한 일이 아니라 **국소 밀도가 한 일**일 수 있다.
`ood_residual_pooling.md` 의 대조군에 이게 빠져 있었다 — 거기 적은 대조군은 `sum` 뭉개기와
E0 스칼라뿐이었다. **뭉개기를 바꾸면 대조군도 같이 바뀌어야 한다는 것을 놓쳤다.**

## 그래서 무엇을 가르는가

- `균일 + smooth_k_max` = 순수 국소 밀도. **template 기여 0.**
- `T-RADIAL + smooth_k_max` = 국소 밀도 + 반경 template 보정.

둘의 짝지은 차이가 곧 **반경 template 의 순수 기여**다.
차이가 없으면 "국소 밀도가 신호였고 template 은 장식" 이라고 적어야 한다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import evaluate_ood, fail_ratio  # noqa: E402
from a22_ood_template import (  # noqa: E402
    assert_one_class,
    fit_radial_template,
    uniform_template,
)
from a23_ood_template_eval import (  # noqa: E402
    BIN_LBL,
    NAMES,
    bootstrap_aupr_ci,
    paired_aupr_diff_ci,
    per_class,
    per_size_bin,
)
from a24_ood_residual import pool_smoothed_max, residual_map  # noqa: E402

OUT = Path("result/ood/o1_density_control")
CHUNK = 2000
KS = (3, 5, 7, 9)
T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def smooth_scores(x, template, ks=KS):
    out = {k: np.empty(len(x), np.float64) for k in ks}
    for a in range(0, len(x), CHUNK):
        xb = x[a:a + CHUNK]
        die = xb > 0
        m = residual_map(xb, template, mode="nll", chunk=CHUNK)
        for k in ks:
            out[k][a:a + len(xb)] = pool_smoothed_max(m, die, k=k)
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
    log("적재 완료")

    t_rad = fit_radial_template(pad_tr, y=y[trn], n_bins=32, smoothing=1.0)
    t_uni = uniform_template(t_rad.global_rate)
    log("반경 template + 균일 대조군 적합 (전역 불량률 %.5f)" % t_rad.global_rate)

    scores = {"E0 불량 다이 비율(바닥)": fail_ratio(pad_te)}
    for k, v in smooth_scores(pad_te, t_uni).items():
        scores["대조 균일/smooth%d_max (=국소 밀도)" % k] = v
    log("균일 대조군 채점 완료")
    for k, v in smooth_scores(pad_te, t_rad).items():
        scores["T-RADIAL/smooth%d_max" % k] = v
    log("반경 template 채점 완료")

    report = {}
    for si, (name, s) in enumerate(scores.items()):
        m = evaluate_ood(s, is_def)
        m["aupr_ci95"] = bootstrap_aupr_ci(s, is_def, n_boot=400, seed=3000 + si)
        m["per_class"] = per_class(s, yte)
        m["per_size_bin"] = per_size_bin(s, is_def, size_te)
        report[name] = m
        log("%-34s AUROC %.4f AUPR %.4f FPR@95 %.4f"
            % (name, m["auroc"], m["aupr"], m["fpr_at_95tpr"]))

    print("\n== 전체 ==")
    print("%-34s %8s %8s %20s %10s" % ("arm", "AUROC", "AUPR", "AUPR 95% CI", "FPR@95TPR"))
    for name, m in sorted(report.items(), key=lambda kv: -kv[1]["aupr"]):
        print("%-34s %8.4f %8.4f  [%.4f, %.4f] %10.4f"
              % (name, m["auroc"], m["aupr"], m["aupr_ci95"][0], m["aupr_ci95"][1],
                 m["fpr_at_95tpr"]))

    print("\n== 반경 template 의 순수 기여 (같은 창 크기에서 짝지은 AUPR 차이) ==")
    pairs = {}
    for k in KS:
        a = "T-RADIAL/smooth%d_max" % k
        b = "대조 균일/smooth%d_max (=국소 밀도)" % k
        lo, hi, p = paired_aupr_diff_ci(scores[a], scores[b], is_def, n_boot=1000, seed=41)
        pairs["k=%d" % k] = {"diff_ci": [lo, hi], "p": p,
                             "aupr_radial": report[a]["aupr"], "aupr_uniform": report[b]["aupr"]}
        print("  k=%d  반경 %.4f vs 균일 %.4f   차이 [%+.4f, %+.4f]  p=%.4f  → %s"
              % (k, report[a]["aupr"], report[b]["aupr"], lo, hi, p,
                 "template 이 기여함" if lo > 0 else
                 ("template 이 해로움" if hi < 0 else "template 기여 없음")))

    print("\n== 국소 밀도 대조군이 바닥을 이기는가 (template 없이) ==")
    for k in KS:
        b = "대조 균일/smooth%d_max (=국소 밀도)" % k
        lo, hi, p = paired_aupr_diff_ci(scores[b], scores["E0 불량 다이 비율(바닥)"],
                                        is_def, n_boot=1000, seed=42)
        print("  k=%d  %.4f vs 0.3856  차이 [%+.4f, %+.4f] p=%.4f" % (
            k, report[b]["aupr"], lo, hi, p))

    print("\n== 클래스별 AUROC ==")
    cls = NAMES[1:]
    print("%-34s " % "arm" + " ".join("%10s" % c for c in cls))
    for name, m in sorted(report.items(), key=lambda kv: -kv[1]["aupr"]):
        print("%-34s " % name + " ".join(
            "%10.4f" % m["per_class"][c]["auroc"] if c in m["per_class"] else "%10s" % "-"
            for c in cls))

    print("\n== dieSize 구간별 AUPR ==")
    print("%-34s " % "arm" + " ".join("%11s" % b for b in BIN_LBL))
    for name, m in sorted(report.items(), key=lambda kv: -kv[1]["aupr"]):
        print("%-34s " % name + " ".join(
            "%11.4f" % m["per_size_bin"][b]["aupr"] if b in m["per_size_bin"]
            else "%11s" % "-" for b in BIN_LBL))

    (OUT / "density_control.json").write_text(
        json.dumps({"report": report, "paired": pairs}, ensure_ascii=False, indent=2))
    np.savez_compressed(OUT / "density_control_scores.npz",
                        **{k: v.astype(np.float32) for k, v in scores.items()},
                        y_test=yte, size_test=size_te)
    log("저장 완료 → %s" % OUT)
