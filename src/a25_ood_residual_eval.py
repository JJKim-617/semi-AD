"""맵 → 스칼라 뭉개기 비교. 잔차 맵의 어느 요약이 어떤 결함을 잡는가.

기획서 §9.2 — 뭉개는 방식은 **하이퍼파라미터가 아니라 설계 선택**이다.
근거는 `docs/experiments/candidate/ood_residual_pooling.md` 에 실행 전에 적었다.

template 3종 x 잔차 모드 2종 x 뭉개기 8종 = 48 arm 을 공식 test 전수로 잰다.
맵은 청크마다 만들고 버린다(전부 들고 있으면 3.9 GB).
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
    fit_position_template,
    fit_radial_template,
)
from a23_ood_template_eval import (  # noqa: E402
    BIN_LBL,
    NAMES,
    bootstrap_aupr_ci,
    ecdf_percentile,
    fisher,
    per_class,
    per_size_bin,
)
from a24_ood_residual import POOLERS, residual_map  # noqa: E402

OUT = Path("result/ood/o1_pooling")
CHUNK = 2000
T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def pooled_scores(x, template, modes=("nll", "llr")):
    """청크로 돌며 (mode, pooler) 조합의 스칼라를 모은다. 맵은 들고 있지 않는다."""
    out = {(m, p): np.empty(len(x), np.float64) for m in modes for p in POOLERS}
    for a in range(0, len(x), CHUNK):
        xb = x[a:a + CHUNK]
        die = xb > 0
        for mode in modes:
            mp = residual_map(xb, template, mode=mode, chunk=CHUNK)
            for pname, fn in POOLERS.items():
                out[(mode, pname)][a:a + len(xb)] = np.asarray(fn(mp, die))
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    tr, te = sp["train"], sp["test"]
    dpad = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = dpad["y"].astype(np.int64)
    size = dpad["die_size"].astype(np.float64)
    trn = tr[y[tr] == 0]
    assert_one_class(y[trn])
    Xpad = dpad["X"]
    pad_tr = np.ascontiguousarray(Xpad[trn])
    pad_te = np.ascontiguousarray(Xpad[te])
    del Xpad, dpad
    dres = np.load("data/wm811k/cache/wm811k_64.npz", allow_pickle=True)
    Xres = dres["X"]
    res_tr = np.ascontiguousarray(Xres[trn])
    res_te = np.ascontiguousarray(Xres[te])
    del Xres, dres
    yte = y[te]
    is_def = (yte != 0).astype(np.int64)
    size_te = size[te]
    log("적재 완료. train none %d / test %d" % (len(trn), len(te)))

    templates = {
        "T-PAD": (fit_position_template(pad_tr, y=y[trn], smoothing=1.0), pad_tr, pad_te),
        "T-RESIZE": (fit_position_template(res_tr, y=y[trn], smoothing=1.0), res_tr, res_te),
        "T-RADIAL": (fit_radial_template(pad_tr, y=y[trn], n_bins=32, smoothing=1.0),
                     pad_tr, pad_te),
    }
    log("template 3종 적합 (정상 %d장만)" % len(trn))

    scores, train_scores = {}, {}
    for tname, (t, xtr, xte) in templates.items():
        s = pooled_scores(xte, t)
        for (mode, pname), v in s.items():
            scores["%s %s/%s" % (tname, mode, pname)] = v
        log("%s test 뭉개기 완료" % tname)
        s_tr = pooled_scores(xtr, t)
        for (mode, pname), v in s_tr.items():
            train_scores["%s %s/%s" % (tname, mode, pname)] = v
        log("%s train-none 뭉개기 완료 (융합 보정용)" % tname)

    ratio_te = fail_ratio(pad_te)
    ratio_tr = fail_ratio(pad_tr)
    scores["E0 불량 다이 비율(대조)"] = ratio_te

    # --- 1차: 전 arm 점추정 + AUPR CI -------------------------------------------
    report = {}
    for si, (name, s) in enumerate(sorted(scores.items())):
        m = evaluate_ood(s, is_def)
        m["aupr_ci95"] = bootstrap_aupr_ci(s, is_def, n_boot=400, seed=2000 + si)
        report[name] = m
    log("1차 평가 완료 (%d arm)" % len(report))

    ranked = sorted(report.items(), key=lambda kv: -kv[1]["aupr"])
    top = [k for k, _ in ranked[:6]]

    # --- 융합: 상위 arm 을 불량 다이 비율과 섞는다 (train-none 으로만 보정) ----------
    u_ratio = ecdf_percentile(ratio_tr, ratio_te)
    for name in top:
        if name not in train_scores:
            continue
        fname = "FUSE 비율+%s" % name
        scores[fname] = fisher([u_ratio, ecdf_percentile(train_scores[name], scores[name])])
        m = evaluate_ood(scores[fname], is_def)
        m["aupr_ci95"] = bootstrap_aupr_ci(scores[fname], is_def, n_boot=400, seed=9000)
        report[fname] = m
    log("융합 arm 완료")

    # --- 2차: 상위 arm + 대조군에만 클래스별/구간별 -------------------------------
    ranked = sorted(report.items(), key=lambda kv: -kv[1]["aupr"])
    detail = [k for k, _ in ranked[:10]] + ["E0 불량 다이 비율(대조)"]
    for name in dict.fromkeys(detail):
        report[name]["per_class"] = per_class(scores[name], yte)
        report[name]["per_size_bin"] = per_size_bin(scores[name], is_def, size_te)
    log("2차 상세 완료")

    (OUT / "o1_pooling_metrics.json").write_text(
        json.dumps({"report": report, "n_train_none": int(len(trn))},
                   ensure_ascii=False, indent=2))
    np.savez_compressed(OUT / "o1_pooling_scores.npz",
                        **{k: v.astype(np.float32) for k, v in scores.items()},
                        y_test=yte, size_test=size_te)

    print("\n== 전 arm (AUPR 내림차순) ==")
    print("%-34s %8s %8s %20s %10s" % ("arm", "AUROC", "AUPR", "AUPR 95% CI", "FPR@95TPR"))
    for name, m in sorted(report.items(), key=lambda kv: -kv[1]["aupr"]):
        print("%-34s %8.4f %8.4f  [%.4f, %.4f] %10.4f"
              % (name, m["auroc"], m["aupr"], m["aupr_ci95"][0], m["aupr_ci95"][1],
                 m["fpr_at_95tpr"]))

    print("\n== 클래스별 AUROC (상세 낸 arm 만) ==")
    cls = NAMES[1:]
    print("%-34s " % "arm" + " ".join("%10s" % c for c in cls))
    for name, m in sorted(report.items(), key=lambda kv: -kv[1]["aupr"]):
        if "per_class" not in m:
            continue
        print("%-34s " % name + " ".join(
            "%10.4f" % m["per_class"][c]["auroc"] if c in m["per_class"] else "%10s" % "-"
            for c in cls))

    print("\n== dieSize 구간별 AUPR (상세 낸 arm 만) ==")
    print("%-34s " % "arm" + " ".join("%11s" % b for b in BIN_LBL))
    for name, m in sorted(report.items(), key=lambda kv: -kv[1]["aupr"]):
        if "per_size_bin" not in m:
            continue
        print("%-34s " % name + " ".join(
            "%11.4f" % m["per_size_bin"][b]["aupr"] if b in m["per_size_bin"]
            else "%11s" % "-" for b in BIN_LBL))
    log("저장 완료 → %s" % OUT)
