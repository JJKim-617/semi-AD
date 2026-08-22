"""O1 실행기 — 정상 통계 template 잔차를 공식 test 전수로 채점한다. CPU 전용.

반증 조건은 `docs/experiments/candidate/ood_template_residual.md` 에 **실행 전에** 박혀 있다.
이 스크립트는 그 조건을 판정하는 데 필요한 표를 전부 낸다:
전체 지표, 클래스별, dieSize 구간별, AUPR 부트스트랩 95% CI.

template 은 **train & y==0 인 29,149장만** 본다(`assert_one_class`).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import aupr, auroc, evaluate_ood, fail_count, fail_ratio  # noqa: E402
from a22_ood_template import (  # noqa: E402
    assert_one_class,
    fit_position_template,
    fit_radial_template,
    score_llr,
    score_nll,
    uniform_template,
)

CACHE_PAD = "data/wm811k/cache/wm811k_64pad.npz"
CACHE_RESIZE = "data/wm811k/cache/wm811k_64.npz"
SPLITS = "data/wm811k/cache/splits_v1.npz"
OUT = Path("result/ood/o1_template")
NAMES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch",
         "Near-full"]
BINS = [0, 400, 562, 776, 1090, 1334, 1600, 10 ** 9]
BIN_LBL = ["<400", "400-562", "562-776", "776-1090", "1090-1334", "1334-1600", ">1600"]
N_BOOT = 1000


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def bootstrap_aupr_ci(score, label, n_boot=N_BOOT, seed=0):
    """웨이퍼 단위 재표본 AUPR 의 95% 백분위 구간.

    매번 argsort 하면 느리다. **한 번 정렬해 두고 재표본을 다항분포 가중치로** 넣으면
    같은 동점 순서를 유지한 채 정확히 같은 계단 적분을 얻는다.
    """
    score = np.asarray(score, np.float64)
    label = np.asarray(label, np.int64)
    order = np.argsort(-score, kind="mergesort")
    l = label[order].astype(np.float64)
    n = len(l)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, np.float64)
    for b in range(n_boot):
        c = rng.multinomial(n, np.full(n, 1.0 / n)).astype(np.float64)
        tp = np.cumsum(l * c)
        tot = np.cumsum(c)
        npos = tp[-1]
        if npos <= 0:
            vals[b] = np.nan
            continue
        prec = tp / np.maximum(tot, 1.0)
        rec = tp / npos
        vals[b] = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def _aupr_from_counts(l_sorted: np.ndarray, c_sorted: np.ndarray) -> float:
    """이미 내림차순 정렬된 라벨과 재표본 가중치로 AUPR 계단 적분."""
    tp = np.cumsum(l_sorted * c_sorted)
    tot = np.cumsum(c_sorted)
    npos = tp[-1]
    if npos <= 0:
        return np.nan
    prec = tp / np.maximum(tot, 1.0)
    rec = tp / npos
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def paired_aupr_diff_ci(score_a, score_b, label, n_boot=1000, seed=0):
    """**같은 재표본**에서 잰 AUPR(a) - AUPR(b) 의 95% 구간과 양측 p.

    두 점수는 같은 test 집합 위에서 계산되므로 표본 변동을 공유한다.
    독립 CI 겹침으로 판정하면 지나치게 보수적이어서 실제 차이를 놓친다
    (`ood_template_residual.md` 반증 조건 6 의 정정 사유).

    재표본 가중치 c 를 원 인덱스 위에서 한 번 뽑고, 두 점수의 정렬 순서로
    각각 재배열해 넣는다. 그래서 두 arm 이 **문자 그대로 같은 웨이퍼 묶음**을 본다.
    """
    a = np.asarray(score_a, np.float64)
    b = np.asarray(score_b, np.float64)
    l = np.asarray(label, np.int64)
    n = len(l)
    oa = np.argsort(-a, kind="mergesort")
    ob = np.argsort(-b, kind="mergesort")
    la, lb = l[oa].astype(np.float64), l[ob].astype(np.float64)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot, np.float64)
    p_uniform = np.full(n, 1.0 / n)
    for i in range(n_boot):
        c = rng.multinomial(n, p_uniform).astype(np.float64)
        d[i] = _aupr_from_counts(la, c[oa]) - _aupr_from_counts(lb, c[ob])
    lo = float(np.nanpercentile(d, 2.5))
    hi = float(np.nanpercentile(d, 97.5))
    frac = float(np.nanmean(d <= 0.0))
    p = 2.0 * min(frac, 1.0 - frac)
    return lo, hi, p


def paired_aupr_blocked_diff_ci(score_a, score_b, label, n_boot=800, seed=0):
    """**주 지표(blocked AUPR)로 하는 짝지은 비교.**

    `paired_aupr_diff_ci` 는 순서 의존 AUPR 을 쓴다. 동점이 적으면 같지만
    `k2` 계열은 고유값이 25개뿐이라 두 관례가 0.017 까지 벌어진다.
    **주 지표를 blocked 로 정해 놓고 검정만 순서 의존으로 하면 다른 것을 재게 된다.**

    재표본 인덱스를 한 번 뽑아 두 점수에 똑같이 먹인다. blocked 은 재표본마다
    블록 구조가 달라지므로 매번 다시 정렬한다 — 느리지만 그게 정의다.
    """
    from a21_ood_metrics import aupr_blocked

    a = np.asarray(score_a, np.float64)
    b = np.asarray(score_b, np.float64)
    l = np.asarray(label, np.int64)
    n = len(l)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot, np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        li = l[idx]
        if li.sum() == 0 or li.sum() == n:
            d[i] = np.nan
            continue
        d[i] = aupr_blocked(a[idx], li) - aupr_blocked(b[idx], li)
    lo = float(np.nanpercentile(d, 2.5))
    hi = float(np.nanpercentile(d, 97.5))
    frac = float(np.nanmean(d <= 0.0))
    return lo, hi, 2.0 * min(frac, 1.0 - frac)


def _fast_auroc(score, label):
    """`a21_ood_metrics.auroc` 와 같은 값. 동점 구간 루프를 벡터화만 했다.

    원본은 동점 그룹을 파이썬 루프로 돈다. 값은 같지만 부트스트랩 수천 회에는 느리다.
    같은 값을 준다는 것은 `test_fast_auroc_matches_reference` 로 박아 둔다.
    """
    score = np.asarray(score, np.float64)
    label = np.asarray(label, np.int64)
    o = np.argsort(score, kind="mergesort")
    s, l = score[o], label[o]
    n = len(s)
    start = np.flatnonzero(np.concatenate([[True], s[1:] != s[:-1]]))
    end = np.concatenate([start[1:], [n]])
    ranks = np.repeat((start + end - 1) / 2.0 + 1.0, end - start)
    n1 = float(l.sum())
    n0 = float(n - n1)
    return float((ranks[l == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bootstrap_auroc_ci(score, label, n_boot=200, seed=0):
    """동점 평균랭크 AUROC 의 부트스트랩 구간. 클래스별 판정에 쓴다."""
    score = np.asarray(score, np.float64)
    label = np.asarray(label, np.int64)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, np.float64)
    pos = np.flatnonzero(label == 1)
    neg = np.flatnonzero(label == 0)
    for b in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        vals[b] = _fast_auroc(score[i], label[i])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def per_class(score, yte, ci_for=("Scratch", "Center", "Loc")):
    ns = score[yte == 0]
    z = np.zeros(len(ns), np.int64)
    rows = {}
    for c in range(1, 9):
        m = yte == c
        if not m.sum():
            continue
        s = np.concatenate([ns, score[m]])
        l = np.concatenate([z, np.ones(int(m.sum()), np.int64)])
        row = {"n": int(m.sum()), "auroc": auroc(s, l)}
        if NAMES[c] in ci_for:
            row["auroc_ci"] = bootstrap_auroc_ci(s, l, n_boot=200, seed=c)
        rows[NAMES[c]] = row
    return rows


def per_size_bin(score, is_def, size):
    rows = {}
    for i, lbl in enumerate(BIN_LBL):
        m = (size >= BINS[i]) & (size < BINS[i + 1])
        if m.sum() == 0 or is_def[m].sum() == 0 or is_def[m].sum() == m.sum():
            continue
        rows[lbl] = {"n": int(m.sum()), "n_def": int(is_def[m].sum()),
                     "auroc": auroc(score[m], is_def[m]), "aupr": aupr(score[m], is_def[m])}
    return rows


def ecdf_percentile(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """train-none 경험분포에서의 백분위. 보정에 정상만 쓴다."""
    ref = np.sort(np.asarray(ref, np.float64))
    return np.searchsorted(ref, np.asarray(x, np.float64), side="right") / (len(ref) + 1.0)


def fisher(us: list[np.ndarray]) -> np.ndarray:
    """일측 p = 1-u 에 Fisher 결합. 큰 값이 더 이상하다."""
    out = np.zeros_like(us[0])
    for u in us:
        out += -2.0 * np.log(np.clip(1.0 - u, 1e-9, 1.0))
    return out


T0 = time.time()

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sp = np.load(SPLITS)
    tr, te = sp["train"], sp["test"]

    dpad = np.load(CACHE_PAD, allow_pickle=True)
    y = dpad["y"].astype(np.int64)
    size = dpad["die_size"].astype(np.float64)
    trn = tr[y[tr] == 0]
    assert_one_class(y[trn])
    log("train none %d (one-class 가드 통과), test %d" % (len(trn), len(te)))

    yte = y[te]
    is_def = (yte != 0).astype(np.int64)
    size_te = size[te]

    Xpad = dpad["X"]
    pad_tr = np.ascontiguousarray(Xpad[trn])
    pad_te = np.ascontiguousarray(Xpad[te])
    del Xpad, dpad
    log("pad 캐시 적재 완료")

    dres = np.load(CACHE_RESIZE, allow_pickle=True)
    Xres = dres["X"]
    res_tr = np.ascontiguousarray(Xres[trn])
    res_te = np.ascontiguousarray(Xres[te])
    del Xres, dres
    log("resize 캐시 적재 완료")

    # --- template 적합 (정상만) ------------------------------------------------
    t_pad = fit_position_template(pad_tr, y=y[trn], smoothing=1.0)
    t_res = fit_position_template(res_tr, y=y[trn], smoothing=1.0)
    t_rad = fit_radial_template(pad_tr, y=y[trn], n_bins=32, smoothing=1.0)
    t_uni = uniform_template(t_pad.global_rate)
    log("template 3종 + 대조군 적합. 전역 불량률 pad %.5f / resize %.5f / radial %.5f"
        % (t_pad.global_rate, t_res.global_rate, t_rad.global_rate))
    log("radial p (32빈): " + " ".join("%.4f" % v for v in t_rad.p))
    log("pad die_count 가 0 인 셀 %d / %d" % (int((t_pad.die_count == 0).sum()), 64 * 64))

    # --- 채점 -----------------------------------------------------------------
    scores = {}
    scores["E0 불량 다이 비율(대조)"] = fail_ratio(pad_te)
    scores["E0 불량 다이 개수(대조)"] = fail_count(pad_te)
    scores["U-NLL 균일 template(대조)"] = score_nll(pad_te, t_uni)
    scores["T-PAD S_nll"] = score_nll(pad_te, t_pad)
    scores["T-PAD S_llr"] = score_llr(pad_te, t_pad)
    scores["T-RESIZE S_nll"] = score_nll(res_te, t_res)
    scores["T-RESIZE S_llr"] = score_llr(res_te, t_res)
    scores["T-RADIAL S_nll"] = score_nll(pad_te, t_rad)
    scores["T-RADIAL S_llr"] = score_llr(pad_te, t_rad)
    log("채점 완료: %d arm" % len(scores))

    # --- 융합 (train-none 으로만 보정) -------------------------------------------
    ratio_tr = fail_ratio(pad_tr)
    llr_tr = {"T-PAD": score_llr(pad_tr, t_pad), "T-RESIZE": score_llr(res_tr, t_res),
              "T-RADIAL": score_llr(pad_tr, t_rad)}
    llr_te = {"T-PAD": scores["T-PAD S_llr"], "T-RESIZE": scores["T-RESIZE S_llr"],
              "T-RADIAL": scores["T-RADIAL S_llr"]}
    u_ratio = ecdf_percentile(ratio_tr, scores["E0 불량 다이 비율(대조)"])
    for k in llr_tr:
        scores["FUSE 비율+%s S_llr" % k] = fisher(
            [u_ratio, ecdf_percentile(llr_tr[k], llr_te[k])])
    log("융합 arm 3종 완료")

    # --- 평가 -----------------------------------------------------------------
    report = {}
    for si, (name, s) in enumerate(scores.items()):
        m = evaluate_ood(s, is_def)
        m["aupr_ci95"] = bootstrap_aupr_ci(s, is_def, seed=1000 + si)
        m["per_class"] = per_class(s, yte)
        m["per_size_bin"] = per_size_bin(s, is_def, size_te)
        report[name] = m
        log("%-28s AUROC %.4f AUPR %.4f [%.4f,%.4f] FPR@95 %.4f"
            % (name, m["auroc"], m["aupr"], m["aupr_ci95"][0], m["aupr_ci95"][1],
               m["fpr_at_95tpr"]))

    (OUT / "o1_template_metrics.json").write_text(
        json.dumps({"report": report,
                    "template_radial_p": t_rad.p.tolist(),
                    "global_rate": {"pad": t_pad.global_rate, "resize": t_res.global_rate,
                                    "radial": t_rad.global_rate},
                    "n_train_none": int(len(trn)), "n_test": int(len(te)),
                    "n_boot": N_BOOT}, ensure_ascii=False, indent=2))
    np.savez_compressed(OUT / "o1_template_scores.npz",
                        **{k: v.astype(np.float32) for k, v in scores.items()},
                        y_test=yte, size_test=size_te,
                        template_pad=t_pad.p.astype(np.float32),
                        template_resize=t_res.p.astype(np.float32),
                        template_radial=t_rad.p.astype(np.float32))
    log("저장 완료 → %s" % OUT)

    # --- 읽기용 표 -------------------------------------------------------------
    print("\n== 전체 ==")
    print("%-28s %8s %8s %20s %10s" % ("arm", "AUROC", "AUPR", "AUPR 95% CI", "FPR@95TPR"))
    for name, m in report.items():
        print("%-28s %8.4f %8.4f  [%.4f, %.4f] %10.4f"
              % (name, m["auroc"], m["aupr"], m["aupr_ci95"][0], m["aupr_ci95"][1],
                 m["fpr_at_95tpr"]))

    print("\n== 클래스별 AUROC ==")
    cls = [c for c in NAMES[1:]]
    print("%-28s " % "arm" + " ".join("%10s" % c for c in cls))
    for name, m in report.items():
        print("%-28s " % name
              + " ".join("%10.4f" % m["per_class"][c]["auroc"] if c in m["per_class"]
                         else "%10s" % "-" for c in cls))

    print("\n== dieSize 구간별 AUPR ==")
    print("%-28s " % "arm" + " ".join("%11s" % b for b in BIN_LBL))
    for name, m in report.items():
        print("%-28s " % name
              + " ".join("%11.4f" % m["per_size_bin"][b]["aupr"] if b in m["per_size_bin"]
                         else "%11s" % "-" for b in BIN_LBL))

    print("\n== Scratch/Center/Loc AUROC 부트스트랩 CI (300회) ==")
    for name, m in report.items():
        parts = []
        for c in ("Scratch", "Center", "Loc"):
            r = m["per_class"].get(c, {})
            if "auroc_ci" in r:
                parts.append("%s %.4f [%.4f,%.4f]" % (c, r["auroc"], r["auroc_ci"][0],
                                                      r["auroc_ci"][1]))
        print("%-28s %s" % (name, "  ".join(parts)))
