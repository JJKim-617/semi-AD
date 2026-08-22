"""봉인 홀드아웃 일회 평가. 반증 조건은 `candidate/ood_sealed_holdout.md` 에 실행 전에 박았다.

**아무것도 다시 적합하지 않는다** — 백분위 참조와 반경 대역 참조는 train-none 그대로고
평가 집합만 바뀐다(9차 감사와 같은 규약).

세 파티션을 같은 자로 잰다.

| | 성질 |
|---|---|
| `test_dev` | 여덟 사이클 동안 본 자료. **낙관적으로 편향된 숫자** |
| `test_sealed` | 같은 분포, lot 분리. 그러나 **이미 본 웨이퍼**라 편향은 안 없어진다 |
| `val_unseen` | OOD arm 이 한 번도 점수를 매긴 적 없다. **편향은 없고 분포 이동이 섞인다** |

유병률이 다르므로(val 0.319 대 test 0.067) val 의 raw AUPR 은 test 와 비교할 수 없다.
정상을 전부 두고 결함만 541장 뽑아 유병률 0.0666 에 맞춘 AUPR 을 200번 재서 병기한다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import aupr, aupr_blocked, auroc, fail_ratio, fpr_at_tpr, operating_points  # noqa: E402
from a22_ood_template import assert_one_class  # noqa: E402
from a23_ood_template_eval import NAMES, _fast_auroc, ecdf_percentile, fisher  # noqa: E402
from a24_ood_residual import local_fail_count_map, local_fail_count_max, local_fail_density_max  # noqa: E402
from a27_line_filter import line_density_max  # noqa: E402
from a29_radial_calibration import band_index, calibrate_map, fit_band_reference  # noqa: E402
from a38_sealed_holdout import assert_not_sealed, load_partition  # noqa: E402

OUT = Path("result/ood/holdout")
CHUNK, N_BANDS, L_LINE = 4000, 32, 11
REASON = "사전등록 candidate/ood_sealed_holdout.md §5-R5 의 일회 평가"
TARGET_PREV = 0.0666
N_MATCH = 200
T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def blocked_ci(score, label, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    n = len(label)
    v = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        v[b] = aupr_blocked(score[i], label[i])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def prevalence_matched_aupr(score, y, target=TARGET_PREV, n_rep=N_MATCH):
    """정상을 전부 두고 결함을 뽑아 유병률을 맞춘 blocked AUPR.

    라벨을 쓰지만 **평가에서만** 쓴다 — 점수 계산에는 들어가지 않는다.
    """
    neg = np.flatnonzero(y == 0)
    pos = np.flatnonzero(y != 0)
    m = int(round(len(neg) * target / (1.0 - target)))
    if m > len(pos):
        raise ValueError("결함이 모자라 유병률을 못 맞춘다")
    v = np.empty(n_rep)
    for r in range(n_rep):
        rng = np.random.default_rng(r)
        take = np.concatenate([neg, rng.choice(pos, m, replace=False)])
        v[r] = aupr_blocked(score[take], (y[take] != 0).astype(np.int64))
    return {"n_defect_kept": m, "prevalence": m / (len(neg) + m),
            "mean": float(v.mean()),
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}


def per_class_auroc(score, yy):
    ns = score[yy == 0]
    z = np.zeros(len(ns), np.int64)
    rows = {}
    for c in range(1, 9):
        m = yy == c
        if not m.sum():
            continue
        s = np.concatenate([ns, score[m]])
        l = np.concatenate([z, np.ones(int(m.sum()), np.int64)])
        rows[NAMES[c]] = {"n": int(m.sum()), "auroc": _fast_auroc(s, l)}
    return rows


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    X = d["X"]

    trn = sp["train"][y[sp["train"]] == 0]
    assert_one_class(y[trn])
    assert_not_sealed(trn, "train-none")          # 적합 자료에 봉인이 섞이지 않았는지
    pad_tr = np.ascontiguousarray(X[trn])

    parts = {
        "test_dev": load_partition("test_dev"),
        "test_sealed": load_partition("test_sealed", unseal=True, reason=REASON),
        "val_unseen": load_partition("val_unseen", unseal=True, reason=REASON),
    }
    pads = {k: np.ascontiguousarray(X[v]) for k, v in parts.items()}
    ys = {k: y[v] for k, v in parts.items()}
    del X, d
    log("적재 완료 " + " ".join("%s %d(결함 %d)" % (k, len(v), int((ys[k] != 0).sum()))
                              for k, v in parts.items()))

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

    ctr = comps(pad_tr)
    log("train-none 참조 완료")

    def arms(x):
        c = comps(x)

        def fu(keys):
            return fisher([ecdf_percentile(ctr[k], c[k]) for k in keys])

        return {
            "채택 fuse3 (A+B+C)": fu(["A", "B", "C"]),
            "채택 A+B (6차)": fu(["A", "B"]),
            "채택 A 단독 (5차)": c["A"],
            "채택 die k=7 (진짜 바닥)": local_fail_density_max(x, k=7, chunk=CHUNK),
            "E0 불량 다이 비율 (원 바닥)": fail_ratio(x),
            "기각 선 L=7 가드없음 (3차)": line_density_max(x, length=7, n_orient=8, chunk=CHUNK),
        }

    report, score_store = {}, {}
    for pname, x in pads.items():
        sc = arms(x)
        score_store[pname] = sc
        yy = ys[pname]
        isd = (yy != 0).astype(np.int64)
        block = {}
        for si, (aname, s) in enumerate(sc.items()):
            row = {"aupr_blocked": aupr_blocked(s, isd), "auroc": auroc(s, isd),
                   "aupr_order": aupr(s, isd), "fpr_at_95tpr": fpr_at_tpr(s, isd, 0.95),
                   "n_unique": int(len(np.unique(s))),
                   "aupr_blocked_ci95": blocked_ci(s, isd, 300, 900 + si),
                   "operating": operating_points(s, isd, (0.5, 0.8, 0.95)),
                   "per_class": per_class_auroc(s, yy)}
            if pname == "val_unseen":
                row["aupr_blocked_prevalence_matched"] = prevalence_matched_aupr(s, yy)
            block[aname] = row
            log("%-12s %-26s AUPRblk %.4f AUROC %.4f FPR@95 %.4f"
                % (pname, aname, row["aupr_blocked"], row["auroc"], row["fpr_at_95tpr"]))
        report[pname] = {"n": len(yy), "n_defect": int(isd.sum()),
                         "prevalence": float(isd.mean()), "arms": block}

    # 재현 확인: dev + sealed 를 합치면 여덟 사이클이 본 test 전수와 같아야 한다
    saved = Path("result/ood/o1_fuse3/fuse3_scores.npz")
    if saved.exists():
        z = np.load(saved, allow_pickle=True)
        key = [k for k in z.files if k.startswith("주 fuse3")][0]
        te = sp["test"]
        pos = {int(v): i for i, v in enumerate(te)}
        got = np.concatenate([score_store["test_dev"]["채택 fuse3 (A+B+C)"],
                              score_store["test_sealed"]["채택 fuse3 (A+B+C)"]])
        want = z[key][[pos[int(v)] for v in np.concatenate([parts["test_dev"],
                                                            parts["test_sealed"]])]]
        report["reproduction"] = {
            "max_abs_diff_vs_saved_test_scores": float(np.abs(got - want).max()),
            "aupr_blocked_recombined": aupr_blocked(
                got, (np.concatenate([ys["test_dev"], ys["test_sealed"]]) != 0).astype(np.int64)),
        }
        log("재현 확인 최대 오차 %.3g, 합쳐 잰 blocked AUPR %.4f"
            % (report["reproduction"]["max_abs_diff_vs_saved_test_scores"],
               report["reproduction"]["aupr_blocked_recombined"]))

    names = list(score_store["test_dev"].keys())
    rank = {}
    for a, b in (("test_dev", "test_sealed"), ("test_dev", "val_unseen")):
        va = [report[a]["arms"][n]["aupr_blocked"] for n in names]
        vb = [report[b]["arms"][n]["aupr_blocked"] for n in names]
        rank["%s vs %s" % (a, b)] = spearman(va, vb)
    report["rank_spearman"] = rank

    print("\n== blocked AUPR ==")
    print("%-26s %10s %12s %10s %14s" % ("arm", "test_dev", "test_sealed", "val(raw)",
                                          "val(유병률맞춤)"))
    for n in names:
        r = [report[p]["arms"][n] for p in ("test_dev", "test_sealed", "val_unseen")]
        print("%-26s %10.4f %12.4f %10.4f %14.4f"
              % (n, r[0]["aupr_blocked"], r[1]["aupr_blocked"], r[2]["aupr_blocked"],
                 r[2]["aupr_blocked_prevalence_matched"]["mean"]))

    print("\n== AUROC / FPR@95TPR (유병률 무관, 직접 비교 가능) ==")
    print("%-26s %19s %19s %19s" % ("arm", "test_dev", "test_sealed", "val_unseen"))
    for n in names:
        cells = []
        for p in ("test_dev", "test_sealed", "val_unseen"):
            m = report[p]["arms"][n]
            cells.append("%.4f / %.4f" % (m["auroc"], m["fpr_at_95tpr"]))
        print("%-26s %19s %19s %19s" % (n, *cells))

    print("\n== 조건 R3: fuse3 − die k=7 격차 (test 에서 +0.1227) ==")
    for p, key in (("test_dev", "aupr_blocked"), ("test_sealed", "aupr_blocked")):
        g = (report[p]["arms"]["채택 fuse3 (A+B+C)"][key]
             - report[p]["arms"]["채택 die k=7 (진짜 바닥)"][key])
        print("  %-12s %+.4f" % (p, g))
    gv = (report["val_unseen"]["arms"]["채택 fuse3 (A+B+C)"]["aupr_blocked_prevalence_matched"]["mean"]
          - report["val_unseen"]["arms"]["채택 die k=7 (진짜 바닥)"]["aupr_blocked_prevalence_matched"]["mean"])
    print("  %-12s %+.4f  (유병률 맞춤. 판정 문턱 +0.0614)" % ("val_unseen", gv))
    report["r3_gap"] = {"test_dev": (report["test_dev"]["arms"]["채택 fuse3 (A+B+C)"]["aupr_blocked"]
                                     - report["test_dev"]["arms"]["채택 die k=7 (진짜 바닥)"]["aupr_blocked"]),
                        "test_sealed": (report["test_sealed"]["arms"]["채택 fuse3 (A+B+C)"]["aupr_blocked"]
                                        - report["test_sealed"]["arms"]["채택 die k=7 (진짜 바닥)"]["aupr_blocked"]),
                        "val_unseen_prevalence_matched": gv, "threshold": 0.0614}

    print("\n== 순위 Spearman ==")
    for k, v in rank.items():
        print("  %-28s %.4f" % (k, v))

    print("\n== 운영 지점 (재현율 80%) precision / 헛경보 ==")
    for n in names:
        cells = ["%.3f/%6d" % (report[p]["arms"][n]["operating"][1]["precision"],
                               report[p]["arms"][n]["operating"][1]["false_positives"])
                 for p in ("test_dev", "test_sealed", "val_unseen")]
        print("  %-26s dev %s  sealed %s  val %s" % (n, *cells))

    print("\n== 클래스별 AUROC — 채택 fuse3 ==")
    print("%-12s " % "파티션" + " ".join("%12s" % c for c in NAMES[1:]))
    for p in ("test_dev", "test_sealed", "val_unseen"):
        pc = report[p]["arms"]["채택 fuse3 (A+B+C)"]["per_class"]
        print("%-12s " % p + " ".join(
            "%6.4f(%4d)" % (pc[c]["auroc"], pc[c]["n"]) if c in pc else "%12s" % "-"
            for c in NAMES[1:]))

    (OUT / "sealed_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    log("저장 완료 → %s" % (OUT / "sealed_eval.json"))
