"""두 지표가 왜 갈리는가 — 학습 없이, 저장된 점수만으로.

## 왜 지금 묻는가

**서로 다른 네 arm 이 같은 방향으로 갈렸다** — 21차 fuse4, 22차 앙상블,
24차 고재현율, 26차 seed 앙상블. 전부 `aupr_blocked` 는 무승부·패배인데
`FPR@95TPR` 은 유의하게 개선이다. **네 번이면 잡음이 아니다.**

**그런데 팹이 실제로 감당하는 것은 후자다** — "결함 95% 를 잡을 때 정상 몇 장을 띄우나".

## 이 스크립트가 하는 것과 안 하는 것

**한다**: 오늘 잰 모든 arm 에 두 지표를 나란히 놓고, 갈라짐을 정량화하고, 기작을 잰다.
**안 한다**: **주 지표를 바꾸지 않는다. 판정을 다시 하지 않는다. 새 arm 을 만들지 않는다.**
여기서 나오는 어떤 수치도 오늘의 판정을 되살리거나 뒤집는 데 쓰지 않는다.
**결과를 보고 지표를 바꾸면 그게 정정 9 의 반복이다.**

`test_sealed` 를 열지 않는다. 학습을 하지 않는다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import (  # noqa: E402
    aupr_blocked, fpr_at_tpr, operating_points, paired_fpr_at_tpr_diff_ci,
)
from a23_ood_template_eval import paired_aupr_blocked_diff_ci  # noqa: E402

OUT = Path("result/ood/o1_metric_divergence")
EVID = Path("docs/research/ood_metric_divergence/evidence")
R = Path("result/ood")
N_BOOT = 200
T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


def ap_by_recall_band(score, label, edges=(0.0, 0.5, 0.8, 0.95, 1.0)):
    """**blocked AP 를 재현율 구간별 기여로 쪼갠다.**

    `AP = sum (dR) * P` 이므로 각 항을 그 항의 재현율이 속한 구간에 더하면
    **AUPR 의 어느 부분이 어디서 오는지**가 그대로 나온다.
    합은 `aupr_blocked` 와 정확히 같아야 한다(테스트로 확인한다).
    """
    s = np.asarray(score, np.float64)
    l = np.asarray(label, np.int64)
    o = np.argsort(-s, kind="mergesort")
    ss, ll = s[o], l[o]
    tp = np.cumsum(ll)
    tot = np.arange(1, len(ll) + 1)
    last = np.flatnonzero(np.concatenate([ss[1:] != ss[:-1], [True]]))
    prec = tp[last] / tot[last]
    rec = tp[last] / tp[-1]
    dR = np.diff(np.concatenate([[0.0], rec]))
    contrib = dR * prec
    out = {}
    for i in range(len(edges) - 1):
        m = (rec > edges[i]) & (rec <= edges[i + 1] + 1e-12)
        if i == 0:
            m = rec <= edges[1] + 1e-12
        out["%.2f-%.2f" % (edges[i], edges[i + 1])] = float(contrib[m].sum())
    return out, float(contrib.sum())


def load_arms():
    """오늘 저장된 점수를 전부 모은다. 같은 파티션인지 확인한다."""
    arms, y = {}, None
    src = [
        ("o1_weighting/weighting_scores.npz", {
            "채택 fuse3 (동일 가중 Fisher)": "fuse3 (채택)",
            "W0 대조 (동일 가중 probit 합)": "W0 probit 합",
            "W1 주 (S^-1 가중 probit 합)": "W1 S^-1 가중",
            "요소 A 단독": "요소 A", "요소 B 단독": "요소 B", "요소 C 단독": "요소 C"}),
        ("o1_tail/tail_scores.npz", {
            "1 Wilkinson 최소 (연접)": "Wilkinson 최소", "2 Edgington 합": "Edgington 합",
            "3 Stouffer probit 합": "Stouffer 합", "5 Tippett 최대 (이접)": "Tippett 최대"}),
        ("o2_ssl_knn/o2_scores.npz", {
            "O2_trained_s0": "O2 12ep s0", "O2_trained_s1": "O2 12ep s1",
            "O2_trained_s2": "O2 12ep s2"}),
        ("o2_ensemble/ensemble_scores.npz", {
            "fuse4_s0_(참고)": "fuse4 s0", "fuse4_s1_(참고)": "fuse4 s1",
            "fuse4_s2_(참고)": "fuse4 s2", "O2_앙상블_단독_(참고)": "O2 앙상블 단독",
            "fuse5_앙상블_(주)": "fuse5 (3 seed)"}),
    ]
    for rel, keys in src:
        z = np.load(R / rel, allow_pickle=True)
        if y is None:
            y = z["y_dev"]
        else:
            assert np.array_equal(y, z["y_dev"]), rel
        for k, name in keys.items():
            arms[name] = np.asarray(z[k], np.float64)
    return arms, y


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    EVID.mkdir(parents=True, exist_ok=True)
    arms, y_dev = load_arms()

    ss = np.load(R / "o2_seed_scaling/seed_u.npz")
    assert np.array_equal(ss["y_dev"], y_dev)
    U = ss["u"].astype(np.float64)
    arms["fuse5 (16 seed)"] = None          # 아래에서 만든다
    isd = (y_dev != 0).astype(np.int64)
    log("arm %d개, test_dev %d (결함 %d)" % (len(arms), len(y_dev), int(isd.sum())))

    # fuse3 의 세 요소 Fisher 기여를 되살린다 (fuse3 = 세 기여의 합)
    f3 = arms["fuse3 (채택)"]
    fis = lambda u: -2.0 * np.log(np.clip(1.0 - u, 1e-9, 1.0))
    f_o2_16 = fis(U.mean(0))
    arms["fuse5 (16 seed)"] = f3 + f_o2_16

    # --- (1) 모든 arm 에 두 지표를 나란히 ------------------------------------------
    base_a = aupr_blocked(f3, isd)
    base_f = fpr_at_tpr(f3, isd, 0.95)
    rows = {}
    for name, s in arms.items():
        a, f = aupr_blocked(s, isd), fpr_at_tpr(s, isd, 0.95)
        rows[name] = {"aupr_blocked": a, "fpr_at_95tpr": f,
                      "d_aupr": a - base_a, "d_fpr": f - base_f}
    for name, s in arms.items():
        if name == "fuse3 (채택)":
            continue
        lo, hi, p = paired_aupr_blocked_diff_ci(s, f3, isd, N_BOOT, 111)
        flo, fhi, fp = paired_fpr_at_tpr_diff_ci(s, f3, isd, 0.95, N_BOOT, 112)
        w_a = "이김" if lo > 0 else ("짐" if hi < 0 else "무승부")
        w_f = "이김" if fhi < 0 else ("짐" if flo > 0 else "무승부")   # 낮을수록 좋다
        rows[name].update({"aupr_ci": [lo, hi], "fpr_ci": [flo, fhi],
                           "aupr_verdict": w_a, "fpr_verdict": w_f,
                           "diverges": bool(w_a != w_f and "무승부" not in (w_a, w_f))
                           or bool(w_a == "짐" and w_f == "이김")})
        log("  %-20s AUPR %s / FPR %s" % (name, w_a, w_f))

    print("\n== 오늘 잰 모든 arm — 두 지표를 나란히 (fuse3 기준) ==")
    print("%-20s %9s %9s | %9s %9s | %s"
          % ("arm", "AUPR블록", "차이", "FPR@95", "차이", "판정 (AUPR / FPR)"))
    for n, r in sorted(rows.items(), key=lambda kv: -kv[1]["aupr_blocked"]):
        v = ("%s / %s" % (r.get("aupr_verdict", "—"), r.get("fpr_verdict", "—")))
        mark = "  <== 갈림" if r.get("diverges") else ""
        print("%-20s %9.4f %+9.4f | %9.4f %+9.4f | %s%s"
              % (n, r["aupr_blocked"], r["d_aupr"], r["fpr_at_95tpr"], r["d_fpr"], v, mark))

    div = [n for n, r in rows.items() if r.get("diverges")]
    print("\n  갈린 arm %d개 / 비교한 %d개: %s" % (len(div), len(rows) - 1, ", ".join(div)))

    # --- (2) 기작 A: AUPR 은 어느 재현율 구간에서 오는가 ----------------------------
    print("\n== 기작 A: blocked AP 를 재현율 구간별 기여로 쪼갠다 ==")
    print("%-20s %11s %11s %11s %11s %10s"
          % ("arm", "0-50%", "50-80%", "80-95%", "**95-100%**", "합"))
    band = {}
    for n in ("fuse3 (채택)", "fuse5 (16 seed)", "Tippett 최대", "요소 A"):
        b, tot = ap_by_recall_band(arms[n], isd)
        band[n] = {"bands": b, "total": tot,
                   "frac_95plus": b["0.95-1.00"] / tot}
        print("%-20s %11.4f %11.4f %11.4f %11.4f %10.4f"
              % (n, b["0.00-0.50"], b["0.50-0.80"], b["0.80-0.95"], b["0.95-1.00"], tot))
    print("  fuse3 에서 95%% 재현율 이상 구간이 AUPR 에 기여하는 몫: **%.2f%%**"
          % (100 * band["fuse3 (채택)"]["frac_95plus"]))

    # --- (3) 기작 B: 꼬리 가중 손잡이로 두 지표를 동시에 추적 ------------------------
    # F = f_A + f_B + f_C + q * f_O2.  q=0 이면 fuse3, q=1 이면 fuse5(16).
    # **q 는 arm 이 아니다. 진단용 손잡이다.** test 로 고르지 않는다.
    print("\n== 기작 B: O2 기여 가중 q (q=0 이 fuse3, q=1 이 fuse5_16) ==")
    print("%6s %11s %11s %14s" % ("q", "AUPR블록", "FPR@95", "95% 헛경보"))
    trace = []
    for q in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        s = f3 + q * f_o2_16
        a, f = aupr_blocked(s, isd), fpr_at_tpr(s, isd, 0.95)
        fp = operating_points(s, isd, (0.95,))[0]["false_positives"]
        trace.append({"q": q, "aupr_blocked": a, "fpr_at_95tpr": f, "fp_at_95": fp})
        print("%6.2f %11.4f %11.4f %14d" % (q, a, f, fp))
    qa = max(trace, key=lambda r: r["aupr_blocked"])["q"]
    qf = min(trace, key=lambda r: r["fpr_at_95tpr"])["q"]
    print("  AUPR 최대 q = %.2f | FPR@95TPR 최소 q = %.2f  ->  %s"
          % (qa, qf, "**두 지표가 서로 다른 q 를 고른다**" if qa != qf else "같은 q 를 고른다"))

    rep = {"base": {"aupr_blocked": base_a, "fpr_at_95tpr": base_f},
           "arms": rows, "diverged": div, "recall_bands": band, "q_trace": trace,
           "q_best_aupr": qa, "q_best_fpr": qf,
           "note": "진단 전용. 주 지표를 바꾸지 않고 판정을 다시 하지 않는다."}
    (OUT / "metric_divergence.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    (EVID / "metric_divergence.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log("저장 완료 → %s  (봉인 미개봉, 학습 없음)" % OUT)
