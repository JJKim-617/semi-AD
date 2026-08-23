"""운영점별 arm 대응표 — 사람이 "우리는 X% 재현율이 필요하다" 고 답하면 즉시 고를 수 있게.

## 왜 이것을 미리 계산하는가

**95% 는 이 워크스트림이 고른 숫자이지 팹의 요구사항이 아니다.**
그 답은 **데이터로 못 푼다 — 사람에게 물어야 한다.**
그런데 지금은 답이 와도 분석을 다시 돌려야 한다. **미리 계산해 두면 답이 오는 즉시 끝난다.**

27차가 두 가지를 이미 보였다 — **AUPR 은 낮은 재현율에 무게가 쏠려 있고**(59.5%가 0~50%),
**arm 마다 최적 꼬리 가중이 다르다.** 그러면 **운영점이 arm 선택을 정할 수 있다.**
**정말 그런지, 그렇다면 경계가 어디인지**를 잰다.

## 이것은 판정이 아니다

**의사결정 자료다.** 주 지표는 `aupr_blocked` 그대로이고 **채택 표를 바꾸지 않는다.**
여기서 어떤 arm 이 어떤 운영점에서 1위여도 **그것으로 채택하지 않는다.**
학습 없음, 저장된 점수만, `test_dev` 만, **봉인 안 연다.**
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import aupr_blocked, operating_points, paired_fpr_at_tpr_diff_ci  # noqa: E402
from diag_metric_divergence import load_arms  # noqa: E402

OUT = Path("result/ood/o1_operating_table")
EVID = Path("docs/research/ood_metric_divergence/evidence")
TARGETS = (0.50, 0.80, 0.90, 0.95, 0.99)
BAND_EDGES = (0.0, 0.50, 0.80, 0.90, 0.95, 0.99, 1.0)
N_BOOT = 150
CTRL = "fuse3 (채택)"
T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


def ap_bands(score, label, edges=BAND_EDGES):
    """blocked AP 를 재현율 구간별 기여로 쪼갠다. 합은 `aupr_blocked` 와 같다."""
    s = np.asarray(score, np.float64)
    l = np.asarray(label, np.int64)
    o = np.argsort(-s, kind="mergesort")
    ss, ll = s[o], l[o]
    tp = np.cumsum(ll)
    tot = np.arange(1, len(ll) + 1)
    last = np.flatnonzero(np.concatenate([ss[1:] != ss[:-1], [True]]))
    prec, rec = tp[last] / tot[last], tp[last] / tp[-1]
    contrib = np.diff(np.concatenate([[0.0], rec])) * prec
    out = {}
    for i in range(len(edges) - 1):
        m = (rec > edges[i]) & (rec <= edges[i + 1] + 1e-12)
        if i == 0:
            m = rec <= edges[1] + 1e-12
        out["%d-%d%%" % (edges[i] * 100, edges[i + 1] * 100)] = float(contrib[m].sum())
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    EVID.mkdir(parents=True, exist_ok=True)
    arms, y_dev = load_arms()

    ss = np.load(Path("result/ood") / "o2_seed_scaling/seed_u.npz")
    U = ss["u"].astype(np.float64)
    f3 = arms[CTRL]
    arms["fuse5 (16 seed)"] = f3 - 2.0 * np.log(np.clip(1.0 - U.mean(0), 1e-9, 1.0))
    isd = (y_dev != 0).astype(np.int64)
    names = list(arms)
    log("arm %d개, 운영점 %d개, test_dev %d (결함 %d)"
        % (len(names), len(TARGETS), len(y_dev), int(isd.sum())))

    rep = {"targets": list(TARGETS), "n_boot": N_BOOT, "control": CTRL,
           "note": "의사결정 자료. 판정이 아니다. 주 지표는 aupr_blocked 그대로.",
           "aupr_bands": {}, "points": {}, "aupr": {}}

    for n in names:
        rep["aupr"][n] = aupr_blocked(arms[n], isd)
        rep["aupr_bands"][n] = ap_bands(arms[n], isd)

    n_pos = int(isd.sum())
    n_neg = int(len(isd) - n_pos)
    for t in TARGETS:
        rows = {}
        for n in names:
            o = operating_points(arms[n], isd, (t,))[0]
            rows[n] = {"fpr": o["false_positives"] / n_neg,
                       "fp": o["false_positives"], "precision": o["precision"],
                       "tied": o["n_tied_at_threshold"], "recall": o["recall"]}
        for n in names:
            if n == CTRL:
                continue
            lo, hi, p = paired_fpr_at_tpr_diff_ci(arms[n], arms[CTRL], isd, t, N_BOOT, 121)
            rows[n]["ci_vs_fuse3"] = [lo, hi]
            rows[n]["verdict"] = "이김" if hi < 0 else ("짐" if lo > 0 else "무승부")
        rep["points"]["%.2f" % t] = rows
        log("재현율 %.0f%% 완료" % (t * 100))

    # --- 출력 ----------------------------------------------------------------------
    print("\n== AUPR 이 각 재현율 구간에서 가져가는 몫 (fuse3) ==")
    b = rep["aupr_bands"][CTRL]
    tot = sum(b.values())
    print("%-12s %10s %8s" % ("구간", "기여", "몫"))
    for k, v in b.items():
        print("%-12s %10.4f %7.2f%%" % (k, v, 100 * v / tot))

    for t in TARGETS:
        rows = rep["points"]["%.2f" % t]
        order = sorted(names, key=lambda n: rows[n]["fpr"])
        print("\n== 재현율 %.0f%% — FPR 오름차순 (낮을수록 좋다) ==" % (t * 100))
        print("%-20s %9s %10s %10s %10s %22s %s"
              % ("arm", "FPR", "헛경보", "precision", "동점", "fuse3 대비 CI", "판정"))
        for n in order:
            r = rows[n]
            ci = ("[%+.4f,%+.4f]" % tuple(r["ci_vs_fuse3"])) if "ci_vs_fuse3" in r else "—"
            tie = "%d%s" % (r["tied"], " **" if r["tied"] > 1000 else "")
            print("%-20s %9.4f %10d %10.3f %10s %22s %s"
                  % (n, r["fpr"], r["fp"], r["precision"], tie, ci, r.get("verdict", "—")))

    # --- 순위가 뒤집히는가 -----------------------------------------------------------
    print("\n== 운영점마다 누가 1위인가 (동점 1,000장 초과는 운영 불가로 제외) ==")
    print("%8s %-22s %10s %12s | %-22s %s"
          % ("재현율", "1위 (동점 조건 통과)", "FPR", "헛경보", "fuse3 순위", "제외된 arm"))
    winners = {}
    for t in TARGETS:
        rows = rep["points"]["%.2f" % t]
        ok = [n for n in names if rows[n]["tied"] <= 1000]
        excl = [n for n in names if rows[n]["tied"] > 1000]
        order_ok = sorted(ok, key=lambda n: rows[n]["fpr"])
        order_all = sorted(names, key=lambda n: rows[n]["fpr"])
        w = order_ok[0]
        winners["%.2f" % t] = {"winner": w, "fuse3_rank": order_all.index(CTRL) + 1,
                               "fuse3_rank_among_operable": order_ok.index(CTRL) + 1,
                               "excluded": excl}
        print("%7.0f%% %-22s %10.4f %12d | %-22s %s"
              % (t * 100, w, rows[w]["fpr"], rows[w]["fp"],
                 "%d위 / %d" % (order_ok.index(CTRL) + 1, len(ok)),
                 ", ".join(excl) if excl else "없음"))
    rep["winners"] = winners

    ws = [winners["%.2f" % t]["winner"] for t in TARGETS]
    print("\n  1위가 운영점에 따라 바뀌는가: %s"
          % ("**바뀐다** — " + " -> ".join("%.0f%%:%s" % (t * 100, w)
                                         for t, w in zip(TARGETS, ws))
             if len(set(ws)) > 1 else "**안 바뀐다** (전 구간 %s)" % ws[0]))
    r3 = [winners["%.2f" % t]["fuse3_rank_among_operable"] for t in TARGETS]
    print("  fuse3 의 운영 가능 arm 중 순위: " +
          ", ".join("%.0f%% -> %d위" % (t * 100, r) for t, r in zip(TARGETS, r3)))

    (OUT / "operating_table.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    (EVID / "operating_table.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log("저장 완료 → %s  (판정 아님, 봉인 미개봉)" % OUT)
