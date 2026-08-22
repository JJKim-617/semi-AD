"""운영 지점 표 + 동점 민감도. 보고서에 상설로 들어가는 숫자를 만든다.

## 왜 운영 지점인가

AUPR 총합은 **어디서 무너지는지를 가린다.** 팹이 감당하는 것은 곡선 아래 넓이가 아니라
"결함 80% 를 잡으려면 정상 몇 장을 띄우는가" 다.

## 왜 동점 민감도인가

조정자가 준 k=3 AUPR 이 0.3845 인데 내 실측은 0.4968 이다. 0.11 차이는 크다.
k=3 밀도는 **고유값이 24개뿐**이라 동점이 어마어마하고,
AUPR 의 계단 적분은 동점 구간을 어떤 순서로 훑느냐에 따라 값이 달라진다.
**두 숫자가 다 맞을 수 있다** — 그렇다면 그 지점에서 AUPR 은 관례를 밝히지 않으면
정의되지 않는 값이다. 여기서 그것을 확인한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import aupr, auroc, evaluate_ood, operating_points  # noqa: E402
from a24_ood_residual import local_fail_density_max  # noqa: E402

OUT = Path("docs/research/ood_operating_points/evidence")
RECALLS = (0.5, 0.8, 0.95)


def tie_sensitivity(score, label, n=40, seed=0):
    """동점을 무작위로 해소했을 때 AUPR 이 어디까지 움직이는가."""
    rng = np.random.default_rng(seed)
    s = np.asarray(score, np.float64)
    span = np.ptp(s) or 1.0
    vals = [aupr(s + rng.uniform(0, 1e-9 * span, size=len(s)), label) for _ in range(n)]
    return float(np.min(vals)), float(np.max(vals)), float(np.median(vals))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    te = sp["test"]
    X = np.ascontiguousarray(d["X"][te])
    del d
    yte = y[te]
    is_def = (yte != 0).astype(np.int64)

    scores = {}
    for k in (3, 5, 7, 9):
        scores["국소 밀도 max k=%d" % k] = local_fail_density_max(X, k=k, chunk=4000)
    cal = np.load("result/ood/o1_radialcal/radial_calibration_scores.npz", allow_pickle=True)
    for key in cal.files:
        if key.startswith("주 "):
            scores["반경보정 밀도 k=7"] = cal[key].astype(np.float64)
    scores["E0 불량 다이 비율"] = (X == 2).sum((1, 2)) / np.maximum((X > 0).sum((1, 2)), 1)

    print("== 운영 지점 (공식 test 118,595장, 정상 110,701 / 결함 7,894) ==")
    print("%-22s %7s %10s %10s %12s %10s" % (
        "arm", "재현율", "precision", "탐지", "헛경보(정상)", "문턱 동점"))
    table = {}
    for name, s in scores.items():
        rows = operating_points(s, is_def, RECALLS)
        table[name] = rows
        for r in rows:
            print("%-22s %7.0f%% %10.3f %10d %12d %10d"
                  % (name, 100 * r["target_recall"], r["precision"],
                     r["true_positives"], r["false_positives"],
                     r["n_tied_at_threshold"]))
        print()

    print("== 총합 지표와 동점 민감도 ==")
    print("%-22s %8s %8s %10s %22s" % ("arm", "AUROC", "AUPR", "고유값", "동점 무작위해소 AUPR"))
    sens = {}
    for name, s in scores.items():
        m = evaluate_ood(s, is_def)
        lo, hi, med = tie_sensitivity(s, is_def)
        sens[name] = {"aupr_det": m["aupr"], "tie_lo": lo, "tie_hi": hi, "tie_med": med,
                      "n_unique": int(len(np.unique(s))), "auroc": m["auroc"]}
        print("%-22s %8.4f %8.4f %10d   [%.4f, %.4f] 폭 %.4f"
              % (name, m["auroc"], m["aupr"], len(np.unique(s)), lo, hi, hi - lo))

    print("\n창 크기는 운영 지점마다 우열이 갈린다 — 하나가 지배하지 않는다:")
    for r_target in RECALLS:
        best = max(((n, [x for x in table[n] if x["target_recall"] == r_target][0]["precision"])
                    for n in table if "국소 밀도" in n), key=lambda t: t[1])
        print("  재현율 %2.0f%% 에서 최고 precision: %s (%.3f)"
              % (100 * r_target, best[0], best[1]))

    (OUT / "operating_points.json").write_text(
        json.dumps({"operating_points": table, "tie_sensitivity": sens},
                   ensure_ascii=False, indent=2))
    print("\n저장 → %s" % (OUT / "operating_points.json"))
