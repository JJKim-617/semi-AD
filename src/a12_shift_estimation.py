"""타겟 사전확률을 BBSE 로 추정하고, 그것으로 선택 지표를 고친다.

이 프로젝트가 8 사이클 동안 쓴 자(尺)가 망가져 있었다. val macro-F1 은 test 순위를
못 맞힌다 — 캐시된 수렴 실행 5개에서 Spearman +0.100 이고, 같은 설정 3 seed 에서는
val 1위가 test 꼴찌였다. val 이 train 분포(결함 31.9%)를 물려받는데 test 는 6.7% 라
두 분포에서 좋은 모델이 서로 다르기 때문이다.

val 표본에 w[y] = q_test[y]/p_val[y] 를 걸면 순위가 맞는다(oracle q 로 +0.900).
문제는 q 를 라벨 없이 추정하는 부분이다. a7 의 classify-and-count 는 분류기 자신의
편향을 물려받아 Edge-Loc 을 1.93배, Scratch 를 1.58배 과대추정했고 순위가 전혀 안 맞았다.

BBSE(Lipton et al. 2018)는 val 에서 잰 혼동행렬로 그 편향을 되돌린다.

    C[i,j] = P_val(pred=i, y=j)      val 라벨 사용
    mu[i]  = P_test(pred=i)          test 라벨 미사용
    C w = mu 를 풀고  q[j] = w[j] * p_val[j]

CC 는 C 가 단위행렬일 때(분류기가 완벽할 때)의 특수한 경우다.

한계. label shift 가정(P(x|y)가 불변)에 기댄다. 이 데이터셋은 같은 Scratch 가 train 은
median 52x52, test 는 31x31 이라 가정이 일부 깨져 있다. 그래서 항상 oracle q 결과를
함께 내서 "추정 오차" 와 "가정 위반" 을 분리한다.
"""

from __future__ import annotations

import numpy as np


def total_variation(p, q) -> float:
    """두 분포의 총변동거리. 같으면 0, 서로소면 1."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.abs(p - q).sum() / 2.0)


def estimate_target_prior_bbse(val_pred: np.ndarray, val_y: np.ndarray,
                               target_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """BBSE. 타겟 라벨을 받지 않는다 — 서명이 그것을 보장한다.

    val 의 혼동행렬로 분류기 오류율을 보정하므로, CC 와 달리 분류기가 편향돼 있어도
    참 사전확률을 되찾는다.
    """
    val_pred = np.asarray(val_pred).ravel()
    val_y = np.asarray(val_y).ravel()
    target_pred = np.asarray(target_pred).ravel()

    n = len(val_y)
    C = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(C, (val_pred, val_y), 1.0)
    C /= n

    mu = np.bincount(target_pred, minlength=num_classes).astype(np.float64)
    mu /= mu.sum()

    with np.errstate(all="ignore"):
        if np.linalg.matrix_rank(C) < num_classes:
            w = np.linalg.lstsq(C, mu, rcond=None)[0]
        else:
            w = np.linalg.solve(C, mu)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    p_val = np.bincount(val_y, minlength=num_classes).astype(np.float64)
    p_val /= p_val.sum()

    q = np.clip(w * p_val, 0.0, None)
    if q.sum() <= 0:
        return np.full(num_classes, 1.0 / num_classes)
    return q / q.sum()


def blend_prior(defect_rate: float, composition) -> np.ndarray:
    """결함률 하나와 결함 내부 구성으로 전체 사전확률을 만든다. 클래스 0 이 정상이다."""
    if not 0.0 <= defect_rate <= 1.0:
        raise ValueError(f"결함률이 [0,1] 밖이다: {defect_rate}")
    comp = np.asarray(composition, dtype=np.float64)
    comp = comp / comp.sum()
    q = np.empty(len(comp) + 1, dtype=np.float64)
    q[0] = 1.0 - defect_rate
    q[1:] = defect_rate * comp
    return q


def estimate_defect_rate(val_pred: np.ndarray, val_y: np.ndarray,
                         target_pred: np.ndarray) -> float:
    """정상/결함 이진 축으로만 BBSE 를 건다. 타겟 라벨을 받지 않는다.

    9차원 BBSE 는 실패했다. 가중치가 q/p_val 이라 문제가 되는 것은 절대오차가 아니라
    비율오차인데, 참 비율이 0.0008 인 클래스에서는 작은 절대오차가 큰 비율오차가 된다.
    이진으로 축소하면 두 셀에 표본이 수천 장씩 있어 그 문제가 사라진다.

    라벨은 다중 클래스로 줘도 된다. 0 이 아니면 결함으로 접는다.
    """
    to_bin = lambda a: (np.asarray(a).ravel() != 0).astype(np.int64)
    q = estimate_target_prior_bbse(to_bin(val_pred), to_bin(val_y),
                                   to_bin(target_pred), num_classes=2)
    return float(np.clip(q[1], 0.0, 1.0))


def shifted_val_macro_f1(logits: np.ndarray, y: np.ndarray, target_prior: np.ndarray,
                         num_classes: int, metric: str = "macro_f1") -> float:
    """타겟 분포에서의 macro-F1 을 val 라벨만으로 근사한다."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from a4_eval_wm811k_cls import evaluate
    from a7_reweighted_offset import importance_weights

    w = importance_weights(y, target_prior, num_classes)
    pred = np.asarray(logits).argmax(1)
    return float(evaluate(y, pred, num_classes, sample_weight=w)[metric])


def rank_by_shifted_val(runs: dict, target_prior: np.ndarray,
                        num_classes: int) -> list:
    """{tag: (val_logits, val_y)} 를 재가중 val 점수 내림차순으로 정렬한다."""
    scored = [(tag, shifted_val_macro_f1(lg, y, target_prior, num_classes))
              for tag, (lg, y) in runs.items()]
    return sorted(scored, key=lambda kv: -kv[1])


def spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else 0.0


def main() -> None:
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from a4_eval_wm811k_cls import evaluate
    from a7_reweighted_offset import estimate_target_prior_cc

    p = argparse.ArgumentParser(
        description="선택 지표 비교: 현행 val vs 재가중 val (CC / BBSE / oracle)")
    p.add_argument("--logits", nargs="+", required=True,
                   help="result/posthoc/<tag>_logits.npz 들")
    p.add_argument("--num-classes", type=int, default=9)
    p.add_argument("--out", default="result/posthoc/selection_metric.json")
    a = p.parse_args()
    nc = a.num_classes

    rows = []
    for path in a.logits:
        d = np.load(path)
        tag = Path(path).stem.replace("_logits", "")
        vy, vl = d["val_y"], d["val_logits"]
        ty, tl = d["test_y"], d["test_logits"]

        p_val = np.bincount(vy, minlength=nc).astype(np.float64); p_val /= p_val.sum()
        q_true = np.bincount(ty, minlength=nc).astype(np.float64); q_true /= q_true.sum()
        q_cc = estimate_target_prior_cc(tl, nc)
        q_bbse = estimate_target_prior_bbse(vl.argmax(1), vy, tl.argmax(1), nc)

        rows.append(dict(
            tag=tag,
            test=float(evaluate(ty, tl.argmax(1), nc)["macro_f1"]),
            val_plain=float(evaluate(vy, vl.argmax(1), nc)["macro_f1"]),
            val_cc=shifted_val_macro_f1(vl, vy, q_cc, nc),
            val_bbse=shifted_val_macro_f1(vl, vy, q_bbse, nc),
            val_oracle=shifted_val_macro_f1(vl, vy, q_true, nc),
            tv_cc=total_variation(q_cc, q_true),
            tv_bbse=total_variation(q_bbse, q_true),
        ))

    print("%-20s %8s %9s %9s %9s %9s" % (
        "run", "test", "val", "val+CC", "val+BBSE", "val+oracle"))
    for r in rows:
        print("%-20s %8.4f %9.4f %9.4f %9.4f %9.4f" % (
            r["tag"], r["test"], r["val_plain"], r["val_cc"], r["val_bbse"], r["val_oracle"]))

    t = [r["test"] for r in rows]
    print()
    print("사전확률 추정 오차 (총변동거리, 작을수록 좋다)")
    print("  CC   평균 %.4f" % np.mean([r["tv_cc"] for r in rows]))
    print("  BBSE 평균 %.4f" % np.mean([r["tv_bbse"] for r in rows]))
    print()
    print("test 순위와의 Spearman 상관 (n=%d)" % len(rows))
    summary = {}
    for key, label in [("val_plain", "현행 val"), ("val_cc", "재가중 val + CC"),
                       ("val_bbse", "재가중 val + BBSE"), ("val_oracle", "재가중 val + oracle")]:
        v = [r[key] for r in rows]
        rho = spearman(v, t)
        summary[key] = rho
        best = rows[int(np.argmax(v))]
        print("  %-22s rho = %+.3f   고르는 실행 %s (test %.4f)" % (
            label, rho, best["tag"], best["test"]))
    print("  %-22s        실제 1위 %s (test %.4f)" % (
        "", rows[int(np.argmax(t))]["tag"], max(t)))

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "spearman": summary}, indent=2,
                              ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
