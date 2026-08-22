"""one-class 이상탐지 지표와 자명한 스칼라 채점기.

## 왜 AUPR 이 주 지표인가

9-class 에서는 macro-F1 이었다. one-class 는 사정이 다르다.

- **AUROC 는 유병률에 불변**이라 9-class 때의 accuracy 함정(자명한 분류기가 0.9334)과
  성질이 다르다. 그러나 **절대 헛경보 부담을 반영하지 못한다** — test 의 정상이
  110,701장이라 FPR 1% 도 1,107장이다.
- **AUPR 은 유병률에 직접 묶인다.** 무작위 기준선이 곧 결함 비율(0.0666)이고,
  어떤 재현율에서 정밀도가 얼마인지가 실사용 가능성을 정한다.
- **FPR@95TPR 은 팹 언어다.** "결함의 95% 를 잡으려면 정상 몇 %를 버리는가."

그래서 주 지표 = AUPR, 함께 보고 = FPR@95TPR, AUROC.

## 바닥은 무작위가 아니다

E0 실측: 학습 없이 **불량 다이를 세기만 해도 AUROC 0.8167 / AUPR 0.3633**,
불량 다이 비율은 AUPR 0.3856 이다. 어떤 방법이든 이걸 넘어야 의미가 있다.
Near-full(1.000), Random(0.997), Donut(0.984)은 개수만으로 이미 갈리고,
남은 값어치는 Scratch(0.656), Center(0.754), Loc(0.810) 에 있다.
"""

from __future__ import annotations

import numpy as np


def _check(score: np.ndarray, label: np.ndarray):
    score = np.asarray(score, dtype=np.float64).ravel()
    label = np.asarray(label).ravel()
    if not np.isin(label, (0, 1)).all():
        raise ValueError("label 은 0/1 이진이어야 한다")
    if label.sum() == 0 or label.sum() == len(label):
        raise ValueError("한 쪽 클래스만 있으면 AUROC/AUPR 이 정의되지 않는다")
    if len(score) != len(label):
        raise ValueError(f"길이가 다르다: score {len(score)} vs label {len(label)}")
    return score, label.astype(np.int64)


def auroc(score, label) -> float:
    """동점을 평균 랭크로 처리하는 Mann-Whitney U 기반 AUROC.

    동점 처리를 틀리면 정보 없는 점수(전부 같은 값)가 0.5 가 아니게 나온다.
    """
    score, label = _check(score, label)
    order = np.argsort(score, kind="mergesort")
    s, l = score[order], label[order]
    ranks = np.empty(len(s), dtype=np.float64)
    bounds = np.flatnonzero(np.concatenate([[True], s[1:] != s[:-1], [True]]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        ranks[a:b] = (a + b - 1) / 2.0 + 1.0
    n1 = float(l.sum())
    n0 = float(len(l) - n1)
    return float((ranks[l == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def aupr(score, label) -> float:
    """Average precision. 재현율 증분 x 정밀도의 합(계단 적분)."""
    score, label = _check(score, label)
    order = np.argsort(-score, kind="mergesort")
    l = label[order]
    tp = np.cumsum(l)
    precision = tp / np.arange(1, len(l) + 1)
    recall = tp / l.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def fpr_at_tpr(score, label, target: float = 0.95) -> float:
    """목표 재현율을 처음 만족하는 지점의 위양성률."""
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target 은 (0,1] 이어야 한다: {target}")
    score, label = _check(score, label)
    order = np.argsort(-score, kind="mergesort")
    l = label[order]
    tpr = np.cumsum(l) / l.sum()
    fpr = np.cumsum(1 - l) / float(len(l) - l.sum())
    idx = int(np.searchsorted(tpr, target))
    return float(fpr[min(idx, len(fpr) - 1)])


def evaluate_ood(score, label, tpr_target: float = 0.95) -> dict:
    """세 지표와 함께 **AUPR 의 바닥(유병률)** 을 항상 같이 돌려준다.

    유병률을 빼고 AUPR 만 보면 0.38 이 좋은 값인지 알 수 없다.
    """
    score, label = _check(score, label)
    return {
        "auroc": auroc(score, label),
        "aupr": aupr(score, label),
        "fpr_at_95tpr": fpr_at_tpr(score, label, tpr_target),
        "prevalence": float(label.mean()),
        "n": int(len(label)),
        "n_anomaly": int(label.sum()),
    }


# --- 자명한 스칼라 채점기 (학습 없음) ---------------------------------------

def fail_count(x: np.ndarray) -> np.ndarray:
    """웨이퍼당 불량 다이 개수. 셀 값 2 만 센다."""
    return (np.asarray(x) == 2).sum(axis=(1, 2)).astype(np.float64)


def fail_ratio(x: np.ndarray) -> np.ndarray:
    """불량 다이 / 전체 다이. **다이가 없는 칸(0)은 분모에서 뺀다.**

    픽셀 수로 나누면 웨이퍼 크기가 다를 때 작은 웨이퍼가 조직적으로 낮게 나온다.
    """
    x = np.asarray(x)
    n_die = (x > 0).sum(axis=(1, 2)).astype(np.float64)
    n_fail = (x == 2).sum(axis=(1, 2)).astype(np.float64)
    return n_fail / np.maximum(n_die, 1.0)


# --- 운영 지점 -----------------------------------------------------------------

def operating_points(score, label, recalls=(0.5, 0.8, 0.95)) -> list:
    """목표 재현율마다 문턱, precision, 헛경보 장수를 돌려준다.

    **AUPR 총합은 어디서 무너지는지를 가린다.** k=7 국소 밀도는 AUPR 0.7145 인데
    결함 50% 를 잡을 때 precision 0.862, 80% 를 잡을 때 0.357 이다 —
    그 사이에서 반토막 난다. 팹이 실제로 감당하는 것은 그 숫자이지 곡선 아래 넓이가 아니다.

    `n_tied_at_threshold` 를 같이 낸다. 동점이 많으면 **운영 지점이 한 점이 아니라
    구간**이 되고, precision 을 한 숫자로 읽으면 안 된다.
    창 3x3 밀도는 고유값이 24개뿐이라 문턱 하나가 수만 장을 한꺼번에 넘긴다.
    """
    score, label = _check(score, label)
    for t in recalls:
        if not 0.0 < t <= 1.0:
            raise ValueError(f"목표 재현율은 (0,1] 이어야 한다: {t}")
    order = np.argsort(-score, kind="mergesort")
    s, l = score[order], label[order]
    tp = np.cumsum(l)
    fp = np.cumsum(1 - l)
    n_pos = int(l.sum())
    rec = tp / n_pos
    out = []
    for t in recalls:
        i = int(np.searchsorted(rec, t))
        i = min(i, len(rec) - 1)
        thr = float(s[i])
        out.append({
            "target_recall": float(t),
            "threshold": thr,
            "recall": float(rec[i]),
            "precision": float(tp[i] / (tp[i] + fp[i])),
            "true_positives": int(tp[i]),
            "false_positives": int(fp[i]),
            "false_negatives": int(n_pos - tp[i]),
            "n_tied_at_threshold": int((score == thr).sum()),
        })
    return out
