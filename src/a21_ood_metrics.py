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


def aupr_blocked(score, label) -> float:
    """**동점을 블록으로 묶는 AUPR. 이것이 이 워크스트림의 관례다.**

    같은 점수를 받은 표본은 어떤 문턱으로도 서로 갈리지 않는다. 그러니 지표도
    그것들을 갈라서는 안 된다. **서로 다른 문턱 값에서만** 곡선 위의 점을 찍으면
    입력 순서와 무관한 하나의 값이 나온다.

    왜 필요한가: 창 3x3 밀도는 고유값이 108개뿐인데, `aupr` 의 결정론적 값 0.4986 이
    동점을 무작위로 해소한 범위 [0.4112, 0.4248] **바깥**이었다.
    동점 안에서 원래 인덱스 순서를 쓰는데 test 배열 위치와 결함 여부에 약한 상관이
    있어서(0.0116) **동점 처리를 통해 라벨이 샌다.**

    정보가 없는 점수(전부 같은 값)에서는 블록이 하나라 **정확히 유병률**이 나온다.
    동점이 없으면 `aupr` 과 같다.
    """
    score, label = _check(score, label)
    order = np.argsort(-score, kind="mergesort")
    s, l = score[order], label[order]
    tp = np.cumsum(l)
    tot = np.arange(1, len(l) + 1)
    last = np.flatnonzero(np.concatenate([s[1:] != s[:-1], [True]]))   # 각 블록의 끝
    prec = tp[last] / tot[last]
    rec = tp[last] / tp[-1]
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def subsample_auroc_null(pos_score, neg_score, n: int, n_rep: int = 2000,
                         seed: int = 0, observed: float | None = None) -> dict:
    """**같은 분포에서 양성을 n 장만 뽑으면 AUROC 이 얼마나 흔들리는가.**

    작은 클래스의 AUROC 을 그대로 믿지 않기 위한 귀무분포다.
    11차 사이클이 `val_unseen` 에서 Scratch 0.7835 (n=92) 를 관측했는데,
    `test_dev` 의 Scratch 는 510장이다. **510장에서 92장을 뽑아 재면 어떤 값이 나오는지**
    를 먼저 보지 않으면 "다른 분포라 낮다" 와 "n 이 작아 낮다" 를 못 가른다.

    이 프로젝트는 **n=2 로 판정했다가 세 번 뒤집혔다.** 작은 n 은 먼저 귀무에 대고 본다.

    - 양성만 **비복원**으로 뽑는다. 복원하면 동점이 생겨 AUROC 이 위로 치우친다.
    - 음성은 매번 전부 쓴다. 여기서 묻는 것은 양성 표본 크기의 효과뿐이다.
    - `observed` 를 주면 귀무에서 그 값 **아래**인 비율을 같이 돌려준다(판정 규칙용).
    """
    pos = np.asarray(pos_score, np.float64).ravel()
    neg = np.asarray(neg_score, np.float64).ravel()
    if n > len(pos):
        raise ValueError(f"양성이 {len(pos)}개뿐인데 {n}개를 뽑을 수 없다")
    if n < 1:
        raise ValueError(f"n 은 1 이상이어야 한다: {n}")
    lab = np.concatenate([np.zeros(len(neg), np.int64), np.ones(n, np.int64)])
    rng = np.random.default_rng(seed)
    vals = np.empty(n_rep, np.float64)
    for i in range(n_rep):
        take = rng.choice(len(pos), n, replace=False)
        vals[i] = auroc(np.concatenate([neg, pos[take]]), lab)
    out = {
        "n": int(n), "n_rep": int(n_rep),
        "values": vals,
        "mean": float(vals.mean()),
        "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
    }
    if observed is not None:
        out["observed"] = float(observed)
        out["frac_below_observed"] = float((vals < observed).mean())
    return out
