"""LOO 판정선을 **잡음 구조에서** 유도한다 (18차 마무리, 학습 없음).

## 왜 이 도구가 필요한가

E20(밀도), E21(선 필터), E22(백본 x translate)가 전부 **군 평균 LOO >= +0.003**
에서 기각됐다. 그런데 18차가 잰 것:

- 지금 풀에서 구성원 하나의 평균 기여가 **+0.0002** 다(k=10~25 크기 곡선).
- 개별 LOO 는 **seed 잡음이 지배**한다(E22 9개에서 seed 내 sd 0.0018 > 백본 간 0.0011).
- **+0.003 은 풀이 8개이던 E14 시절에 정한 상수**다.

**자가 고장 났으면 그 자로 잰 판정이 무엇을 뜻하는지 알 수 없다.**
기각이 "값어치가 없어서" 인지 "선이 도달 불가능해서" 인지 구분이 안 된다.

## 유도 순서 (뒤집으면 안 된다)

1. **먼저** 군 평균 LOO 의 **귀무분포**를 잰다.
2. **그 다음** 그 분포에서 선을 정한다.

**"과거 기각을 통과시키는 값" 을 역산하지 않는다.**

## 귀무가설과 왜 치환이 공짜인가

귀무가설은 *"이 군의 구성원이 풀의 나머지와 교환 가능하다"* 이다.

**구성원별 LOO 는 그 구성원이 어느 군에 속하는지와 무관하게 계산된다** —
풀의 조합에서만 나온다. 그러므로 치환은 **라벨만 바꾸면 되고 LOO 를 다시 계산할
필요가 없다.** 군 평균 LOO 의 귀무분포는 곧
**풀의 구성원 LOO 값에서 g 개를 비복원 추출한 평균의 분포**다.

학습도, 로짓 재계산도, 조합 재열거도 필요 없다.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

MAX_EXACT = 300_000        # 이보다 조합이 많으면 몬테카를로로 바꾼다


def null_sd_of_group_mean(loo_values, g: int) -> float:
    """크기 g 인 군의 평균 LOO 가 귀무가설에서 얼마나 흔들리나.

    유한모집단 비복원 추출이므로 **유한모집단 보정**이 들어간다.

        sd(mean) = sigma / sqrt(g) * sqrt((N - g) / (N - 1))

    `sigma` 는 풀의 구성원 LOO 의 모표준편차(ddof=0)다.
    **g = N 이면 0 이다** — 풀 전체를 군으로 잡으면 뽑을 것이 없고 판정이 불가능하다.
    """
    v = np.asarray(loo_values, dtype=np.float64)
    n = len(v)
    if not 1 <= g <= n:
        raise ValueError(f"군 크기 {g} 가 풀 크기 {n} 을 벗어난다")
    if n < 2:
        return 0.0
    sigma = v.std(ddof=0)
    return float(sigma / math.sqrt(g) * math.sqrt((n - g) / (n - 1)))


def null_resolution(n: int, g: int) -> tuple:
    """그 풀에서 **가능한 결과가 몇 가지이고 가장 작은 p 값이 얼마인가.**

    C(N,g) 가 작으면 검정의 해상도 자체가 없다 — alpha=0.05 를 못 만드는 경우도 있다.
    """
    total = math.comb(n, g)
    return total, 1.0 / total


def _null_means(loo_values, g: int, rng=None, n_mc: int = 200_000):
    """귀무분포 표본. 조합이 적으면 **전수**, 많으면 몬테카를로."""
    v = np.asarray(loo_values, dtype=np.float64)
    n = len(v)
    total = math.comb(n, g)
    if total <= MAX_EXACT:
        return np.fromiter((v[list(c)].mean()
                            for c in itertools.combinations(range(n), g)),
                           dtype=np.float64, count=total), True
    rng = rng or np.random.default_rng(0)
    idx = np.argsort(rng.random((n_mc, n)), axis=1)[:, :g]
    return v[idx].mean(1), False


def required_threshold(loo_values, g: int, alpha: float = 0.05) -> float:
    """`p <= alpha` 가 되는 **가장 낮은 관측값** — 이 값 이상이어야 우연이 아니다.

    **단측**이다. 기여는 방향이 있는 양이고 양측을 쓰면 선이 불필요하게 높아진다.

    **보간 분위수를 쓰지 않는다.** 귀무분포가 C(N,g) 개짜리 **이산** 분포라
    보간하면 `obs >= 선` 인데 `p > alpha` 인 모순이 생긴다(실제로 겪었다).
    정의를 p 값과 일치시킨다 — 선을 넘는 것과 p <= alpha 가 **동치**가 된다.

    **풀이 작으면 이 선이 최댓값이 되어 버린다.** C(7,3)=35 이면 가장 큰 하나만
    p=1/35=0.029 <= 0.05 이므로, **35가지 중 1등이 아니면 통과할 수 없다.**
    그 사실 자체가 "그 풀에서는 판정할 해상도가 없다" 는 신호다.
    """
    v = np.asarray(loo_values, dtype=np.float64)
    if len(v) < 2 or g >= len(v):
        return float("inf")
    means, _ = _null_means(v, g)
    srt = np.sort(means)
    m = len(srt)
    # P(null >= srt[k]) = (m - k) / m  (동점 무시). 그것이 alpha 이하인 최소 k.
    k = int(np.ceil(m * (1.0 - alpha)))
    if k >= m:
        # **가장 큰 값조차 p > alpha 다. 그 풀에서는 어떤 결과도 유의할 수 없다.**
        # 클램프해서 도달 가능한 척하면 안 된다 — 무한대가 정직한 답이다.
        return float("inf")
    return float(srt[k])


def permutation_p_two_sided(values, group_idx) -> float:
    """양측 치환 p — **"다른가" 를 물을 때 쓴다**(어느 쪽이 큰가가 아니라).

    p = P(|귀무 군평균 - 전체평균| >= |관측 군평균 - 전체평균|).
    부호를 뒤집어도 같은 값이 나온다.
    """
    v = np.asarray(values, dtype=np.float64)
    idx = list(group_idx)
    if any(i < 0 or i >= len(v) for i in idx):
        raise ValueError("군 색인이 범위를 벗어난다")
    mu = v.mean()
    obs = abs(v[idx].mean() - mu)
    means, _ = _null_means(v, len(idx))
    return float((np.abs(means - mu) >= obs - 1e-15).mean())


def permutation_p(loo_values, group_idx) -> float:
    """관측된 군 평균 LOO 가 귀무분포에서 얼마나 흔한가 (단측 p 값).

    p = P(귀무 군 평균 >= 관측). 관측 자신을 포함해 센다.
    """
    v = np.asarray(loo_values, dtype=np.float64)
    idx = list(group_idx)
    if any(i < 0 or i >= len(v) for i in idx):
        raise ValueError("군 색인이 풀 범위를 벗어난다")
    obs = v[idx].mean()
    means, _ = _null_means(v, len(idx))
    return float((means >= obs - 1e-15).mean())


# --- 아래는 보고용. 위 셋만 시험이 있다 -------------------------------------

def leave_one_out_values(P, ty, pool, ks=(2, 3, 4)):
    """판정에 쓴 것과 **같은 방법**으로 구성원별 LOO 를 낸다.

    `bench_ensemble.leave_one_out_delta` 를 k 별로 돌려 평균한다 —
    E20, E21, E22, E24 판정이 전부 이 경로였다.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from a4_eval_wm811k_cls import evaluate
    from bench_ensemble import leave_one_out_delta

    scores = {}
    for k in ks:
        if k > len(pool):
            break
        for c in itertools.combinations(pool, k):
            scores[c] = float(evaluate(
                ty, np.mean([P[t] for t in c], 0).argmax(1), 9)["macro_f1"])
    return {t: float(np.mean([d["delta"] for d in
                              leave_one_out_delta(scores, t).values()]))
            for t in pool}
