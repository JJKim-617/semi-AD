"""요소 가중 — **정상 데이터만으로** 결합 가중을 유도한다.

## 무엇을 고치는가

채택된 융합은 세 요소의 train-none 백분위를 **Fisher 결합**한다.
Fisher 의 결합 통계량은 **p 값들이 독립일 때만** 그 분포를 갖는다.
그런데 요소 A(k2 5x5 + 반경 대역 보정)와 요소 C(k2 3x3)는 **둘 다 k2 밀도의 최대값**이다 —
창 크기만 다르고 같은 것을 센다. **구조적으로 독립일 수 없다.**

독립이 아닌 것을 독립으로 취급하면 겹치는 신호를 두 번 센다.
동일 가중이므로 서로 닮은 A 와 C 가 사실상 **2표**를 갖고,
혼자 다른 것을 보는 B(온전창 선 L=11)가 **1표**를 갖는다.

## 어떻게 고치는가 — 손잡이를 만들지 않는다

백분위를 정규 분위수로 바꾸고(`probit`), **train-none 에서** 공분산 `S` 를 잰 뒤

    w ∝ S^-1 1,    s = w' z

를 쓴다. 이것은 **상관된 잡음 아래에서 공통 평균 이동을 검출하는 최적 선형 결합**이고,
가중이 **데이터에서 유도된다.** test 로 고를 자유도가 0 이다 —
`S` 는 정상 웨이퍼 하나에서만 나온다.

닮은 요소가 왜 깎이는지는 2x2 로 바로 보인다. `S = [[1,r],[r,1]]` 이면
`S^-1 1 = (1/(1+r))[1,1]` 이라 상관이 클수록 둘 다 눌린다.
A, C 만 상관되고 B 가 무상관이면 **B 만 안 눌린다.**

## 범위 선언

**범위 A 다.** 결함 데이터도 결함 모양 사전지식도 안 쓴다.
`fit_component_weights` 는 라벨을 받아 **정상만인지 확인하는 데만** 쓴다.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def probit(u) -> np.ndarray:
    """표준정규 분위수. **0 과 1 을 잘라 무한대를 막는다.**

    백분위는 `ecdf_percentile` 이 만드는 값이라 경계에 붙을 수 있고,
    한 표본이 무한대가 되면 점수 전체가 오염된다.
    """
    from scipy import special
    v = np.clip(np.asarray(u, np.float64), EPS, 1.0 - EPS)
    return np.sqrt(2.0) * special.erfinv(2.0 * v - 1.0)


def optimal_weights(cov) -> np.ndarray:
    """`w ∝ S^-1 1`, 합이 1 이 되게 정규화.

    `S` 가 특이하면(두 요소가 완전히 같으면) `pinv` 로 되돌린다.
    터지느니 가장 가까운 답을 주고, 그 사실을 호출부가 상관행렬로 볼 수 있게 한다.
    """
    S = np.asarray(cov, np.float64)
    one = np.ones(len(S))
    try:
        w = np.linalg.solve(S, one)
        if not np.all(np.isfinite(w)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(S) @ one
    s = w.sum()
    if not np.isfinite(s) or abs(s) < 1e-12:
        return np.full(len(S), 1.0 / len(S))
    return w / s


def fit_component_weights(z_train_none, y=None) -> dict:
    """**train-none 의 요소 z 행렬에서만** 가중을 적합한다.

    `y` 를 주면 전부 정상인지 확인한다. 결함이 하나라도 섞이면 예외 —
    그 순간 이 arm 은 범위 A 가 아니게 된다.
    """
    z = np.asarray(z_train_none, np.float64)
    if z.ndim != 2:
        raise ValueError(f"z 는 (n, 요소수) 여야 한다: {z.shape}")
    if y is not None:
        from a22_ood_template import assert_one_class
        assert_one_class(np.asarray(y))
    S = np.cov(z, rowvar=False)
    S = np.atleast_2d(S)
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    corr = S / np.outer(d, d)
    return {"cov": S, "corr": corr, "weights": optimal_weights(S), "n": int(len(z))}


def weighted_score(z, weights) -> np.ndarray:
    """`s = w' z`. 큰 값이 더 이상하다(z 가 백분위의 분위수라 그렇다)."""
    return np.asarray(z, np.float64) @ np.asarray(weights, np.float64)
