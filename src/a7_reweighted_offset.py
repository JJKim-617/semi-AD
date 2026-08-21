"""타겟 분포로 재가중한 뒤 클래스별 오프셋을 튜닝한다.

a6 에서 val 로 고른 오프셋이 test 를 악화시켰다(0.6705 -> 0.6554). 이론적 이유가 있다.
test 의 93.3% 가 none 이라 none 에서 오탐률이 조금만 생겨도 결함 클래스의 precision 이
붕괴한다. 따라서 test 에서 macro-F1 을 최대화하는 결정 규칙은 val 에서보다 결함 예측에
**더 보수적**이어야 하는데, val 에서 튜닝하면 정반대 방향으로 밀린다.

해법은 val 표본에 w_y = q_target(y) / p_val(y) 를 곱해 "타겟 분포에서의 macro-F1" 을
val 라벨만으로 근사하는 것이다. 타겟 사전확률은 모델이 라벨 없는 test 에 매긴 예측 빈도로
추정한다(classify-and-count). test 라벨은 쓰지 않는다.

한계. CC 추정은 분류기 편향을 그대로 물려받고, 이 데이터셋은 클래스 조건부 분포도
일부 변해서(같은 Scratch 가 train 은 median 52x52, test 는 31x31) 순수 label shift 가정이
깨져 있다. 그래서 이 경로는 만능이 아니라 "val 튜닝보다 나은가" 를 묻는 실험이다.
"""

from __future__ import annotations

import numpy as np


def estimate_target_prior_cc(logits: np.ndarray, num_classes: int,
                             floor: float = 0.5) -> np.ndarray:
    """classify-and-count. 라벨 없이 예측 빈도로 타겟 사전확률을 추정한다.

    라벨을 인자로 받지 않는다. 서명 자체가 test 라벨 미사용을 보장한다.
    """
    pred = np.asarray(logits).argmax(1)
    counts = np.bincount(pred, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = floor
    return counts / counts.sum()


def importance_weights(y: np.ndarray, target_prior: np.ndarray,
                       num_classes: int) -> np.ndarray:
    """표본별 가중치 w_i = q(y_i) / p_source(y_i).

    가중치를 적용한 뒤 클래스별 질량 비율이 target_prior 와 일치한다.
    """
    y = np.asarray(y).ravel()
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    source = np.where(counts > 0, counts, 1.0)
    source = source / source.sum()
    w = np.asarray(target_prior, dtype=np.float64) / source
    return w[y]
