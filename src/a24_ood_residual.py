"""다이 단위 잔차 맵과 맵→스칼라 뭉개기. 학습 없음, CPU 전용.

## 왜 맵이 1급 산출물인가

기획서 §9: OOD 는 이제 파이프라인 1단계이고 하류는
**국소화 → MLLM 8종 분류 → 유저 질의응답** 이다.
"어디에 문제가 있어?" 에 답하려면 넘겨줄 것이 스칼라가 아니라 **다이별 맵**이어야 한다.

`a22_ood_template` 의 두 채점기는 사실 맵의 합이다. 여기서 그 합을 풀어
셀별 기여를 그대로 내놓는다. **맵을 더하면 정확히 원래 스칼라가 나온다**는 것을
테스트로 박아 둔다(`test_a24_ood_residual.py`) — 그게 깨지면 하류가 보는 것과
우리가 보고한 AUPR 이 서로 다른 것을 재게 된다.

## 픽셀 단위 정답은 만들지 않는다

WM-811K 에는 다이 단위 마스크가 없다. `failureType` 은 웨이퍼당 문자열 하나이고
셀 값 `2` 는 "떨어진 다이"이지 "결함 패턴에 속한 다이" 가 아니다.
정답을 합성해 P-AUROC 를 내면 모델이 아니라 **합성 규칙**을 재게 된다(기획서 §9.3).
그래서 이 모듈은 국소화 품질을 **직접 재지 않는다**. 맵을 만들고, 뭉개고, 저장할 뿐이다.
국소화 판정은 정성 시각화와 하류 8종 macro-F1(간접)으로만 한다.

## 뭉개기는 하이퍼파라미터가 아니라 설계 선택이다

같은 맵이라도 무엇을 보느냐에 따라 다른 결함이 보인다.

| 연산자 | 무엇을 재는가 | 어떤 결함을 노리는가 |
|---|---|---|
| `sum` | 전체 이상량 | 개수형 (Near-full, Random) |
| `mean` | 다이당 평균 이상량 | 크기 정규화된 개수형 |
| `max` | 가장 이상한 다이 하나 | 국소 극점. 잡음에 약하다 |
| `topk_mean` | 상위 k% 평균 | 작은 덩어리. max 보다 잡음에 강하다 |
| `smoothed_max` | kxk 이웃 평균의 최대 | **뭉침**. Loc, Center |
| `max_component` | 이상 다이 최대 연결성분 크기 | **연결성**. Scratch, Loc |

`smoothed_max` 와 `max_component` 만이 **공간적 뭉침**을 본다. E0 에서 전역 2차 모멘트가
무작위보다 낮았던(0.4727) 이유가 뭉침을 전역 통계로 재려 했기 때문이라면,
국소 연산자는 다른 답을 줘야 한다. 그게 이 표의 가설이다.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from a22_ood_template import EPS

CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


def residual_map(x, template, mode: str = "llr", chunk: int = 2000) -> np.ndarray:
    """다이별 이상 기여도 (B,H,W) float64. 다이가 없는 칸은 정확히 0.

    - `mode="nll"` — 셀별 Bernoulli 음의 로그가능도. 합 = `score_nll`.
    - `mode="llr"` — 웨이퍼 자신의 불량률로 재척도한 뒤의 로그가능도비 기여.
      합 = `score_llr`. **개수를 맞춘 뒤 "어디에 놓였는가" 만 남는다.**

    다이가 없는 칸을 0 으로 두는 것이 중요하다. 값이 새면 시각화와 하류 프롬프트가
    웨이퍼 바깥을 가리킨다.
    """
    if mode not in ("nll", "llr"):
        raise ValueError("mode 는 'nll' 또는 'llr': %r" % mode)
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[None]
    out = np.zeros(x.shape, np.float64)
    for a in range(0, len(x), chunk):
        xb = x[a:a + chunk]
        die = (xb > 0).astype(np.float64)
        fail = (xb == 2).astype(np.float64)
        q = np.clip(np.asarray(template.cell_prob(xb), np.float64), EPS, 1.0 - EPS)
        if mode == "nll":
            m = -(fail * np.log(q) + (die - fail) * np.log1p(-q))
        else:
            n_die = np.maximum(die.sum((1, 2)), 1.0)
            r = fail.sum((1, 2)) / n_die
            qbar = np.maximum((q * die).sum((1, 2)) / n_die, EPS)
            q0 = np.clip(r, EPS, 1.0 - EPS)[:, None, None]
            q1 = np.clip(q * (r / qbar)[:, None, None], EPS, 1.0 - EPS)
            m = -(fail * (np.log(q1) - np.log(q0))
                  + (die - fail) * (np.log1p(-q1) - np.log1p(-q0)))
        out[a:a + len(xb)] = m * die
    return out


# --- 뭉개기 연산자 -------------------------------------------------------------

def pool_sum(m, die):
    return np.asarray(m).sum((1, 2))


def pool_mean(m, die):
    d = np.asarray(die).sum((1, 2)).astype(np.float64)
    return np.asarray(m).sum((1, 2)) / np.maximum(d, 1.0)


def pool_max(m, die):
    m = np.where(np.asarray(die), np.asarray(m), -np.inf)
    v = m.max((1, 2))
    return np.where(np.isfinite(v), v, 0.0)


def pool_topk_mean(m, die, frac: float = 0.05):
    """다이 셀 중 상위 frac 비율의 평균. max 보다 잡음에 강하고 mean 보다 국소적이다.

    비율로 정하므로 웨이퍼 크기에 따라 셀 수가 자동으로 늘어난다.
    """
    m = np.asarray(m, np.float64)
    die = np.asarray(die)
    b = len(m)
    masked = np.where(die, m, -np.inf).reshape(b, -1)
    n_die = die.reshape(b, -1).sum(1)
    k = np.maximum((n_die * frac).astype(np.int64), 1)
    # 전체 정렬은 낭비다. 필요한 것은 상위 kmax 개뿐이라 부분 정렬로 자른다.
    kmax = int(min(max(k.max(), 1), masked.shape[1]))
    part = np.partition(masked, masked.shape[1] - kmax, axis=1)[:, -kmax:]
    order = np.sort(part, axis=1)[:, ::-1]
    idx = np.arange(order.shape[1])[None, :]
    take = idx < k[:, None]
    vals = np.where(take & np.isfinite(order), order, 0.0)
    return vals.sum(1) / np.maximum(k.astype(np.float64), 1.0)


def smooth_map(m, die, k: int = 3):
    """kxk 이웃 **평균** 맵. 창 안의 다이 개수로 나눠 가장자리 희석을 막는다.

    **이 맵 자체가 하류에 넘길 국소화 산출물이다.** 스칼라(`pool_smoothed_max`)는
    이 맵의 최대일 뿐이고, "어디에 문제가 있어?" 에 답하는 것은 맵 쪽이다.
    다이가 없는 칸은 0 으로 되돌려 웨이퍼 밖을 가리키지 않게 한다.
    """
    m = np.asarray(m, np.float64)
    d = np.asarray(die).astype(np.float64)
    num = ndimage.uniform_filter(m, size=(1, k, k), mode="constant")
    den = ndimage.uniform_filter(d, size=(1, k, k), mode="constant")
    sm = num / np.maximum(den, 1.0 / (k * k))
    return np.where(np.asarray(die), sm, 0.0)


def pool_smoothed_max(m, die, k: int = 3):
    """kxk 이웃 평균의 최대. 개수가 같아도 뭉쳐 있으면 커진다.

    **실측 결과 이 뭉개기가 O1 전체를 뒤집었다** — 자세한 것은
    `docs/experiments/candidate/ood_residual_pooling.md` 판정 절.
    """
    return pool_max(smooth_map(m, die, k), die)


def pool_max_component(m, die, threshold=None, connectivity=CONNECTIVITY_8):
    """이상으로 표시된 다이의 **최대 8-연결성분 크기**. Scratch, Loc 를 노린다.

    `threshold=None` 이면 **웨이퍼 자신의 다이 평균 잔차**를 문턱으로 쓴다.
    바깥에서 값을 정하면 그게 곧 튜닝 손잡이가 되고 template 척도에 따라 뜻이 달라진다.
    자기 평균은 척도 자유이며 웨이퍼별로 자동으로 맞춰진다.
    """
    m = np.asarray(m, np.float64)
    die = np.asarray(die)
    b = len(m)
    out = np.zeros(b, np.float64)
    if threshold is None:
        n_die = np.maximum(die.reshape(b, -1).sum(1), 1)
        thr = m.reshape(b, -1).sum(1) / n_die
    else:
        thr = np.full(b, float(threshold))
    flags = die & (m > thr[:, None, None])
    for i in range(b):
        if not flags[i].any():
            continue
        lab, n = ndimage.label(flags[i], structure=connectivity)
        if n:
            out[i] = float(np.bincount(lab.ravel())[1:].max())
    return out


def local_fail_density_map(x, k: int = 7, chunk: int = 2000) -> np.ndarray:
    """창 kxk 안 **불량 다이 비율** 맵. template 이 전혀 없다 — 학습이 0 이다.

    창 안의 다이 개수로 나누므로 웨이퍼 가장자리에서도 희석되지 않고,
    비율이라 웨이퍼 크기에 자동으로 불변이다.
    """
    x = np.asarray(x)
    out = np.zeros(x.shape, np.float64)
    for a in range(0, len(x), chunk):
        xb = x[a:a + chunk]
        die = (xb > 0).astype(np.float64)
        fail = (xb == 2).astype(np.float64)
        num = ndimage.uniform_filter(fail, size=(1, k, k), mode="constant")
        den = ndimage.uniform_filter(die, size=(1, k, k), mode="constant")
        out[a:a + len(xb)] = np.where(xb > 0, num / np.maximum(den, 1e-12), 0.0)
    return out


def local_fail_density_max(x, k: int = 7, chunk: int = 2000) -> np.ndarray:
    """위 맵의 최대. **이것이 이 워크스트림의 새 바닥이다.**

    O0 은 자명한 기준선으로 전역 스칼라만 쟀고 바닥을 AUPR 0.3856 으로 적었다.
    같은 정도로 자명한 **국소** 스칼라가 AUPR 0.7147 을 낸다(k=7).
    `tests/test_a24_local_density_pin.py` 에 박아 뒀다.
    """
    x = np.asarray(x)
    out = np.empty(len(x), np.float64)
    for a in range(0, len(x), chunk):
        xb = x[a:a + chunk]
        m = local_fail_density_map(xb, k=k, chunk=chunk)
        out[a:a + len(xb)] = pool_max(m, xb > 0)
    return out


POOLERS = {
    "sum": pool_sum,
    "mean": pool_mean,
    "max": pool_max,
    "top1%_mean": lambda m, d: pool_topk_mean(m, d, frac=0.01),
    "top5%_mean": lambda m, d: pool_topk_mean(m, d, frac=0.05),
    "smooth3_max": lambda m, d: pool_smoothed_max(m, d, k=3),
    "smooth5_max": lambda m, d: pool_smoothed_max(m, d, k=5),
    "max_component": pool_max_component,
}
