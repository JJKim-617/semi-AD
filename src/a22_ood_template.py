"""O1 — 정상 통계 template 잔차. 학습 없음(닫힌 형태 계수 추정), CPU 전용.

## 발상

정상 웨이퍼만 모아 **다이 위치별 불량 확률 지도** p 를 만든다.
그러면 어떤 웨이퍼든 "정상이라면 여기가 이만큼 불량일 텐데" 라는 기대값이 생기고,
관측과의 차이가 이상 점수가 된다. 3D-AD 의 잔차 발상을 2D 명목형 격자로 옮긴 것이다.

## 왜 채점기가 둘인가

E0 에서 **불량 다이 개수만으로 AUROC 0.8167** 이 나왔다. 그래서 새 점수가 좋아 보여도
그게 개수 신호의 재포장일 수 있다. 두 채점기를 나눠 그것을 가른다.

- `score_nll` — 그대로의 Bernoulli 잔차. 개수와 구조가 섞여 있다.
- `score_llr` — **웨이퍼 자신의 불량률로 template 을 재척도한 뒤의 로그가능도비.**
  개수를 맞춘 뒤 "그 불량들이 어디에 놓였는가" 만 남는다.
  template 이 균일하면 이 점수는 **항등적으로 0** 이다(테스트로 고정).

## 왜 좌표계가 셋인가

기획서 §1: train 의 58.1% 가 한 dieSize 대역에 몰려 있고 test 의 45.9% 는 train 이
14.2% 뿐인 대역에 있다. 극좌표가 정확히 이 커버리지 구멍에서 무너졌다.
절대 격자 template(`PositionTemplate` on pad 캐시)은 같은 함정에 그대로 걸릴 수 있어
**리사이즈 좌표**와 **웨이퍼 자기 반경으로 정규화한 1-D 반경 빈**을 나란히 둔다.

`RadialTemplate` 은 극좌표 표현과 다르다. 격자를 다시 샘플링하지 않고
웨이퍼 자신의 die 최대 반경으로만 나누므로 크기 자유도가 남지 않는다
(`test_is_invariant_to_wafer_size` 로 박혀 있다).

## one-class 격리

template 은 **train & y==0 인 웨이퍼만** 본다. 결함이 한 장이라도 섞이면 예외를 던진다.
`assert_one_class` 를 fit 진입점마다 호출한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6


def assert_one_class(y) -> None:
    """결함 라벨이 섞였으면 즉시 멈춘다.

    학습 단계에서 결함을 본 순간 그 실험은 one-class 가 아니다.
    조용히 진행되면 나중에 수치만 남고 무효인 줄 모른다.
    """
    y = np.asarray(y)
    if y.size and int((y != 0).sum()):
        raise ValueError(
            "one-class 위반: template 추정에 결함 라벨 %d장이 섞였다" % int((y != 0).sum()))
    return None


@dataclass
class PositionTemplate:
    """셀 위치별 불량 확률. `p[i,j]`, 그 자리에 die 가 있었던 횟수 `die_count`."""

    p: np.ndarray
    die_count: np.ndarray
    global_rate: float

    def cell_prob(self, xb: np.ndarray) -> np.ndarray:
        """(B,H,W) 배치에 대응하는 (B,H,W) 확률 지도."""
        return np.broadcast_to(self.p[None], xb.shape)


@dataclass
class RadialTemplate:
    """정규화 반경 빈별 불량 확률. 빈 경계는 [0,1] 등간격."""

    p: np.ndarray
    die_count: np.ndarray
    global_rate: float

    @property
    def n_bins(self) -> int:
        return len(self.p)

    def cell_prob(self, xb: np.ndarray) -> np.ndarray:
        idx = _radial_bin_index(xb, self.n_bins)
        out = np.where(idx >= 0, self.p[np.maximum(idx, 0)], self.global_rate)
        return out.astype(np.float64)


def _smoothed(n_fail: np.ndarray, n_die: np.ndarray, smoothing: float,
              global_rate: float) -> np.ndarray:
    """Laplace 평활 후 [EPS, 1-EPS] 로 자른다.

    die 가 한 번도 없던 셀은 0 이나 NaN 이 아니라 **전역 불량률**로 되돌린다.
    큰 test 웨이퍼는 train 이 밟지 않은 바깥 셀을 반드시 밟기 때문에,
    거기서 0 을 주면 log(0) 으로 점수가 발산한다.
    """
    p = (n_fail + smoothing) / np.maximum(n_die + 2.0 * smoothing, EPS)
    p = np.where(n_die > 0, p, global_rate)
    return np.clip(p, EPS, 1.0 - EPS)


def fit_position_template(x, y=None, smoothing: float = 1.0,
                          chunk: int = 2000) -> PositionTemplate:
    """정상 웨이퍼 묶음에서 셀 위치별 불량 확률을 센다."""
    if y is not None:
        assert_one_class(y)
    x = np.asarray(x)
    h, w = x.shape[1], x.shape[2]
    n_die = np.zeros((h, w), np.float64)
    n_fail = np.zeros((h, w), np.float64)
    for a in range(0, len(x), chunk):
        xb = x[a:a + chunk]
        n_die += (xb > 0).sum(0)
        n_fail += (xb == 2).sum(0)
    total_die = float(n_die.sum())
    rate = float(n_fail.sum() / max(total_die, 1.0))
    return PositionTemplate(
        p=_smoothed(n_fail, n_die, smoothing, rate), die_count=n_die, global_rate=rate)


def _radial_bin_index(xb: np.ndarray, n_bins: int) -> np.ndarray:
    """웨이퍼마다 **자기 die 영역**으로 중심과 최대 반경을 정해 정규화 반경 빈을 준다.

    die 가 아닌 칸은 -1. 크기 정규화가 여기 한 줄에 들어 있다 — 나누는 값이
    웨이퍼 자신의 최대 반경이므로 물리적 크기가 점수에 남지 않는다.
    """
    xb = np.asarray(xb)
    b, h, w = xb.shape
    die = xb > 0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    n = np.maximum(die.sum((1, 2)), 1).astype(np.float64)
    cy = (die * yy).sum((1, 2)) / n
    cx = (die * xx).sum((1, 2)) / n
    r = np.sqrt((yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2)
    rmax = np.where(die, r, -np.inf).max((1, 2))
    rmax = np.maximum(rmax, EPS)[:, None, None]
    u = np.clip(r / rmax, 0.0, 1.0 - 1e-12)
    idx = (u * n_bins).astype(np.int64)
    return np.where(die, idx, -1)


def fit_radial_template(x, y=None, n_bins: int = 32, smoothing: float = 1.0,
                        chunk: int = 2000) -> RadialTemplate:
    """정규화 반경 빈별 불량 확률. 크기 불변이며 빈당 표본이 많아 분산이 작다."""
    if y is not None:
        assert_one_class(y)
    x = np.asarray(x)
    n_die = np.zeros(n_bins, np.float64)
    n_fail = np.zeros(n_bins, np.float64)
    for a in range(0, len(x), chunk):
        xb = x[a:a + chunk]
        idx = _radial_bin_index(xb, n_bins)
        m = idx >= 0
        n_die += np.bincount(idx[m], minlength=n_bins).astype(np.float64)
        f = m & (xb == 2)
        n_fail += np.bincount(idx[f], minlength=n_bins).astype(np.float64)
    rate = float(n_fail.sum() / max(n_die.sum(), 1.0))
    return RadialTemplate(
        p=_smoothed(n_fail, n_die, smoothing, rate), die_count=n_die, global_rate=rate)


def uniform_template(rate: float, shape=(64, 64)) -> PositionTemplate:
    """대조군 — template 이 아무 구조도 담지 않은 조건.

    이때 `score_nll` 은 불량 개수와 die 개수의 선형결합으로 붕괴하고
    `score_llr` 은 0 이 된다. T-* 가 이것을 못 넘으면 template 이 한 일이 없다.
    """
    return PositionTemplate(
        p=np.full(shape, float(rate)), die_count=np.ones(shape), global_rate=float(rate))


def _prep(x, template):
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[None]
    die = x > 0
    fail = (x == 2).astype(np.float64)
    q = np.clip(np.asarray(template.cell_prob(x), np.float64), EPS, 1.0 - EPS)
    return die, fail, q


def score_nll(x, template, chunk: int = 2000) -> np.ndarray:
    """template 아래 관측 패턴의 Bernoulli 음의 로그가능도 합. 개수와 구조가 섞여 있다."""
    x = np.asarray(x)
    out = np.empty(len(x), np.float64)
    for a in range(0, len(x), chunk):
        die, fail, q = _prep(x[a:a + chunk], template)
        d = die.astype(np.float64)
        nll = -(fail * np.log(q) + (d - fail) * np.log1p(-q))
        out[a:a + len(nll)] = nll.sum((1, 2))
    return out


def score_llr(x, template, chunk: int = 2000) -> np.ndarray:
    """**개수를 맞춘 뒤의** 구조 점수.

    웨이퍼 자신의 불량률 r 로 template 을 재척도해 `q1 = q * r / mean_die(q)` 를 만들고,
    구조 없는 귀무 `q0 = r` 과의 로그가능도비를 뒤집어 점수로 쓴다.
    "불량이 몇 개인가" 는 두 가설이 공유하므로 상쇄되고 "어디에 놓였는가" 만 남는다.
    template 이 균일하면 q1 == q0 이라 정확히 0 이다.
    """
    x = np.asarray(x)
    out = np.empty(len(x), np.float64)
    for a in range(0, len(x), chunk):
        die, fail, q = _prep(x[a:a + chunk], template)
        d = die.astype(np.float64)
        n_die = np.maximum(d.sum((1, 2)), 1.0)
        r = fail.sum((1, 2)) / n_die
        qbar = np.maximum((q * d).sum((1, 2)) / n_die, EPS)
        q0 = np.clip(r, EPS, 1.0 - EPS)[:, None, None]
        q1 = np.clip(q * (r / qbar)[:, None, None], EPS, 1.0 - EPS)
        llr = fail * (np.log(q1) - np.log(q0)) + (d - fail) * (np.log1p(-q1) - np.log1p(-q0))
        out[a:a + len(llr)] = -llr.sum((1, 2))
    return out
