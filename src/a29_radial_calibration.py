"""정상 웨이퍼 자체를 귀무로 쓰는 밀도 보정. 정상만 학습, CPU 전용.

## 왜

3차 사이클 실측: **정상 웨이퍼의 바깥 링에는 연속으로 죽은 다이가 줄줄이 있다.**
길이 7 직선 창이 정상의 70.21% 에서 포화하고 그 95.49% 가 반경 0.85 바깥이다.

즉 **밀도 0.6 의 뜻이 안쪽과 가장자리에서 다르다.** 가장자리에서는 흔하고 안쪽에서는 이상하다.
그런데 지금 채점기는 위치를 안 보고 밀도의 최대만 취한다 — 그러면 가장자리가 늘 이기고
헛경보가 거기서 난다.

→ 각 다이의 밀도를 **같은 정규화 반경 대역에 있는 train-none 다이들의 경험분포**에서의
백분위로 바꾼다. "이 자리 치고 얼마나 드문가" 가 되고, 가장자리의 높은 밀도는 저절로 깎인다.

무작위 배치 귀무를 쓰지 않는 이유: **정상의 뭉침이 진짜다.**
독립 Bernoulli 아래 기대 포화율이 0.2% 인데 실측이 70% 였다. 귀무가 너무 느슨하면
정상 웨이퍼가 전부 이상해 보인다. **정상 자체를 귀무로 놓아야 한다.**

## 원리적 대가

진짜 Edge 결함도 가장자리에 있으므로 같이 깎인다. 이건 부작용이 아니라
이 방법이 하는 일의 정의다. 그래서 클래스별 표를 반드시 같이 낸다.

## one-class 격리

참조 분포는 **train & y==0** 만 본다. `assert_one_class` 로 막는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from a22_ood_template import assert_one_class

MAX_REF_PER_BAND = 400_000


def band_index(x, n_bands: int = 32) -> np.ndarray:
    """웨이퍼 **자기 die 반경**으로 정규화한 반경 대역. 다이가 아니면 -1.

    `a22_ood_template._radial_bin_index` 와 같은 정규화다 — 크기 자유도를 남기지 않는다.
    """
    x = np.asarray(x)
    b, h, w = x.shape
    die = x > 0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    n = np.maximum(die.sum((1, 2)), 1).astype(np.float64)
    cy = (die * yy).sum((1, 2)) / n
    cx = (die * xx).sum((1, 2)) / n
    r = np.sqrt((yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2)
    rmax = np.maximum(np.where(die, r, -np.inf).max((1, 2)), 1e-9)[:, None, None]
    u = np.clip(r / rmax, 0.0, 1.0 - 1e-12)
    return np.where(die, (u * n_bands).astype(np.int64), -1)


@dataclass
class RadialBandReference:
    """대역마다 정렬된 정상 다이 밀도 표본. 경험 CDF 의 원자료."""

    values: list = field(default_factory=list)
    n_bands: int = 32


def fit_band_reference(value_map, x, y=None, n_bands: int = 32,
                       max_per_band: int = MAX_REF_PER_BAND,
                       seed: int = 0) -> RadialBandReference:
    """정상 웨이퍼의 다이별 값을 대역별로 모아 정렬한다.

    train-none 은 다이가 2천만 개가 넘어 전부 들고 있으면 무겁다.
    대역마다 최대 `max_per_band` 개로 **고정 seed 무작위 부분표본**을 쓴다 —
    결정론적이므로 재현된다.
    """
    if y is not None:
        assert_one_class(y)
    value_map = np.asarray(value_map, np.float64)
    bands = band_index(x, n_bands)
    rng = np.random.default_rng(seed)
    out = []
    for b in range(n_bands):
        m = bands == b
        v = value_map[m]
        if len(v) == 0:
            v = np.array([0.0])
        elif len(v) > max_per_band:
            v = v[rng.choice(len(v), max_per_band, replace=False)]
        out.append(np.sort(v))
    return RadialBandReference(values=out, n_bands=n_bands)


def fit_global_reference(value_map, x, y=None, max_ref: int = MAX_REF_PER_BAND,
                         seed: int = 0) -> RadialBandReference:
    """**대조군** — 대역을 나누지 않은 하나의 경험분포.

    이게 없으면 "보정" 의 효과와 "반경 대역" 의 효과를 못 가른다.
    대역 1개짜리 참조로 만들어 같은 코드 경로를 타게 한다.
    """
    if y is not None:
        assert_one_class(y)
    value_map = np.asarray(value_map, np.float64)
    v = value_map[np.asarray(x) > 0]
    rng = np.random.default_rng(seed)
    if len(v) > max_ref:
        v = v[rng.choice(len(v), max_ref, replace=False)]
    return RadialBandReference(values=[np.sort(v)], n_bands=1)


def calibrate_map(value_map, bands, ref: RadialBandReference) -> np.ndarray:
    """각 다이의 값을 자기 대역 경험분포에서의 백분위로 바꾼다.

    분모를 `len+1` 로 둬 **정확히 1.0 이 나오지 않게** 한다.
    1.0 이면 뒤에서 -log(1-u) 같은 변환이 발산한다.
    """
    value_map = np.asarray(value_map, np.float64)
    bands = np.asarray(bands)
    out = np.zeros(value_map.shape, np.float64)
    n_bands = ref.n_bands
    for b in range(n_bands):
        m = bands == b if n_bands > 1 else bands >= 0
        if not m.any():
            continue
        r = ref.values[b if n_bands > 1 else 0]
        out[m] = np.searchsorted(r, value_map[m], side="right") / (len(r) + 1.0)
    return np.where(bands >= 0, out, 0.0)


def calibrated_max(value_map, x, ref: RadialBandReference,
                   n_bands: int | None = None) -> np.ndarray:
    """보정된 맵의 웨이퍼별 최대. 이것이 arm 의 점수다."""
    nb = ref.n_bands if n_bands is None else n_bands
    bands = band_index(x, nb) if ref.n_bands > 1 else np.where(np.asarray(x) > 0, 0, -1)
    u = calibrate_map(value_map, bands, ref)
    return u.reshape(len(u), -1).max(1)
