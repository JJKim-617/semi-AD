"""국소 불량 밀도가 **학습 없는 자명한 스칼라**라는 것을 코드로 박는다.

## 왜 이 파일이 필요한가

O0 은 자명한 기준선으로 **전역** 스칼라만 쟀고 바닥을 AUPR 0.3856 이라고 적었다.
그런데 창 하나 크기의 **국소** 스칼라가 AUPR 0.7147 을 낸다.
바닥이 잘못 잡혀 있었고, 그 위에서 판정한 O1 사다리의 채택 기준도 잘못돼 있었다.

여기서 세 가지를 고정한다.

1. **국소 밀도 채점기는 아무것도 학습하지 않는다** — 정의상 template 이 없다.
2. **template 을 균일 상수로 둔 잔차 뭉개기와 순위가 정확히 같다.**
   이게 참이라서 "smooth_max 가 O1 을 살렸다" 는 첫 해석이 틀렸다고 말할 수 있다.
   그 점수는 template 이 아니라 국소 밀도가 낸 것이었다.
3. **새 바닥 수치를 회귀 테스트로 박는다.** E0 를 박아 둔 것과 같은 장치다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a21_ood_metrics import evaluate_ood  # noqa: E402
from a22_ood_template import uniform_template  # noqa: E402
from a24_ood_residual import (  # noqa: E402
    local_fail_density_map,
    local_fail_density_max,
    pool_smoothed_max,
    residual_map,
    smooth_map,
)

CACHE = Path("data/wm811k/cache/wm811k_64pad.npz")
SPLITS = Path("data/wm811k/cache/splits_v1.npz")


def _synthetic(n=40, size=64, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = ((yy - (size - 1) / 2) ** 2 + (xx - (size - 1) / 2) ** 2) <= (size * 0.37) ** 2
    x = np.zeros((n, size, size), np.uint8)
    for i in range(n):
        x[i][disc] = 1
        idx = np.flatnonzero(x[i].ravel() > 0)
        x[i].ravel()[rng.choice(idx, 20 + 8 * i, replace=False)] = 2
    return x


class TestLocalDensityLearnsNothing:
    """**첫 해석이 틀렸다는 증거.**

    균일 template 의 셀 잔차는 불량이면 -log(p), 정상 다이면 -log(1-p) 다.
    창 평균을 창 안 다이 개수로 나누면 국소 불량 비율 f 에 대해

        sm = -log(1-p) + f * log((1-p)/p)

    즉 **기울기가 양수인 아핀 함수**가 된다. 그러므로 `smooth_k_max` 의 성능은
    template 이 낸 것이 아니라 국소 밀도가 낸 것이다.

    처음에는 이걸 "argsort 결과가 같다" 로 박으려 했는데 **틀린 검사**였다.
    아핀 변환이 1 ULP 차이를 없애거나 만들어서 **동점 구조**가 달라지고,
    그러면 순열이 달라진다(실측: 항등식 오차 8e-16, 동점 개수 3 vs 4).
    등식 자체를 박고 기울기 부호를 따로 박는 것이 맞다.
    """

    @pytest.mark.parametrize("k", [3, 5, 7])
    @pytest.mark.parametrize("p", [0.02, 0.13, 0.40])
    def test_uniform_template_residual_is_affine_in_the_local_fail_fraction(self, k, p):
        x = _synthetic(20, seed=1)
        die = x > 0
        got = smooth_map(residual_map(x, uniform_template(p, shape=x.shape[1:]),
                                      mode="nll"), die, k=k)
        f = local_fail_density_map(x, k=k)
        expect = np.where(die, -np.log1p(-p) + f * np.log((1 - p) / p), 0.0)
        assert np.abs(got[die] - expect[die]).max() < 1e-12

    @pytest.mark.parametrize("p", [0.001, 0.02, 0.13, 0.40, 0.499])
    def test_the_affine_slope_is_positive_so_the_ordering_is_preserved(self, p):
        assert np.log((1 - p) / p) > 0

    @pytest.mark.parametrize("k", [3, 5, 7])
    def test_the_two_routes_order_every_genuinely_different_pair_the_same_way(self, k):
        """**정확히 주장할 수 있는 것은 여기까지다.**

        값이 실제로 다른 쌍에서는 두 경로의 대소가 항상 같다.
        값이 부동소수점 잡음 수준으로 같은 쌍은 한쪽에서 동점, 다른 쪽에서 1 ULP 차이가
        되어 순서가 갈릴 수 있다 — 그래서 argsort 나 AUROC 를 bit 단위로 비교하면 안 된다.
        실측에서 그 차이가 AUROC 0.5006 대 0.5019 로 나타났고, 공식 test 118,595장에서는
        보고 자릿수(1e-3)에서 구분되지 않는다(`TestNewTrivialFloorIsPinned`).
        """
        x = _synthetic(60, seed=4)
        d = local_fail_density_max(x, k=k)
        v = pool_smoothed_max(
            residual_map(x, uniform_template(0.13, shape=x.shape[1:]), mode="nll"),
            x > 0, k=k)
        dd = d[:, None] - d[None, :]
        vv = v[:, None] - v[None, :]
        real = np.abs(dd) > 1e-9
        assert real.sum() > 100
        assert np.array_equal(np.sign(dd[real]), np.sign(vv[real]))

    def test_density_is_a_fraction_between_zero_and_one(self):
        x = _synthetic(12, seed=3)
        m = local_fail_density_map(x, k=5)
        assert m.min() >= -1e-12 and m.max() <= 1.0 + 1e-12
        assert np.all(m[x == 0] == 0.0)

    def test_an_all_fail_neighbourhood_reaches_one(self):
        x = np.zeros((1, 32, 32), np.uint8)
        x[0, 8:24, 8:24] = 1
        x[0, 12:20, 12:20] = 2
        assert local_fail_density_max(x, k=3)[0] == pytest.approx(1.0)


@pytest.mark.skipif(not (CACHE.exists() and SPLITS.exists()), reason="캐시 없음")
class TestNewTrivialFloorIsPinned:
    """**바닥은 0.3856 이 아니라 0.7147 이다.** 이 아래로 돌아가면 테스트가 깨진다."""

    @pytest.fixture(scope="class")
    def scored(self):
        sp = np.load(SPLITS)
        d = np.load(CACHE, allow_pickle=True)
        te = sp["test"]
        X = np.ascontiguousarray(d["X"][te])
        is_def = (d["y"].astype(np.int64)[te] != 0).astype(np.int64)
        return {k: local_fail_density_max(X, k=k) for k in (3, 7)}, is_def

    def test_local_density_7x7_is_the_real_floor(self, scored):
        s, is_def = scored
        m = evaluate_ood(s[7], is_def)
        assert m["auroc"] == pytest.approx(0.9206, abs=2e-3)
        assert m["aupr"] == pytest.approx(0.7147, abs=2e-3)
        assert m["fpr_at_95tpr"] == pytest.approx(0.4852, abs=2e-3)

    def test_the_local_floor_is_far_above_the_global_one(self, scored):
        """O0 이 잡은 바닥(0.3856)은 전역 스칼라만 재서 나온 값이었다."""
        s, is_def = scored
        assert evaluate_ood(s[7], is_def)["aupr"] > 0.3856 + 0.25

    def test_window_size_matters_and_three_is_too_small(self, scored):
        s, is_def = scored
        assert evaluate_ood(s[3], is_def)["aupr"] == pytest.approx(0.4968, abs=2e-3)
