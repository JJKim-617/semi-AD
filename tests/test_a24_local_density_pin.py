"""국소 불량 밀도가 **학습 없는 자명한 스칼라**라는 것을 코드로 박는다.

## 왜 이 파일이 필요한가

O0 은 자명한 기준선으로 **전역** 스칼라만 쟀고 바닥을 AUPR 0.3856 이라고 적었다.
그런데 창 하나 크기의 **국소** 스칼라가 그보다 훨씬 높다.
바닥이 잘못 잡혀 있었고, 그 위에서 판정한 O1 사다리의 채택 기준도 잘못돼 있었다.

## 정정 (2026-08-23) — 여기 있던 k=3 수치가 틀렸었다

처음에 k=3 AUPR 을 **0.4968** 로 박았다. **측정 도구가 만든 값이었다.**
`scipy.ndimage.uniform_filter` 로 창합을 구하면 분리 가능 running sum 의 반올림 오차가
쌓여, 분모가 9 이하인 유리수 **23개**밖에 못 갖는 값이 **96~108개**로 쪼개진다
(음수 밀도까지 나왔다). 그 가짜 구분이 동점을 갈랐고, 동점은 원래 인덱스 순서로
정렬되는데 test 배열 위치가 결함 여부와 약하게 상관돼(0.0116) **라벨 정보가 샜다.**

정수 창합으로 고치니 **0.3844** 다. k>=5 에서는 영향이 0.002 이하다.

**그래서 두 가지를 바꿨다.**
1. 창합을 정확히 구한다(`window_sum`). 같은 (불량, 다이) 개수는 같은 비트를 준다.
2. 주 지표를 **`aupr_blocked`** 로 바꾼다 — 동점을 블록으로 묶어 입력 순서와 무관하게 만든다.
   같은 점수를 받은 표본은 어떤 문턱으로도 안 갈리므로 지표도 갈라선 안 된다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a21_ood_metrics import (  # noqa: E402
    aupr,
    aupr_blocked,
    auroc,
    evaluate_ood,
)
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
    """균일 template 잔차 뭉개기 = 국소 밀도. template 의 기여가 0 이다.

    `smooth_k_max` 의 성과를 template 의 것으로 읽었던 첫 해석이 틀렸다는 증거다.
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
        """값이 실제로 다른 쌍에서는 두 경로의 대소가 항상 같다.

        argsort 나 AUROC 를 bit 단위로 비교하면 안 된다 — 아핀 변환이 **동점 구조**를
        바꿔 순서가 갈릴 수 있다.
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
        assert m.min() >= 0.0 and m.max() <= 1.0
        assert np.all(m[x == 0] == 0.0)

    def test_an_all_fail_neighbourhood_reaches_one(self):
        x = np.zeros((1, 32, 32), np.uint8)
        x[0, 8:24, 8:24] = 1
        x[0, 12:20, 12:20] = 2
        assert local_fail_density_max(x, k=3)[0] == pytest.approx(1.0)


@pytest.mark.skipif(not (CACHE.exists() and SPLITS.exists()), reason="캐시 없음")
class TestNewTrivialFloorIsPinned:
    """**바닥은 0.3856 이 아니다.** 이 아래로 돌아가면 테스트가 깨진다."""

    @pytest.fixture(scope="class")
    def scored(self):
        sp = np.load(SPLITS)
        d = np.load(CACHE, allow_pickle=True)
        te = sp["test"]
        X = np.ascontiguousarray(d["X"][te])
        is_def = (d["y"].astype(np.int64)[te] != 0).astype(np.int64)
        return {k: local_fail_density_max(X, k=k, chunk=4000) for k in (3, 5, 7)}, is_def

    def test_local_density_7x7_is_the_real_floor(self, scored):
        s, is_def = scored
        m = evaluate_ood(s[7], is_def)
        assert m["auroc"] == pytest.approx(0.9206, abs=2e-3)
        assert m["aupr"] == pytest.approx(0.7129, abs=2e-3)
        assert m["fpr_at_95tpr"] == pytest.approx(0.4854, abs=2e-3)

    def test_the_blocked_convention_is_what_we_report(self, scored):
        """주 지표. 입력 순서와 무관하다."""
        s, is_def = scored
        assert aupr_blocked(s[7], is_def) == pytest.approx(0.7098, abs=2e-3)
        assert aupr_blocked(s[5], is_def) == pytest.approx(0.6746, abs=2e-3)

    def test_the_local_floor_is_far_above_the_global_one(self, scored):
        """O0 이 잡은 바닥(0.3856)은 전역 스칼라만 재서 나온 값이었다."""
        s, is_def = scored
        assert aupr_blocked(s[7], is_def) > 0.3856 + 0.25

    def test_the_corrected_k3_value(self, scored):
        """**정정된 값.** 예전에 0.4968 로 박았던 것은 부동소수점 잡음이 만든 값이다."""
        s, is_def = scored
        assert aupr(s[3], is_def) == pytest.approx(0.3844, abs=2e-3)
        assert aupr_blocked(s[3], is_def) == pytest.approx(0.4115, abs=3e-3)

    def test_k3_density_takes_only_a_handful_of_distinct_values(self, scored):
        """분모가 9 이하인 유리수뿐이다. 100개가 넘게 나오면 도구가 잡음을 만든 것이다."""
        s, _ = scored
        assert len(np.unique(s[3])) <= 30

    def test_window_size_matters(self, scored):
        s, is_def = scored
        assert aupr_blocked(s[7], is_def) > aupr_blocked(s[5], is_def)
        assert aupr_blocked(s[5], is_def) > aupr_blocked(s[3], is_def)
