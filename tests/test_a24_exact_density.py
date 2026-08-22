"""국소 밀도는 **정수 창합의 비율**이다 — 부동소수점이 동점을 쪼개면 안 된다.

## 왜 이게 문제인가

창 3x3 밀도가 취할 수 있는 값은 분모가 9 이하인 유리수뿐이라 23개밖에 없다.
그런데 `scipy.ndimage.uniform_filter` 로 계산하면 **96개**가 나왔다.
창합을 분리 가능 running sum 으로 구하면서 마지막 비트에 오차가 쌓여
**같아야 할 값들이 1e-16 수준에서 갈린 것**이다.

그 가짜 구분이 AUPR 을 0.365 에서 0.446 으로 부풀렸다.
동점이 인덱스 순서로 정렬되는데 test 배열 위치가 결함 여부와 약하게 상관되기 때문이다.
**측정 도구가 만든 잡음이 라벨 정보로 둔갑했다.**

그래서 창합을 **정확히** 구한다. 0/1 값의 정수합은 float64 에서 오차가 없고
(2^53 까지), IEEE 나눗셈은 올바르게 반올림되므로 같은 유리수는 같은 비트가 된다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a24_ood_residual import local_fail_density_map  # noqa: E402


def _brute_force(x, k):
    """창 안 불량/다이 개수를 정수로 직접 세서 나눈 기준값."""
    b, h, w = x.shape
    half = k // 2
    out = np.zeros((b, h, w), np.float64)
    for n in range(b):
        for i in range(h):
            for j in range(w):
                if x[n, i, j] == 0:
                    continue
                i0, i1 = max(0, i - half), min(h, i + half + 1)
                j0, j1 = max(0, j - half), min(w, j + half + 1)
                win = x[n, i0:i1, j0:j1]
                die = int((win > 0).sum())
                fail = int((win == 2).sum())
                out[n, i, j] = fail / die if die else 0.0
    return out


def _wafer(seed=0, size=24, radius=10.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    m = ((yy - c) ** 2 + (xx - c) ** 2) <= radius ** 2
    x = np.zeros((1, size, size), np.uint8)
    x[0][m] = 1
    idx = np.flatnonzero(x[0].ravel() > 0)
    x[0].ravel()[rng.choice(idx, len(idx) // 4, replace=False)] = 2
    return x


class TestDensityIsExact:
    @pytest.mark.parametrize("k", [3, 5, 7])
    def test_it_matches_integer_counting_bit_for_bit(self, k):
        """`allclose` 가 아니라 `==` 다. 마지막 비트가 동점을 가른다."""
        x = _wafer(seed=1)
        got = local_fail_density_map(x, k=k)
        want = _brute_force(x, k)
        assert np.array_equal(got, want)

    def test_three_by_three_takes_only_rationals_with_denominator_at_most_nine(self):
        """가능한 값이 23개인데 96개가 나오면 도구가 잡음을 만든 것이다."""
        xs = np.concatenate([_wafer(seed=s, size=28, radius=12.0) for s in range(12)])
        vals = local_fail_density_map(xs, k=3)
        uniq = np.unique(vals[xs > 0])
        rationals = {0.0}
        for den in range(1, 10):
            for num in range(0, den + 1):
                rationals.add(num / den)
        assert len(uniq) <= len(rationals)
        for v in uniq:
            assert any(v == r for r in rationals), v

    def test_equal_counts_give_bitwise_equal_densities(self):
        """창 안 (불량, 다이) 개수가 같으면 값이 정확히 같아야 한다."""
        x = np.zeros((2, 20, 20), np.uint8)
        x[:, 5:15, 5:15] = 1
        x[0, 9, 9] = 2
        x[0, 9, 10] = 2
        x[1, 12, 12] = 2
        x[1, 12, 13] = 2
        m = local_fail_density_map(x, k=5)
        assert m[0, 9, 9] == m[1, 12, 12]

    def test_it_still_respects_the_die_mask(self):
        x = _wafer(seed=2)
        m = local_fail_density_map(x, k=5)
        assert np.all(m[x == 0] == 0.0)
        assert m.max() <= 1.0
