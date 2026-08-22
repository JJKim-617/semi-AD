"""다이 단위 잔차 맵과 뭉개기 연산자 — 단위 테스트.

파이프라인 1단계가 하류(국소화 → MLLM)에 넘기는 것은 스칼라가 아니라 **맵**이다.
그래서 여기서 박는 성질은 "맵이 스칼라와 앞뒤가 맞는가" 와
"뭉개기 연산자가 각자 의도한 것을 실제로 재는가" 두 가지다.

**픽셀 단위 정답을 합성해 P-AUROC 를 만들지 않는다**(기획서 §9.3).
데이터에 다이 단위 마스크가 없으므로 여기 테스트도 합성 정답을 쓰지 않고
연산자의 대수적 성질만 검사한다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a22_ood_template import PositionTemplate, score_llr, score_nll  # noqa: E402
from a24_ood_residual import (  # noqa: E402
    POOLERS,
    pool_max,
    pool_max_component,
    pool_smoothed_max,
    pool_sum,
    pool_topk_mean,
    residual_map,
)


def _wafer(n=1, radius=25.0, size=64):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    m = ((yy - c) ** 2 + (xx - c) ** 2) <= radius ** 2
    x = np.zeros((size, size), np.uint8)
    x[m] = 1
    return np.repeat(x[None], n, axis=0)


def _template(p_value=0.08, size=64):
    return PositionTemplate(p=np.full((size, size), p_value),
                            die_count=np.full((size, size), 100.0), global_rate=p_value)


def _varied_template(size=64):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    r = np.sqrt((yy - 31.5) ** 2 + (xx - 31.5) ** 2)
    p = np.full((size, size), 0.02)
    p[r > 20.0] = 0.35
    return PositionTemplate(p=p, die_count=np.full((size, size), 100.0), global_rate=0.08)


class TestMapAgreesWithScalar:
    """맵을 더하면 정확히 스칼라가 나와야 한다.

    이게 깨지면 하류가 보는 맵과 우리가 보고한 AUPR 이 서로 다른 것을 재고 있는 것이다.
    """

    def test_nll_map_sums_to_score_nll(self):
        x = _wafer(4)
        rng = np.random.default_rng(0)
        for i in range(4):
            idx = np.flatnonzero(x[i].ravel() > 0)
            x[i].ravel()[rng.choice(idx, 50 + 20 * i, replace=False)] = 2
        t = _varied_template()
        m = residual_map(x, t, mode="nll")
        assert np.allclose(m.sum((1, 2)), score_nll(x, t), atol=1e-8)

    def test_llr_map_sums_to_score_llr(self):
        x = _wafer(4)
        rng = np.random.default_rng(1)
        for i in range(4):
            idx = np.flatnonzero(x[i].ravel() > 0)
            x[i].ravel()[rng.choice(idx, 30 + 25 * i, replace=False)] = 2
        t = _varied_template()
        m = residual_map(x, t, mode="llr")
        assert np.allclose(m.sum((1, 2)), score_llr(x, t), atol=1e-8)


class TestMapIsZeroWhereThereIsNoDie:
    def test_no_die_cells_carry_no_residual(self):
        """다이가 없는 칸에 값이 새면 시각화와 하류 프롬프트가 웨이퍼 밖을 가리킨다."""
        x = _wafer(2, radius=12.0)
        x[:, 31, 31] = 2
        t = _varied_template()
        for mode in ("nll", "llr"):
            m = residual_map(x, t, mode=mode)
            assert np.all(m[x == 0] == 0.0)


class TestPoolers:
    def test_sum_pooling_reproduces_the_scalar_score(self):
        x = _wafer(3)
        x[:, 31, 31] = 2
        t = _varied_template()
        m = residual_map(x, t, mode="llr")
        assert np.allclose(pool_sum(m, x > 0), m.sum((1, 2)))

    def test_max_pooling_finds_the_single_worst_die(self):
        x = _wafer(2)
        x[0, 31, 31] = 2                       # template 이 낮은 안쪽
        x[1, 31, 31] = 2
        x[1, 5, 31] = 2                        # 다이 없는 칸이라 무시돼야 한다
        t = _varied_template()
        m = residual_map(x, t, mode="llr")
        assert pool_max(m, x > 0) == pytest.approx(m.max((1, 2)))

    def test_topk_mean_is_between_mean_and_max(self):
        rng = np.random.default_rng(2)
        x = _wafer(1)
        idx = np.flatnonzero(x[0].ravel() > 0)
        x[0].ravel()[rng.choice(idx, 120, replace=False)] = 2
        t = _varied_template()
        m = residual_map(x, t, mode="llr")
        die = x > 0
        v = pool_topk_mean(m, die, frac=0.05)
        assert (m[die].mean() < v[0] <= m.max()) or np.isclose(v[0], m.max())

    def test_smoothed_max_prefers_a_cluster_over_the_same_count_scattered(self):
        """Scratch/Loc 이 값어치를 내야 하는 지점 — 개수가 같고 뭉침만 다르다."""
        x = _wafer(2, radius=25.0)
        rng = np.random.default_rng(3)
        line = [(31, 20 + k) for k in range(12)]
        for (i, j) in line:
            x[0, i, j] = 2
        idx = np.flatnonzero((x[1] > 0).ravel())
        x[1].ravel()[rng.choice(idx, 12, replace=False)] = 2
        t = _template(0.08)
        m = residual_map(x, t, mode="nll")
        die = x > 0
        s = pool_smoothed_max(m, die, k=3)
        assert s[0] > s[1]

    def test_max_component_counts_the_largest_connected_flagged_region(self):
        x = _wafer(2, radius=25.0)
        for k in range(10):
            x[0, 31, 20 + k] = 2               # 붙어 있는 10개
        for k in range(10):
            x[1, 20 + 2 * k % 20, 15 + 3 * k] = 2   # 흩어진 10개
        t = _template(0.08)
        m = residual_map(x, t, mode="nll")
        die = x > 0
        thr = float(np.quantile(m[die], 0.5))
        c = pool_max_component(m, die, threshold=thr)
        assert c[0] >= 10
        assert c[0] > c[1]

    def test_max_component_is_zero_when_nothing_passes_the_threshold(self):
        x = _wafer(1, radius=20.0)
        t = _template(0.08)
        m = residual_map(x, t, mode="nll")
        assert pool_max_component(m, x > 0, threshold=1e9)[0] == 0

    def test_registry_names_every_pooler_and_returns_one_value_per_wafer(self):
        x = _wafer(5, radius=20.0)
        x[:, 31, 31] = 2
        t = _varied_template()
        m = residual_map(x, t, mode="llr")
        die = x > 0
        assert len(POOLERS) >= 5
        for name, fn in POOLERS.items():
            v = np.asarray(fn(m, die))
            assert v.shape == (5,), name
            assert np.isfinite(v).all(), name


class TestSmoothMapIsTheLocalisationProduct:
    """맵을 그대로 내보내려면 스칼라와 앞뒤가 맞아야 한다."""

    def test_max_of_the_smoothed_map_equals_the_pooled_scalar(self):
        from a24_ood_residual import pool_smoothed_max, smooth_map
        x = _wafer(3, radius=22.0)
        for i in range(3):
            x[i, 30, 20 + 3 * i:26 + 3 * i] = 2
        t = _varied_template()
        m = residual_map(x, t, mode="nll")
        die = x > 0
        sm = smooth_map(m, die, k=5)
        assert np.allclose(sm.max((1, 2)), pool_smoothed_max(m, die, k=5))

    def test_smoothed_map_is_zero_outside_the_die_region(self):
        from a24_ood_residual import smooth_map
        x = _wafer(2, radius=10.0)
        x[:, 31, 31] = 2
        t = _varied_template()
        sm = smooth_map(residual_map(x, t, mode="nll"), x > 0, k=5)
        assert np.all(sm[x == 0] == 0.0)

    def test_smoothing_spreads_a_single_defect_over_its_neighbourhood(self):
        """국소화가 되려면 한 점이 이웃으로 번져야 사람이 볼 수 있다."""
        from a24_ood_residual import smooth_map
        x = _wafer(1, radius=22.0)
        x[0, 31, 31] = 2
        t = _template(0.08)
        m = residual_map(x, t, mode="nll")
        sm = smooth_map(m, x > 0, k=5)
        assert (sm[0] > sm[0][x[0] > 0].min()).sum() > (m[0] > m[0][x[0] > 0].min()).sum()
