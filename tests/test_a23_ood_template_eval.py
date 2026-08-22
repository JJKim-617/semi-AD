"""O1 실행기의 통계 도구 — 지름길이 원본과 같은 값을 주는지 박는다.

부트스트랩 때문에 AUROC 를 벡터화한 사본과 재정렬 없는 AUPR 재표본을 썼다.
둘 다 **원본과 같은 값을 준다는 전제** 위에서만 신뢰구간이 의미가 있다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a21_ood_metrics import aupr, auroc  # noqa: E402
from a23_ood_template_eval import (  # noqa: E402
    _fast_auroc,
    bootstrap_aupr_ci,
    ecdf_percentile,
    fisher,
)


class TestFastAurocMatchesReference:
    def test_fast_auroc_matches_reference_on_continuous_scores(self):
        rng = np.random.default_rng(0)
        l = (rng.random(4000) < 0.07).astype(np.int64)
        s = rng.normal(size=4000) + l * 0.8
        assert _fast_auroc(s, l) == pytest.approx(auroc(s, l), abs=1e-12)

    def test_fast_auroc_matches_reference_with_heavy_ties(self):
        """불량 다이 개수처럼 동점이 많은 점수에서 갈리면 CI 가 통째로 틀린다."""
        rng = np.random.default_rng(1)
        l = (rng.random(5000) < 0.1).astype(np.int64)
        s = np.floor(rng.gamma(1.2, 3.0, size=5000) + l * 4.0)
        assert (np.unique(s).size < 60)
        assert _fast_auroc(s, l) == pytest.approx(auroc(s, l), abs=1e-12)

    def test_uninformative_score_is_half(self):
        l = np.array([0, 1] * 50)
        assert _fast_auroc(np.ones(100), l) == pytest.approx(0.5)


class TestBootstrapAupr:
    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(2)
        l = (rng.random(6000) < 0.08).astype(np.int64)
        s = rng.normal(size=6000) + l * 1.2
        lo, hi = bootstrap_aupr_ci(s, l, n_boot=200, seed=3)
        assert lo < aupr(s, l) < hi

    def test_interval_is_reproducible_for_a_fixed_seed(self):
        rng = np.random.default_rng(4)
        l = (rng.random(3000) < 0.1).astype(np.int64)
        s = rng.normal(size=3000) + l
        assert bootstrap_aupr_ci(s, l, 100, 7) == bootstrap_aupr_ci(s, l, 100, 7)


class TestOneClassCalibration:
    def test_percentile_uses_only_the_reference_sample(self):
        ref = np.arange(100, dtype=np.float64)
        u = ecdf_percentile(ref, np.array([-5.0, 49.5, 200.0]))
        assert u[0] == pytest.approx(0.0)
        assert 0.4 < u[1] < 0.6
        assert u[2] < 1.0                       # 1.0 이면 log(0) 으로 터진다

    def test_fisher_grows_with_every_component(self):
        a = np.array([0.1, 0.9])
        b = np.array([0.5, 0.5])
        assert fisher([a, b])[1] > fisher([a, b])[0]
        assert (fisher([a, b]) > fisher([a])).all()
