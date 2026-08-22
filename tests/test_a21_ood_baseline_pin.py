"""E0 기준선을 고정한다. 이 숫자 아래로 되돌아가면 테스트가 깨진다.

9-class 에서 "전부 none 으로 찍는 분류기가 accuracy 0.9334" 를 테스트로 박아
지표가 조용히 되돌려지는 것을 막았다. one-class 도 같은 장치가 필요하다.

여기서 고정하는 것은 **학습 없는 스칼라의 성능**이다. 어떤 방법이든 이걸 넘어야
채택할 이유가 있다. 바닥은 무작위(AUROC 0.5)가 아니라 불량 다이 개수(0.8167)다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a21_ood_metrics import evaluate_ood, fail_count, fail_ratio

CACHE = Path("data/wm811k/cache/wm811k_64pad.npz")
SPLITS = Path("data/wm811k/cache/splits_v1.npz")
pytestmark = pytest.mark.skipif(
    not (CACHE.exists() and SPLITS.exists()), reason="캐시 없음 (데이터 심링크 필요)")


@pytest.fixture(scope="module")
def test_split():
    sp = np.load(SPLITS)
    d = np.load(CACHE, allow_pickle=True)
    te = sp["test"]
    X = d["X"][te]                      # npz 지연 로딩 — 한 번만 읽는다
    y = d["y"].astype(np.int64)[te]
    return X, y, (y != 0).astype(np.int64)


class TestTrivialBaselineIsPinned:
    def test_prevalence_matches_the_official_split(self, test_split):
        _, _, is_def = test_split
        assert is_def.mean() == pytest.approx(0.0666, abs=1e-3)
        assert len(is_def) == 118_595
        assert int(is_def.sum()) == 7_894

    def test_fail_count_baseline(self, test_split):
        X, _, is_def = test_split
        m = evaluate_ood(fail_count(X), is_def)
        assert m["auroc"] == pytest.approx(0.8167, abs=1e-3)
        assert m["aupr"] == pytest.approx(0.3633, abs=1e-3)
        assert m["fpr_at_95tpr"] == pytest.approx(0.7367, abs=1e-3)

    def test_fail_ratio_baseline(self, test_split):
        X, _, is_def = test_split
        m = evaluate_ood(fail_ratio(X), is_def)
        assert m["auroc"] == pytest.approx(0.7899, abs=1e-3)
        assert m["aupr"] == pytest.approx(0.3856, abs=1e-3)

    def test_aupr_floor_is_prevalence_not_half(self, test_split):
        """AUPR 0.38 이 좋은 값인지 알려면 바닥이 0.0666 이라는 것을 함께 봐야 한다."""
        X, _, is_def = test_split
        m = evaluate_ood(fail_ratio(X), is_def)
        assert m["prevalence"] == pytest.approx(0.0666, abs=1e-3)
        assert m["aupr"] > m["prevalence"] * 5

    def test_operating_point_is_unusable_despite_high_auroc(self, test_split):
        """AUROC 0.82 인데 결함 95% 를 잡으려면 정상의 74% 를 버린다.

        이 대비가 AUROC 만 보면 안 되는 이유다. 개선됐다고 주장하려면
        이 숫자가 내려가야 한다.
        """
        X, _, is_def = test_split
        m = evaluate_ood(fail_count(X), is_def)
        assert m["auroc"] > 0.8
        assert m["fpr_at_95tpr"] > 0.7
