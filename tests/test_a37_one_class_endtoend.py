"""one-class 보장을 **파이프라인 끝에서** 확인한다.

`assert_one_class` 는 fit 진입점에서 라벨을 검사할 뿐이다. 그건
"결함 라벨을 넘기지 않았다" 를 보장하지 그 **픽셀**이 안 새어 들어갔다는 보장은 아니다.

여기서 박는 것은 더 강한 성질이다:
**train 의 결함 웨이퍼 픽셀을 통째로 망가뜨려도 test 점수가 비트 단위로 같아야 한다.**
그게 참이면 채점기가 결함 웨이퍼를 어떤 경로로도 안 봤다는 뜻이다.
어딘가에서 `y==0` 필터를 빠뜨렸다면 이 시험이 깨진다.

기획서 §1: "학습 단계에서 결함을 본 순간 그 실험은 무효다 — 코드로 막고 테스트로 고정해라."
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a23_ood_template_eval import ecdf_percentile, fisher  # noqa: E402
from a24_ood_residual import local_fail_count_map, local_fail_count_max  # noqa: E402
from a27_line_filter import line_density_max  # noqa: E402
from a29_radial_calibration import (  # noqa: E402
    band_index,
    calibrate_map,
    fit_band_reference,
)

CACHE = Path("data/wm811k/cache/wm811k_64pad.npz")
SPLITS = Path("data/wm811k/cache/splits_v1.npz")
pytestmark = pytest.mark.skipif(
    not (CACHE.exists() and SPLITS.exists()), reason="캐시 없음")

N_BANDS, L_LINE = 32, 11


def _fusion_scores(x_train_all, y_train_all, x_test):
    """현재 최선(3-요소 융합)을 처음부터 끝까지. **`y==0` 필터는 여기 한 번만 있다.**"""
    trn = x_train_all[y_train_all == 0]
    v = local_fail_count_map(trn, k=5, chunk=1000).astype(np.float32)
    ref = fit_band_reference(v, trn, y=y_train_all[y_train_all == 0],
                             n_bands=N_BANDS, seed=0)

    def a(x):
        u = calibrate_map(local_fail_count_map(x, k=5, chunk=1000),
                          band_index(x, N_BANDS), ref)
        return u.reshape(len(x), -1).max(1)

    comps = [
        (a(trn), a(x_test)),
        (line_density_max(trn, length=L_LINE, n_orient=8, chunk=1000, min_dies=L_LINE),
         line_density_max(x_test, length=L_LINE, n_orient=8, chunk=1000, min_dies=L_LINE)),
        (local_fail_count_max(trn, k=3, chunk=1000),
         local_fail_count_max(x_test, k=3, chunk=1000)),
    ]
    return fisher([ecdf_percentile(t, e) for t, e in comps])


@pytest.fixture(scope="module")
def data():
    sp = np.load(SPLITS)
    d = np.load(CACHE, allow_pickle=True)
    y = d["y"].astype(np.int64)
    tr, te = sp["train"][:6000], sp["test"][:1500]
    X = d["X"]
    return (np.ascontiguousarray(X[tr]), y[tr], np.ascontiguousarray(X[te]))


class TestPipelineIsBlindToTrainDefects:
    def test_corrupting_train_defect_pixels_does_not_move_the_test_score(self, data):
        """**이 파일의 요점.** 비트 단위로 같아야 한다 — `allclose` 가 아니다."""
        x_tr, y_tr, x_te = data
        assert (y_tr != 0).sum() > 100, "train 결함이 있어야 시험이 의미가 있다"
        base = _fusion_scores(x_tr, y_tr, x_te)

        rng = np.random.default_rng(0)
        x_corrupt = x_tr.copy()
        defect = y_tr != 0
        x_corrupt[defect] = rng.integers(0, 3, size=x_corrupt[defect].shape,
                                         dtype=np.uint8)
        after = _fusion_scores(x_corrupt, y_tr, x_te)
        assert np.array_equal(base, after)

    def test_deleting_train_defect_wafers_entirely_does_not_move_it_either(self, data):
        """행을 지워도 같아야 한다 — 인덱스에 의존하는 경로가 없다는 뜻이다."""
        x_tr, y_tr, x_te = data
        base = _fusion_scores(x_tr, y_tr, x_te)
        keep = y_tr == 0
        after = _fusion_scores(x_tr[keep], y_tr[keep], x_te)
        assert np.array_equal(base, after)

    def test_the_guard_still_fires_if_defects_are_handed_to_the_fit(self, data):
        """가드가 살아 있는지도 같이 확인한다 (약한 보장이지만 있어야 한다)."""
        from a22_ood_template import assert_one_class
        x_tr, y_tr, _ = data
        with pytest.raises(ValueError, match="one-class"):
            assert_one_class(y_tr)

    def test_corrupting_train_NORMAL_pixels_DOES_move_the_score(self, data):
        """대조군. 정상까지 안 쓰면 그건 보장이 아니라 그냥 상수 채점기다."""
        x_tr, y_tr, x_te = data
        base = _fusion_scores(x_tr, y_tr, x_te)
        rng = np.random.default_rng(1)
        x_corrupt = x_tr.copy()
        normal = y_tr == 0
        x_corrupt[normal] = rng.integers(0, 3, size=x_corrupt[normal].shape,
                                         dtype=np.uint8)
        after = _fusion_scores(x_corrupt, y_tr, x_te)
        assert not np.array_equal(base, after)
