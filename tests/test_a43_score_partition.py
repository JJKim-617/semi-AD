"""O2 채점 경로 — 유효 창이 없는 웨이퍼의 되돌림과 특징 차원.

## 왜 이 파일이 생겼는가

18차 사이클에서 **깊이 1층(24채널)의 특징으로 채점하려다 터졌다.**
되돌림 경로가 특징 차원을 **32 로 하드코딩**하고 있었다
(`f[i].reshape(-1, M.FEAT_DIM)`). 32채널만 쓰던 동안에는 안 드러났다.

O2 본 실행 결과에는 **영향이 없다**(전부 32채널이었다). 그러나
**"차원을 바꾸면 조용히 틀린 모양으로 재해석되는" 코드**였고, 여기서 박는다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

import a42_ood_ssl_knn as M  # noqa: E402
from a43_ood_ssl_eval import gather_bank, score_partition  # noqa: E402
from diag_o2_feature_collapse import patch_feats  # noqa: E402


class TinyEncoder(torch.nn.Module):
    """채널 수를 마음대로 정하는 최소 인코더. 출력은 위치별 L2 정규화."""

    def __init__(self, ch):
        super().__init__()
        self.conv = torch.nn.Conv2d(M.NUM_CATEGORIES, ch, 3, padding=1)

    def forward(self, x):
        f = self.conv(x)
        return f / f.pow(2).sum(1, keepdim=True).clamp_min(1e-12).sqrt()


def make_wafers():
    """유효 7x7 창이 있는 웨이퍼 하나와 **하나도 없는** 웨이퍼 하나."""
    big = np.zeros((16, 16), np.uint8)
    big[2:14, 2:14] = 1
    big[7, 7] = 2
    tiny = np.zeros((16, 16), np.uint8)
    tiny[7:10, 7:10] = 1          # 3x3 뿐이라 7x7 온전 창이 없다
    return np.stack([big, tiny])


def test_the_fixture_really_has_a_wafer_without_any_valid_window():
    X = make_wafers()
    v = M.valid_window_mask(X, M.RF)
    assert v[0].any() and not v[1].any()


@pytest.mark.parametrize("ch", [8, 24, 32, 47])
def test_scoring_works_for_any_feature_dimension(ch):
    """**되돌림 경로가 차원을 하드코딩하면 여기서 터진다.**"""
    torch.manual_seed(0)
    enc = TinyEncoder(ch).eval()
    X = make_wafers()
    idx = np.array([0, 1])
    bank = np.random.default_rng(0).normal(size=(16, ch)).astype(np.float32)
    bank /= np.linalg.norm(bank, axis=1, keepdims=True)
    s, fb = score_partition(enc, X, idx, bank, "cpu")
    assert s.shape == (2,)
    assert np.all(np.isfinite(s))
    assert fb == 1, "유효 창이 없는 웨이퍼 하나가 되돌림으로 세어져야 한다"


@pytest.mark.parametrize("ch", [8, 24, 32])
def test_patch_feats_works_for_any_feature_dimension(ch):
    torch.manual_seed(0)
    enc = TinyEncoder(ch).eval()
    X = make_wafers()
    f = patch_feats(enc, X, np.array([0, 1]), 4, 0)
    assert f.shape[1] == ch
    assert np.all(np.isfinite(f))


def test_a_wafer_without_a_valid_window_still_gets_a_score():
    """조용히 빠지면 그 웨이퍼가 집계에서 사라진다(§10.2)."""
    torch.manual_seed(0)
    enc = TinyEncoder(24).eval()
    X = make_wafers()
    bank = np.eye(24, dtype=np.float32)[:16]
    s, fb = score_partition(enc, X, np.array([1]), bank, "cpu")
    assert len(s) == 1 and np.isfinite(s[0]) and fb == 1


@pytest.mark.parametrize("ch", [8, 24, 32, 47])
def test_bank_building_works_for_any_feature_dimension(ch):
    """**세 진입점을 다 덮는다.** 처음 고칠 때 `gather_bank` 를 빠뜨려 두 번 터졌다.

    `score_partition` 과 `patch_feats` 만 고치고 돌렸더니 `gather_bank` 가
    같은 하드코딩으로 터졌다. 같은 성질은 **같은 성질을 가진 함수 전부**에 박아야 한다.
    """
    torch.manual_seed(0)
    enc = TinyEncoder(ch).eval()
    X = make_wafers()
    bank, cnt, tot = gather_bank(enc, X, np.array([0, 1]), 8, 0, "cpu")
    assert bank.shape[1] == ch
    assert np.all(np.isfinite(bank))
    assert tot == int(M.valid_window_mask(X, M.RF).sum())
    assert cnt.tolist() == M.valid_window_mask(X, M.RF).reshape(2, -1).sum(1).tolist()


def test_bank_never_exceeds_the_available_valid_locations():
    torch.manual_seed(0)
    enc = TinyEncoder(16).eval()
    X = make_wafers()
    bank, cnt, tot = gather_bank(enc, X, np.array([0, 1]), 10 ** 6, 0, "cpu")
    assert len(bank) == tot
