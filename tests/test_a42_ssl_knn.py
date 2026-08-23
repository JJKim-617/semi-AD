"""O2 — 자기지도 인코더 + kNN 패치 뱅크. 단위 테스트.

설계는 `docs/experiments/candidate/ood_o2_ssl_knn.md` §10 에 **코드보다 먼저** 박았다.
여기서 박는 성질은 그 §10 의 약속 그대로다.

1. **수용영역이 정확히 7x7 이다**(§10.1). 이 값이 이 arm 의 공정성을 정한다 —
   채택된 창 통계와 같은 창 안에서 겨루게 하려고 고른 값이다.
2. **온전 창만 점수를 낸다**(§10.2, 정정 11).
3. **뱅크와 kNN 이 정의대로 계산된다**(§10.4).
4. **시드가 같으면 비트 단위로 같다**(§10.9 가 요구한다 — 결정론이 없으면
   one-class 성질을 비트로 못 잰다).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from a42_ood_ssl_knn import (  # noqa: E402
    RF,
    PatchEncoder,
    build_bank,
    knn_mean_distance,
    masked_valid_ce,
    to_onehot,
    valid_window_mask,
    wafer_max_score,
)


def wafer(rows):
    m = {".": 0, "o": 1, "x": 2}
    return np.array([[m[c] for c in r] for r in rows], dtype=np.uint8)


# --- §10.2 온전 창 ---------------------------------------------------------------

def test_valid_window_requires_every_cell_to_be_a_die():
    """7x7 창 49칸이 전부 다이인 위치만 유효하다."""
    x = np.ones((1, 9, 9), np.uint8)
    m = valid_window_mask(x, k=7)
    assert m.sum() == 9          # 3x3 개의 중심 위치
    x2 = x.copy()
    x2[0, 0, 0] = 0              # 모서리 하나만 다이가 아니게
    assert valid_window_mask(x2, k=7).sum() == 8


def test_valid_window_counts_failed_dies_as_dies():
    """불량(2)도 다이다. 다이 없음(0)만 뺀다."""
    a = np.ones((1, 7, 7), np.uint8)
    b = np.full((1, 7, 7), 2, np.uint8)
    assert valid_window_mask(a, k=7).sum() == valid_window_mask(b, k=7).sum() == 1


def test_valid_window_is_empty_for_a_wafer_smaller_than_the_window():
    x = np.ones((1, 5, 5), np.uint8)
    assert valid_window_mask(x, k=7).sum() == 0


def test_valid_window_is_exact_integer_arithmetic():
    """`uniform_filter` 계열 반올림을 쓰면 안 된다(정정 8)."""
    rng = np.random.default_rng(0)
    x = (rng.random((4, 20, 20)) > 0.3).astype(np.uint8)
    m = valid_window_mask(x, k=7)
    die = (x > 0)
    for b in range(4):
        for i in range(3, 17):
            for j in range(3, 17):
                assert m[b, i, j] == bool(die[b, i - 3:i + 4, j - 3:j + 4].all())


# --- §10.1 수용영역 ---------------------------------------------------------------

def test_receptive_field_is_exactly_seven():
    """**입력 한 칸을 흔들면 출력이 반경 3 안에서만 변해야 한다.**

    수용영역이 더 크면 요소 B(선 L=11)와의 비교가 공정하지 않게 되고,
    "학습이 이겼다" 가 "창을 키워서 이겼다" 와 섞인다.
    """
    torch.manual_seed(0)
    enc = PatchEncoder().eval()
    x = torch.randn(1, 3, 21, 21, requires_grad=True)
    out = enc(x)
    out[0, :, 10, 10].sum().backward()
    g = x.grad[0].abs().sum(0).numpy()
    r = (RF - 1) // 2
    outside = g.copy()
    outside[10 - r:10 + r + 1, 10 - r:10 + r + 1] = 0.0
    assert outside.max() == 0.0, "수용영역 밖으로 새어 나갔다"
    ring = np.concatenate([g[10 - r, :], g[10 + r, :], g[:, 10 - r], g[:, 10 + r]])
    assert ring.max() > 0.0, "반경 3 에 아무 영향도 없다면 수용영역이 7 보다 작다"


def test_encoder_output_is_l2_normalised_per_location():
    torch.manual_seed(0)
    enc = PatchEncoder().eval()
    f = enc(torch.randn(2, 3, 16, 16))
    n = f.pow(2).sum(1).sqrt()
    assert torch.allclose(n, torch.ones_like(n), atol=1e-5)


def test_encoder_keeps_spatial_size():
    torch.manual_seed(0)
    f = PatchEncoder().eval()(torch.randn(2, 3, 16, 16))
    assert f.shape[0] == 2 and f.shape[2:] == (16, 16)


def test_onehot_has_no_channel_for_outside_the_wafer():
    """웨이퍼 밖은 세 채널 모두 0 이 아니라 채널 0 이 1 이다 — 범주가 셋이다."""
    x = wafer(["...", ".o.", "..x"])[None]
    h = to_onehot(x)
    assert h.shape == (1, 3, 3, 3)
    assert h[0, :, 0, 0].tolist() == [1.0, 0.0, 0.0]
    assert h[0, :, 1, 1].tolist() == [0.0, 1.0, 0.0]
    assert h[0, :, 2, 2].tolist() == [0.0, 0.0, 1.0]


# --- §10.3 마스킹 손실 -------------------------------------------------------------

def test_loss_ignores_positions_that_are_not_masked():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 8, 8)
    tgt = torch.randint(0, 3, (2, 8, 8))
    valid = torch.ones(2, 8, 8, dtype=torch.bool)
    m1 = torch.zeros(2, 8, 8, dtype=torch.bool)
    m1[:, :4, :4] = True
    a = masked_valid_ce(logits, tgt, m1, valid)
    logits2 = logits.clone()
    logits2[:, :, 4:, 4:] = torch.randn(2, 3, 4, 4)      # 안 가린 곳만 바꾼다
    assert torch.allclose(a, masked_valid_ce(logits2, tgt, m1, valid))


def test_loss_ignores_positions_outside_the_valid_window():
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 8, 8)
    tgt = torch.randint(0, 3, (1, 8, 8))
    mask = torch.ones(1, 8, 8, dtype=torch.bool)
    v = torch.zeros(1, 8, 8, dtype=torch.bool)
    v[0, 2:5, 2:5] = True
    a = masked_valid_ce(logits, tgt, mask, v)
    logits2 = logits.clone()
    logits2[0, :, 6, 6] = torch.randn(3)
    assert torch.allclose(a, masked_valid_ce(logits2, tgt, mask, v))


def test_loss_is_zero_when_nothing_is_both_masked_and_valid():
    logits = torch.randn(1, 3, 4, 4)
    tgt = torch.zeros(1, 4, 4, dtype=torch.long)
    z = torch.zeros(1, 4, 4, dtype=torch.bool)
    assert float(masked_valid_ce(logits, tgt, z, z)) == 0.0


# --- §10.4 뱅크와 kNN --------------------------------------------------------------

def test_knn_mean_distance_matches_brute_force():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(11, 5)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    b = rng.normal(size=(40, 5)); b /= np.linalg.norm(b, axis=1, keepdims=True)
    got = knn_mean_distance(q, b, k=3)
    want = np.array([np.sort(1.0 - b @ q[i])[:3].mean() for i in range(len(q))])
    assert np.allclose(got, want, atol=1e-10)


def test_knn_distance_is_zero_for_a_point_already_in_the_bank():
    """뱅크에 있는 점은 자기 자신까지의 거리가 0 이다.

    허용 오차가 1e-6 인 이유: 거리를 float32 내적으로 계산한다(1e-10 을 요구하면
    구현이 아니라 정밀도가 깨진다). **float64 로 바꾸면 5.2e13 FLOP 짜리 채점이
    두 배로 느려지므로, 정밀도가 아니라 허용 오차를 맞춘다.**
    이 결정을 여기 적어 두는 이유는 나중에 "왜 느슨한가" 를 다시 묻지 않기 위해서다.
    """
    rng = np.random.default_rng(1)
    b = rng.normal(size=(20, 4)); b /= np.linalg.norm(b, axis=1, keepdims=True)
    assert knn_mean_distance(b[:3], b, k=1) == pytest.approx(0.0, abs=1e-6)


def test_bank_is_deterministic_given_a_seed_and_respects_size():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(500, 6))
    a = build_bank(f, n=64, seed=3)
    assert a.shape == (64, 6)
    assert np.array_equal(a, build_bank(f, n=64, seed=3))
    assert not np.array_equal(a, build_bank(f, n=64, seed=4))


def test_bank_returns_everything_when_asked_for_more_than_available():
    f = np.arange(30, dtype=np.float64).reshape(10, 3)
    assert build_bank(f, n=999, seed=0).shape == (10, 3)


# --- §10.2 집계와 되돌림 ------------------------------------------------------------

def test_wafer_score_is_the_max_over_valid_locations():
    s = np.array([0.1, 0.9, 0.4, 5.0])
    v = np.array([True, True, True, False])          # 마지막은 유효하지 않다
    assert wafer_max_score(s, v) == pytest.approx(0.9)


def test_wafer_with_no_valid_location_is_not_silently_dropped():
    """유효 위치가 없는 웨이퍼(train-none 의 0.11%)도 점수를 받아야 한다."""
    s = np.array([0.3, 0.7])
    v = np.array([False, False])
    got = wafer_max_score(s, v)
    assert np.isfinite(got) and got == pytest.approx(0.7)
