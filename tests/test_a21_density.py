"""국소 불량 밀도 채널 (E20).

밀도 맵 자체는 학습이 없는 순수 계산이므로 **정답을 손으로 적을 수 있다.**
아래 시험은 전부 해석적으로 계산한 값과 대조한다 — 회귀 방지가 아니라 정의 확인이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a21_density import (  # noqa: E402
    density_stack,
    density_stack_torch,
    local_fail_density_map,
)
from a3_train_wm811k_cls import (  # noqa: E402
    build_model,
    encode,
    encode_device,
    predict,
    to_onehot,
    train_one_epoch,
)


# --- 밀도 맵의 정의 -----------------------------------------------------------

def test_all_normal_gives_zero_density():
    """불량이 하나도 없으면 어디서도 0 이다."""
    assert local_fail_density_map(np.ones((1, 16, 16), np.uint8), k=5).max() == 0.0


def test_die_free_cells_are_exactly_zero():
    """다이가 없는 칸은 정확히 0. 값이 새면 웨이퍼 바깥을 가리키게 된다."""
    x = np.full((1, 16, 16), 2, np.uint8)
    x[0, :, :4] = 0
    m = local_fail_density_map(x, k=5)
    assert m[0, :, :4].max() == 0.0
    assert np.isclose(m[0, 8, 8], 1.0)  # 창 안 다이가 전부 불량이면 1


def test_window_size_sets_the_scale():
    """3x3 불량 덩어리 하나. 창이 커질수록 최대 밀도가 희석된다 — 값이 해석적으로 정해진다."""
    x = np.ones((1, 21, 21), np.uint8)
    x[0, 9:12, 9:12] = 2
    assert np.isclose(local_fail_density_map(x, k=5).max(), 9 / 25)
    assert np.isclose(local_fail_density_map(x, k=7).max(), 9 / 49)


def test_normalized_by_die_count_not_window_area():
    """가장자리에서 희석되지 않는다. 모서리 3x3 덩어리만 있는 웨이퍼에서 1.0 이 나와야 한다."""
    x = np.zeros((1, 20, 20), np.uint8)
    x[0, :3, :3] = 2
    assert np.isclose(local_fail_density_map(x, k=5)[0, 1, 1], 1.0)


def test_equivariant_under_dihedral():
    """회전, 반사와 교환된다. 그래서 증강 뒤에 계산해도 되고 TTA 도 그대로 성립한다."""
    x = np.random.default_rng(0).integers(0, 3, size=(2, 17, 17), dtype=np.uint8)
    m = local_fail_density_map(x, k=5)
    for k in range(4):
        rot = np.ascontiguousarray(np.rot90(x, k, (1, 2)))
        assert np.allclose(local_fail_density_map(rot, k=5), np.rot90(m, k, (1, 2)))
    flipped = np.ascontiguousarray(x[:, :, ::-1])
    assert np.allclose(local_fail_density_map(flipped, k=5), m[:, :, ::-1])


def test_chunking_does_not_change_result():
    x = np.random.default_rng(1).integers(0, 3, size=(9, 13, 13), dtype=np.uint8)
    assert np.allclose(local_fail_density_map(x, k=5, chunk=2),
                       local_fail_density_map(x, k=5, chunk=9))


def test_density_stack_shape_and_order():
    x = np.ones((3, 12, 12), np.uint8)
    x[:, 5:8, 5:8] = 2
    s = density_stack(x, ks=(5, 7))
    assert s.shape == (3, 2, 12, 12)
    assert s.dtype == np.float32
    assert np.allclose(s[:, 0], local_fail_density_map(x, k=5))
    assert np.allclose(s[:, 1], local_fail_density_map(x, k=7))


def test_even_window_is_rejected():
    """짝수 창은 중심이 없어 등변성이 깨진다. 조용히 어긋나느니 터뜨린다."""
    with pytest.raises(ValueError):
        density_stack(np.ones((1, 8, 8), np.uint8), ks=(4,))


# --- 입력 인코딩 ---------------------------------------------------------------

def test_encode_defaults_to_plain_onehot():
    """기본값이 바뀌면 기존 체크포인트 41개를 못 읽는다."""
    x = np.ones((2, 8, 8), np.uint8)
    assert np.allclose(encode(x).numpy(), to_onehot(x).numpy())


def test_encode_appends_density_after_onehot():
    x = np.ones((2, 12, 12), np.uint8)
    x[:, 5:8, 5:8] = 2
    t = encode(x, density_ks=(5, 7))
    assert t.shape == (2, 5, 12, 12)
    assert np.allclose(t[:, :3].numpy(), to_onehot(x).numpy())
    assert np.allclose(t[:, 3].numpy(), density_stack(x, (5,))[:, 0], atol=1e-6)
    assert np.allclose(t[:, 4].numpy(), density_stack(x, (7,))[:, 0], atol=1e-6)


def test_shuffled_density_breaks_only_the_correspondence():
    """대조군의 성질: one-hot 과 채널 통계는 그대로고 짝만 어긋난다."""
    x = np.random.default_rng(0).integers(0, 3, size=(64, 16, 16), dtype=np.uint8)
    real = encode(x, density_ks=(5,))
    fake = encode(x, density_ks=(5,), shuffle_seed=7)
    assert np.allclose(real[:, :3].numpy(), fake[:, :3].numpy())
    assert not np.allclose(real[:, 3].numpy(), fake[:, 3].numpy())
    assert np.isclose(float(real[:, 3].sum()), float(fake[:, 3].sum()), rtol=1e-5)


def test_shuffle_is_deterministic():
    x = np.random.default_rng(2).integers(0, 3, size=(32, 16, 16), dtype=np.uint8)
    a = encode(x, density_ks=(5,), shuffle_seed=7)
    b = encode(x, density_ks=(5,), shuffle_seed=7)
    assert np.allclose(a.numpy(), b.numpy())


# --- 텐서 경로는 numpy 기준 구현과 같아야 한다 --------------------------------------
#
# scipy 구현이 정의고 텐서 구현은 속도용 사본이다. 1 에폭 스모크에서 numpy 경로가
# 96s/epoch(기준선 38s)였다 — 밀도 계산이 학습보다 비쌌다. 둘이 어긋나면
# 학습이 보는 것과 이 파일이 검증한 정의가 달라지므로 직접 대조한다.

def test_torch_density_matches_scipy_reference():
    x = np.random.default_rng(3).integers(0, 3, size=(8, 24, 24), dtype=np.uint8)
    x[:, :, :5] = 0                                  # 다이 없는 영역도 섞는다
    ref = density_stack(x, ks=(5, 7))
    got = density_stack_torch(torch.from_numpy(x.astype(np.int64)), ks=(5, 7))
    assert got.shape == ref.shape
    assert np.allclose(got.numpy(), ref, atol=1e-5)


def test_encode_device_matches_encode():
    x = np.random.default_rng(4).integers(0, 3, size=(6, 20, 20), dtype=np.uint8)
    assert np.allclose(encode_device(x, (5, 7), "cpu").numpy(),
                       encode(x, (5, 7)).numpy(), atol=1e-5)
    assert np.allclose(encode_device(x, (), "cpu").numpy(), to_onehot(x).numpy())


def test_encode_device_shuffle_matches_encode_shuffle():
    """대조군 경로도 두 구현이 같은 순열을 써야 짝지은 비교가 성립한다."""
    x = np.random.default_rng(5).integers(0, 3, size=(16, 16, 16), dtype=np.uint8)
    assert np.allclose(encode_device(x, (5,), "cpu", shuffle_seed=11).numpy(),
                       encode(x, (5,), shuffle_seed=11).numpy(), atol=1e-5)


# --- 모델 입력 채널 -------------------------------------------------------------

def test_build_model_default_input_is_three_channels():
    assert build_model().conv1.in_channels == 3


def test_build_model_accepts_extra_input_channels():
    m = build_model(in_channels=5)
    assert m.conv1.in_channels == 5
    assert m(torch.zeros(2, 5, 32, 32)).shape == (2, 9)


def test_other_backbones_reject_extra_channels():
    """a19 백본들은 3채널 전제로 stem 이 수정돼 있다. 조용히 틀리느니 거절한다."""
    with pytest.raises(NotImplementedError):
        build_model(in_channels=5, backbone="shufflenet_v2")


# --- 학습 경로 -----------------------------------------------------------------

def test_train_and_predict_with_density_channels():
    """증강(dihedral + translate) 뒤 밀도를 계산하는 경로가 끝까지 돈다."""
    torch.manual_seed(0)
    x = np.random.default_rng(0).integers(0, 3, size=(16, 32, 32), dtype=np.uint8)
    y = np.arange(16) % 9
    m = build_model(in_channels=5)
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    loss = train_one_epoch(m, x, y, opt, batch_size=8, device="cpu", augment="dihedral",
                           extra_augment=("translate",), density_ks=(5, 7),
                           rng=np.random.default_rng(0))
    assert np.isfinite(loss)
    assert predict(m, x, 8, "cpu", density_ks=(5, 7)).shape == (16,)
