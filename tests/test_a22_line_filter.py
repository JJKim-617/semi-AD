"""방향성 선형 필터 (E21 진단용 이식).

E20 진단에서 나온 것: **Scratch 의 국소 밀도 중앙값 0.500 은 none 의 0.400 다음으로 낮다.**
가는 선은 정사각 창을 채우지 못하므로 밀도로는 원리적으로 안 잡힌다.
창 모양을 선으로 바꾸면 같은 선이 1.0 이 된다 — 그 성질을 시험으로 박는다.

구현은 `Semi-AD-OOD/src/a27_line_filter.py` 에서 옮겨 왔다(그쪽은 읽기만 했다).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a21_density import local_fail_density_map  # noqa: E402
from a22_line_filter import line_density_map, line_kernels  # noqa: E402


def test_kernels_pass_through_the_centre_and_stay_thin():
    ks = line_kernels(length=7, n_orient=8, width=1)
    assert len(ks) == 8
    for k in ks:
        assert k.shape == (7, 7)
        assert k[3, 3] == 1                    # 중심을 지난다
        assert 5 <= int(k.sum()) <= 7          # 이산화로 줄 수는 있어도 두꺼워지지 않는다


def test_even_length_is_rejected():
    with pytest.raises(ValueError):
        line_kernels(length=6)


def test_thin_line_saturates_where_square_window_dilutes():
    """이 필터를 만든 이유 자체다. 1픽셀 폭 길이 7 선.

    정사각 7x7 창은 7/49 로 희석하고, 선 모양 창은 1.0 을 낸다.
    """
    x = np.ones((1, 21, 21), np.uint8)
    x[0, 10, 7:14] = 2                          # 가로 1픽셀 선, 길이 7
    assert np.isclose(local_fail_density_map(x, k=7).max(), 7 / 49)
    assert np.isclose(line_density_map(x, length=7, n_orient=8).max(), 1.0)


def test_blob_does_not_saturate_but_line_does():
    """대비가 핵심이다. 같은 필터에서 3x3 덩어리는 포화하지 않고 가는 선은 포화한다.

    덩어리의 정확한 값은 방향별 커널의 이산화에 달려 있다 — 비스듬한 커널은
    겹치는 칸 때문에 7칸보다 작아질 수 있어(모듈 docstring) 3/5 = 0.6 이 나온다.
    그 값을 고정하지 않고 **1.0 에 못 미친다**는 것만 박는다.
    """
    blob = np.ones((1, 21, 21), np.uint8)
    blob[0, 9:12, 9:12] = 2
    line = np.ones((1, 21, 21), np.uint8)
    line[0, 10, 7:14] = 2
    b = line_density_map(blob, length=7, n_orient=8).max()
    ln = line_density_map(line, length=7, n_orient=8).max()
    assert b < 0.7
    assert np.isclose(ln, 1.0)
    assert ln > b


def test_equivariant_under_rot90():
    """8방향은 [0,pi) 를 22.5 도로 나눈 것이라 90 도 회전이 방향 집합을 자기 자신으로 보낸다."""
    x = np.random.default_rng(0).integers(0, 3, size=(2, 17, 17), dtype=np.uint8)
    m = line_density_map(x, length=7, n_orient=8)
    rot = np.ascontiguousarray(np.rot90(x, 1, (1, 2)))
    assert np.allclose(line_density_map(rot, length=7, n_orient=8),
                       np.rot90(m, 1, (1, 2)))


def test_die_free_cells_are_zero():
    x = np.full((1, 16, 16), 2, np.uint8)
    x[0, :, :4] = 0
    assert line_density_map(x, length=7)[0, :, :4].max() == 0.0


def test_min_dies_rejects_truncated_windows():
    """가장자리에서 창이 잘리면 연속 3개만 불량이어도 1.0 이 된다 — 그것을 버린다.

    2026-08-23 진단: 이 degeneracy 때문에 **참 none 의 71.1% 가 포화**했다
    (포화 칸의 창 안 다이 최빈값이 3). `min_dies=length` 는 온전한 창만 본다.
    """
    x = np.zeros((1, 21, 21), np.uint8)
    x[0, 0, :3] = 2                              # 모서리에 3연속 불량, 나머지는 다이 없음
    assert np.isclose(line_density_map(x, length=7, n_orient=8).max(), 1.0)
    assert line_density_map(x, length=7, n_orient=8, min_dies=7).max() == 0.0


def test_min_dies_keeps_full_windows():
    x = np.ones((1, 21, 21), np.uint8)
    x[0, 10, 7:14] = 2
    assert np.isclose(
        line_density_map(x, length=7, n_orient=8, min_dies=7).max(), 1.0)


# --- 텐서 경로 (학습용) --------------------------------------------------------
#
# numpy 경로는 29,000장에 28초다 — 에폭마다 43,223장을 돌리면 학습보다 비싸다.
# 8방향 커널을 그대로 conv2d 필터 뱅크로 놓으면 같은 값이 나온다.
# **정의는 numpy 쪽이고 텐서는 사본이다.** 직접 대조한다.

def test_torch_line_density_matches_numpy():
    import torch

    from a22_line_filter import line_density_stack_torch

    x = np.random.default_rng(6).integers(0, 3, size=(5, 24, 24), dtype=np.uint8)
    x[:, :, :4] = 0
    ref = line_density_map(x, length=7, n_orient=8, min_dies=7)
    got = line_density_stack_torch(torch.from_numpy(x.astype(np.int64)), ls=(7,))
    assert got.shape == (5, 1, 24, 24)
    assert np.allclose(got.numpy()[:, 0], ref, atol=1e-5)


def test_torch_line_density_two_lengths_ordered():
    import torch

    from a22_line_filter import line_density_stack_torch

    x = np.random.default_rng(7).integers(0, 3, size=(3, 20, 20), dtype=np.uint8)
    got = line_density_stack_torch(torch.from_numpy(x.astype(np.int64)), ls=(7, 11)).numpy()
    assert np.allclose(got[:, 0], line_density_map(x, length=7, min_dies=7), atol=1e-5)
    assert np.allclose(got[:, 1], line_density_map(x, length=11, min_dies=11), atol=1e-5)


def test_train_and_predict_with_line_channels():
    """증강 뒤 선 채널을 계산하는 경로가 끝까지 돈다 (CPU). GPU 스모크는 따로 돌린다."""
    import torch

    from a3_train_wm811k_cls import build_model, encode, encode_device, predict, train_one_epoch

    torch.manual_seed(0)
    x = np.random.default_rng(0).integers(0, 3, size=(16, 32, 32), dtype=np.uint8)
    y = np.arange(16) % 9
    assert encode(x, density_ks=(5,), line_ls=(11,)).shape == (16, 5, 32, 32)
    assert np.allclose(encode_device(x, (5,), "cpu", line_ls=(11,)).numpy(),
                       encode(x, (5,), line_ls=(11,)).numpy(), atol=1e-5)
    m = build_model(in_channels=4)
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    loss = train_one_epoch(m, x, y, opt, batch_size=8, device="cpu", augment="dihedral",
                           extra_augment=("translate",), line_ls=(11,),
                           rng=np.random.default_rng(0))
    assert np.isfinite(loss)
    assert predict(m, x, 8, "cpu", line_ls=(11,)).shape == (16,)
