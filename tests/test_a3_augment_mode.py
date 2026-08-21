"""증강은 입력 표현에 맞아야 한다.

E15 를 돌리다 잡은 결함이다. `dihedral` 은 rot90 을 쓰는데, 극좌표 이미지에서
rot90 은 **반지름 축과 각도 축을 맞바꾼다.** 극좌표에서 유효한 대칭은 두 가지뿐이다.

  - 회전  -> 각도 축(열)의 순환이동
  - 반사  -> 각도 축의 뒤집기

반지름 축(행)은 뒤집으면 안 된다. 웨이퍼 중심과 가장자리가 뒤바뀐다.
dihedral 8개 중 극좌표에서 의미가 유지되는 것은 항등과 각도 뒤집기 둘뿐이고
나머지 6개는 구조를 파괴한다.

실제로 극좌표 학습이 불안정했다. lr 이 0 에 가까워진 epoch 36~40 에서 val macro-F1 이
0.872 -> 0.612 -> 0.718 -> 0.746 -> 0.699 로 튀었다. 카르테시안 실행은 같은 구간에서
0.94 근처로 평평하다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a3_train_wm811k_cls import angular_augment, augment_batch


def sample(n=4, h=8, w=8, seed=0):
    return np.random.default_rng(seed).integers(0, 3, (n, h, w), dtype=np.uint8)


class TestAngularAugment:
    def test_preserves_the_radius_axis_content_of_each_row(self):
        """행은 반지름이다. 행 안에서 섞이는 건 되지만 행 사이는 안 된다."""
        x = sample()
        out = angular_augment(x, shift=3, flip=False)
        assert np.array_equal(np.sort(out, axis=2), np.sort(x, axis=2))

    def test_never_reverses_the_radius_axis(self):
        """중심과 가장자리가 뒤바뀌면 안 된다."""
        x = np.zeros((1, 4, 4), np.uint8)
        x[0, 0, :] = 2                       # 중심 행만 불량
        for shift in range(4):
            for flip in (False, True):
                out = angular_augment(x, shift=shift, flip=flip)
                assert (out[0, 0] == 2).all(), "중심 행이 다른 곳으로 갔다"
                assert (out[0, 1:] == 0).all()

    def test_shift_is_circular(self):
        x = np.arange(8, dtype=np.uint8).reshape(1, 1, 8)
        assert np.array_equal(angular_augment(x, shift=1, flip=False), np.roll(x, 1, axis=2))

    def test_zero_shift_without_flip_is_the_identity(self):
        x = sample()
        assert np.array_equal(angular_augment(x, shift=0, flip=False), x)

    def test_flip_reverses_the_angle_axis(self):
        x = sample()
        assert np.array_equal(angular_augment(x, shift=0, flip=True), x[:, :, ::-1])

    def test_invents_no_new_cell_values(self):
        x = sample()
        out = angular_augment(x, shift=5, flip=True)
        assert set(np.unique(out)).issubset(set(np.unique(x)))

    def test_keeps_shape_and_dtype(self):
        x = sample()
        out = angular_augment(x, shift=2, flip=True)
        assert out.shape == x.shape and out.dtype == x.dtype


class TestAugmentBatchDispatch:
    def test_dihedral_mode_can_transpose_the_axes(self):
        """카르테시안에서는 rot90 이 정당하다. 이 성질이 살아 있어야 한다."""
        rng = np.random.default_rng(0)
        x = sample(n=1, h=8, w=8, seed=7)
        seen = {augment_batch(x, "dihedral", rng).tobytes() for _ in range(200)}
        assert len(seen) > 2, "dihedral 인데 변환이 거의 안 일어난다"

    def test_angular_mode_never_transposes_the_axes(self, ):
        rng = np.random.default_rng(0)
        x = np.zeros((1, 4, 8), np.uint8)
        x[0, 0, :] = 2
        for _ in range(200):
            out = augment_batch(x, "angular", rng)
            assert out.shape == x.shape
            assert (out[0, 0] == 2).all(), "반지름 축이 섞였다"

    def test_none_mode_returns_the_input_unchanged(self):
        rng = np.random.default_rng(0)
        x = sample()
        assert np.array_equal(augment_batch(x, "none", rng), x)

    def test_rejects_an_unknown_mode(self):
        with pytest.raises(ValueError):
            augment_batch(sample(), "spiral", np.random.default_rng(0))

    def test_angular_mode_actually_varies(self):
        rng = np.random.default_rng(1)
        x = sample(n=1, h=8, w=8, seed=9)
        seen = {augment_batch(x, "angular", rng).tobytes() for _ in range(200)}
        assert len(seen) > 4, "각도 증강이 거의 항등이다"
