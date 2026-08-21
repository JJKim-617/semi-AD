"""a3 학습 루프와 보조 함수 테스트."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a3_train_wm811k_cls import (build_model, class_weights, dihedral,
                                 predict, train_one_epoch)


class TestClassWeights:
    def test_rare_class_gets_larger_weight(self):
        y = np.array([0] * 90 + [1] * 10)
        w = class_weights(y, num_classes=2)
        assert w[1] > w[0]

    def test_balanced_labels_give_equal_weights(self):
        y = np.array([0, 0, 1, 1])
        w = class_weights(y, num_classes=2)
        assert w[0] == pytest.approx(w[1])

    def test_absent_class_does_not_produce_inf(self):
        y = np.array([0, 0, 0])
        w = class_weights(y, num_classes=3)
        assert torch.isfinite(w).all()


class TestDihedral:
    def test_preserves_shape(self):
        x = np.arange(9, dtype=np.uint8).reshape(1, 3, 3) % 3
        assert dihedral(x, k=1, flip=False).shape == x.shape

    def test_preserves_categorical_values(self):
        """회전, 반전은 값을 만들어내면 안 된다. 보간이 개입하면 실패한다."""
        rng = np.random.default_rng(0)
        x = rng.integers(0, 3, size=(4, 8, 8), dtype=np.uint8)
        for k in range(4):
            for flip in (False, True):
                out = dihedral(x, k=k, flip=flip)
                assert set(np.unique(out).tolist()) <= {0, 1, 2}

    def test_four_rotations_return_to_original(self):
        rng = np.random.default_rng(1)
        x = rng.integers(0, 3, size=(2, 5, 5), dtype=np.uint8)
        out = x
        for _ in range(4):
            out = dihedral(out, k=1, flip=False)
        assert np.array_equal(out, x)

    def test_identity_transform_is_noop(self):
        rng = np.random.default_rng(2)
        x = rng.integers(0, 3, size=(2, 5, 5), dtype=np.uint8)
        assert np.array_equal(dihedral(x, k=0, flip=False), x)


class TestTrainLoop:
    def _toy(self, n=64):
        """클래스가 좌우 절반의 불량 위치로 구분되는 학습 가능한 장난감 문제."""
        rng = np.random.default_rng(0)
        X = np.ones((n, 16, 16), dtype=np.uint8)
        y = rng.integers(0, 2, size=n)
        for i, c in enumerate(y):
            if c == 0:
                X[i, :, :4] = 2
            else:
                X[i, :, -4:] = 2
        return X, y.astype(np.int64)

    def test_one_epoch_reduces_loss_on_learnable_data(self):
        X, y = self._toy()
        torch.manual_seed(0)
        m = build_model(num_classes=2, pretrained=False)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        first = train_one_epoch(m, X, y, opt, batch_size=16, device="cpu")
        for _ in range(3):
            last = train_one_epoch(m, X, y, opt, batch_size=16, device="cpu")
        assert last < first

    def test_predict_returns_one_label_per_sample(self):
        X, y = self._toy(n=8)
        m = build_model(num_classes=2, pretrained=False)
        pred = predict(m, X, batch_size=4, device="cpu")
        assert pred.shape == (8,)
        assert set(np.unique(pred).tolist()) <= {0, 1}

    def test_predict_does_not_change_weights(self):
        X, _ = self._toy(n=8)
        m = build_model(num_classes=2, pretrained=False)
        before = m.fc.weight.detach().clone()
        predict(m, X, batch_size=4, device="cpu")
        assert torch.allclose(before, m.fc.weight)
