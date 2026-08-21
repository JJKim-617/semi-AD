"""자기지도 사전학습(masked die prediction) 테스트."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a11_ssl_pretrain import MaskedWaferNet, masked_ce_loss, random_patch_mask


class TestMask:
    def test_masks_roughly_the_requested_fraction(self):
        m = random_patch_mask((64, 64), patch=8, ratio=0.5,
                              rng=np.random.default_rng(0))
        assert 0.4 < m.mean() < 0.6

    def test_zero_ratio_masks_nothing(self):
        m = random_patch_mask((32, 32), patch=8, ratio=0.0,
                              rng=np.random.default_rng(0))
        assert not m.any()

    def test_full_ratio_masks_everything(self):
        m = random_patch_mask((32, 32), patch=8, ratio=1.0,
                              rng=np.random.default_rng(0))
        assert m.all()

    def test_is_deterministic_given_seed(self):
        a = random_patch_mask((32, 32), 8, 0.5, np.random.default_rng(3))
        b = random_patch_mask((32, 32), 8, 0.5, np.random.default_rng(3))
        assert np.array_equal(a, b)

    def test_masks_contiguous_patches_not_scattered_pixels(self):
        """픽셀 단위로 흩뿌리면 이웃에서 베낄 수 있어 과제가 너무 쉬워진다."""
        m = random_patch_mask((32, 32), patch=8, ratio=0.25,
                              rng=np.random.default_rng(1))
        blocks = m.reshape(4, 8, 4, 8).transpose(0, 2, 1, 3).reshape(16, 64)
        assert set(blocks.sum(1).tolist()) <= {0, 64}

    def test_rejects_patch_not_dividing_shape(self):
        with pytest.raises(ValueError):
            random_patch_mask((30, 30), patch=8, ratio=0.5,
                              rng=np.random.default_rng(0))


class TestLoss:
    def test_unmasked_positions_do_not_contribute(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 8, 8)
        target = torch.randint(0, 3, (2, 8, 8))
        mask = torch.zeros(2, 8, 8, dtype=torch.bool)
        mask[:, :4, :] = True
        a = masked_ce_loss(logits, target, mask)
        logits2 = logits.clone()
        logits2[:, :, 4:, :] += 100.0        # 가리지 않은 곳만 크게 바꾼다
        assert torch.allclose(a, masked_ce_loss(logits2, target, mask))

    def test_perfect_prediction_on_masked_gives_near_zero(self):
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        logits = torch.full((1, 3, 4, 4), -10.0)
        logits[:, 0] = 10.0
        mask = torch.ones(1, 4, 4, dtype=torch.bool)
        assert masked_ce_loss(logits, target, mask).item() < 1e-3

    def test_empty_mask_returns_zero_not_nan(self):
        logits = torch.randn(1, 3, 4, 4)
        target = torch.randint(0, 3, (1, 4, 4))
        loss = masked_ce_loss(logits, target, torch.zeros(1, 4, 4, dtype=torch.bool))
        assert torch.isfinite(loss) and loss.item() == 0.0


class TestModel:
    def test_reconstructs_input_resolution(self):
        m = MaskedWaferNet()
        assert m(torch.zeros(2, 3, 64, 64)).shape == (2, 3, 64, 64)

    def test_encoder_matches_classifier_stem(self):
        """사전학습한 인코더를 분류기로 옮기려면 구조가 같아야 한다."""
        m = MaskedWaferNet()
        assert m.encoder.conv1.kernel_size == (3, 3)
        assert m.encoder.conv1.stride == (1, 1)
        assert isinstance(m.encoder.maxpool, torch.nn.Identity)

    def test_encoder_state_dict_loads_into_classifier(self):
        from a3_train_wm811k_cls import build_model
        m = MaskedWaferNet()
        clf = build_model(num_classes=9, pretrained=False)
        missing, unexpected = clf.load_state_dict(m.encoder.state_dict(), strict=False)
        assert all(k.startswith("fc.") for k in missing)
