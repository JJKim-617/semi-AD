"""웨이퍼 크기를 보조 입력으로 쓰는 모델 테스트."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a10_size_feature import SizeAwareNet, standardize_size


class TestStandardize:
    def test_train_stats_give_zero_mean_unit_std(self):
        x = np.array([100.0, 200.0, 300.0, 400.0])
        z, stats = standardize_size(x)
        assert z.mean() == pytest.approx(0.0, abs=1e-9)
        assert z.std() == pytest.approx(1.0, abs=1e-9)

    def test_applying_saved_stats_does_not_recompute(self):
        """test 통계를 쓰면 누수다. 저장된 train 통계를 그대로 적용해야 한다."""
        tr = np.array([100.0, 200.0, 300.0])
        _, stats = standardize_size(tr)
        te = np.array([1000.0, 2000.0])
        z, _ = standardize_size(te, stats=stats)
        assert z.mean() != pytest.approx(0.0, abs=1e-6)

    def test_constant_input_does_not_divide_by_zero(self):
        z, _ = standardize_size(np.full(5, 42.0))
        assert np.isfinite(z).all()


class TestModel:
    def test_outputs_one_logit_per_class(self):
        m = SizeAwareNet(num_classes=9)
        out = m(torch.zeros(4, 3, 64, 64), torch.zeros(4, 1))
        assert out.shape == (4, 9)

    def test_size_input_changes_the_output(self):
        """크기를 무시하면 이 실험 자체가 성립하지 않는다."""
        torch.manual_seed(0)
        m = SizeAwareNet(num_classes=9).eval()
        x = torch.randn(2, 3, 64, 64)
        a = m(x, torch.zeros(2, 1))
        b = m(x, torch.full((2, 1), 3.0))
        assert not torch.allclose(a, b)

    def test_image_input_still_matters(self):
        torch.manual_seed(0)
        m = SizeAwareNet(num_classes=9).eval()
        s = torch.zeros(2, 1)
        a = m(torch.zeros(2, 3, 64, 64), s)
        b = m(torch.randn(2, 3, 64, 64), s)
        assert not torch.allclose(a, b)

    def test_stem_is_modified_for_low_resolution(self):
        m = SizeAwareNet(num_classes=9)
        assert m.backbone.conv1.kernel_size == (3, 3)
        assert isinstance(m.backbone.maxpool, torch.nn.Identity)

    def test_gradient_reaches_the_size_branch(self):
        m = SizeAwareNet(num_classes=9)
        s = torch.zeros(2, 1, requires_grad=True)
        m(torch.zeros(2, 3, 64, 64), s).sum().backward()
        assert s.grad is not None and torch.isfinite(s.grad).all()
