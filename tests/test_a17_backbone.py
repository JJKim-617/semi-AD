"""TTA 는 백본에 상관없이 걸려야 한다.

`a17_tta` 의 가드는 백본이 resnet18 하나뿐이던 때 넣은 것이다. E18 로 계열이 늘었으니
그 가드가 곧 "백본 모델에는 TTA 를 못 건다" 가 된다 — 실제로 백본 6개가 전부 건너뛰어졌다.
dihedral 변환은 입력에 거는 것이라 백본과 무관하다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a17_tta import cache_tta_logits


@pytest.fixture
def tiny(tmp_path):
    from a3_train_wm811k_cls import build_model
    n, size, n_cls = 300, 16, 3
    rng = np.random.default_rng(0)
    cache = tmp_path / "c.npz"
    np.savez_compressed(cache, X=rng.integers(0, 3, (n, size, size), dtype=np.uint8),
                        y=(np.arange(n) % n_cls).astype(np.int64),
                        classes=np.array(["a", "b", "c"]))
    splits = tmp_path / "s.npz"
    np.savez(splits, val=np.arange(0, 150), test=np.arange(150, 300))
    return dict(cache=str(cache), splits=str(splits), out=str(tmp_path / "o"),
                tmp=tmp_path, n_cls=n_cls)


class TestBackboneSupport:
    def test_accepts_a_non_resnet_backbone(self, tiny):
        from a3_train_wm811k_cls import build_model
        ck = tiny["tmp"] / "sh.pt"
        torch.save(build_model(num_classes=tiny["n_cls"], backbone="shufflenet_v2").state_dict(), ck)
        p = cache_tta_logits(str(ck), tiny["cache"], tiny["splits"], tiny["out"], "sh",
                             backbone="shufflenet_v2")
        d = np.load(p)
        assert d["test_logits"].shape == (150, tiny["n_cls"])
        assert np.isfinite(d["test_logits"]).all()

    def test_defaults_to_resnet18(self, tiny):
        """기존 호출부가 backbone 없이 부른다."""
        from a3_train_wm811k_cls import build_model
        ck = tiny["tmp"] / "rn.pt"
        torch.save(build_model(num_classes=tiny["n_cls"]).state_dict(), ck)
        p = cache_tta_logits(str(ck), tiny["cache"], tiny["splits"], tiny["out"], "rn")
        assert np.load(p)["test_logits"].shape == (150, tiny["n_cls"])
