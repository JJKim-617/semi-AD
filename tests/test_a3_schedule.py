"""학습률 스케줄 테스트."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a3_train_wm811k_cls import cosine_lr


def test_starts_at_base_lr():
    assert cosine_lr(0, total=10, base=0.1) == pytest.approx(0.1)


def test_ends_near_zero():
    assert cosine_lr(10, total=10, base=0.1) == pytest.approx(0.0, abs=1e-9)


def test_is_monotonically_decreasing():
    v = [cosine_lr(e, 20, 0.1) for e in range(21)]
    assert all(a >= b for a, b in zip(v, v[1:]))


def test_halfway_is_half_of_base():
    assert cosine_lr(10, total=20, base=0.1) == pytest.approx(0.05)


def test_warmup_ramps_from_below_base():
    assert cosine_lr(0, total=20, base=0.1, warmup=3) < 0.1
    assert cosine_lr(3, total=20, base=0.1, warmup=3) == pytest.approx(0.1)


def test_rejects_nonpositive_total():
    with pytest.raises(ValueError):
        cosine_lr(0, total=0, base=0.1)
