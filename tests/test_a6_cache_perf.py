"""로짓 캐시가 웨이퍼 배열을 배치마다 다시 읽지 않아야 한다.

npz 는 지연 로딩이라 `d["X"]` 는 접근할 때마다 배열 전체를 압축 해제한다.
실측으로 wm811k_64pad.npz 의 `d["X"][batch]` 한 번이 1.699초 걸렸다(708MB).
test 232 배치 + val 22 배치면 모델 하나당 432초가 압축 해제에만 쓰인다.
배열을 루프 밖으로 끌어내면 배치당 0.00097초가 된다 — 1750배.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def tiny_run(tmp_path):
    """배치가 여러 개 나오도록 512 표본보다 크게 만든다. 그래야 재읽기가 드러난다."""
    import torch
    from a3_train_wm811k_cls import build_model

    n, size, n_cls = 1400, 8, 3
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, (n, size, size), dtype=np.uint8)
    y = (np.arange(n) % n_cls).astype(np.int64)

    cache = tmp_path / "cache.npz"
    np.savez_compressed(cache, X=X, y=y, classes=np.array(["a", "b", "c"]))
    splits = tmp_path / "splits.npz"
    np.savez(splits, val=np.arange(0, 700), test=np.arange(700, 1400))
    ckpt = tmp_path / "m.pt"
    torch.save(build_model(num_classes=n_cls, pretrained=False).state_dict(), ckpt)
    return dict(cache=str(cache), splits=str(splits), ckpt=str(ckpt),
                out=str(tmp_path / "out"), n_cls=n_cls)


def _count_npz_reads(monkeypatch):
    from numpy.lib.npyio import NpzFile
    seen = []
    original = NpzFile.__getitem__

    def counting(self, key):
        seen.append(key)
        return original(self, key)

    monkeypatch.setattr(NpzFile, "__getitem__", counting)
    return seen


class TestLogitCacheDoesNotRereadTheWaferArray:
    def test_reads_the_wafer_array_a_bounded_number_of_times(self, tiny_run, monkeypatch):
        """배치마다 읽으면 700/512 -> 2 배치 x 2 split = 4회 이상이 된다."""
        from a6_perclass_offset import cache_logits

        seen = _count_npz_reads(monkeypatch)
        cache_logits(tiny_run["ckpt"], tiny_run["cache"], tiny_run["splits"],
                     tiny_run["out"], "t")
        assert seen.count("X") <= 2, f"X 를 {seen.count('X')}번 읽었다 (배치마다 재읽기)"

    def test_still_produces_logits_for_both_splits(self, tiny_run):
        from a6_perclass_offset import cache_logits

        path = cache_logits(tiny_run["ckpt"], tiny_run["cache"], tiny_run["splits"],
                            tiny_run["out"], "t2")
        d = np.load(path)
        assert d["val_logits"].shape == (700, tiny_run["n_cls"])
        assert d["test_logits"].shape == (700, tiny_run["n_cls"])
        assert d["val_y"].shape == (700,) and d["test_y"].shape == (700,)
        assert np.isfinite(d["test_logits"]).all()
