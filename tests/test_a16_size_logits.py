"""SizeAwareNet 체크포인트의 로짓을 캐시한다.

지금까지 시험한 앙상블 다양성 축은 입력 표현 하나뿐이다(E14). 구조 다양성은 안 재봤다.
E11 이 남긴 SizeAwareNet 체크포인트 두 개는 백본은 같지만 헤드가 다른 모델이라
구조 축의 첫 표본이 된다. E14 에서 단독 성능과 앙상블 기여가 무관하다는 것이
확인됐으므로, E11 이 기각됐다는 사실은 여기에 아무 상관이 없다.

a6 의 cache_logits 는 plain ResNet 을 만들므로 이 체크포인트를 못 읽는다.
그리고 SizeAwareNet.forward 는 (이미지, 크기) 두 입력을 받는다. 크기 표준화 통계는
학습 때와 **똑같이** train split 에서만 계산해야 하고, 대조군은 같은 seed 의 같은
순열로 셔플해야 한다. 그러지 않으면 학습 때와 다른 모델을 평가하게 된다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a16_size_logits import cache_size_logits, size_inputs_for


@pytest.fixture
def tiny(tmp_path):
    import torch
    from a10_size_feature import SizeAwareNet

    n, size, n_cls = 300, 8, 3
    rng = np.random.default_rng(0)
    cache = tmp_path / "cache.npz"
    np.savez_compressed(
        cache,
        X=rng.integers(0, 3, (n, size, size), dtype=np.uint8),
        y=(np.arange(n) % n_cls).astype(np.int64),
        die_size=(rng.random(n) * 5000 + 100).astype(np.float32),
        classes=np.array(["a", "b", "c"]),
    )
    splits = tmp_path / "splits.npz"
    np.savez(splits, train=np.arange(0, 150), val=np.arange(150, 220), test=np.arange(220, 300))
    ckpt = tmp_path / "m.pt"
    torch.save(SizeAwareNet(num_classes=n_cls).state_dict(), ckpt)
    return dict(cache=str(cache), splits=str(splits), ckpt=str(ckpt),
                out=str(tmp_path / "out"), n_cls=n_cls, n=n)


class TestSizeInputs:
    def test_standardises_with_train_statistics_only(self, tiny):
        """test 통계를 쓰면 누수다. train 부분의 평균이 0 이어야 한다."""
        s = size_inputs_for(tiny["cache"], tiny["splits"], shuffle=False, seed=0)
        tr = np.load(tiny["splits"])["train"]
        assert s[tr].mean() == pytest.approx(0.0, abs=1e-9)
        assert s[tr].std() == pytest.approx(1.0, abs=1e-9)

    def test_covers_every_sample_not_just_the_train_split(self, tiny):
        s = size_inputs_for(tiny["cache"], tiny["splits"], shuffle=False, seed=0)
        assert s.shape == (tiny["n"],)

    def test_the_shuffled_control_is_a_permutation_of_the_same_values(self, tiny):
        a = size_inputs_for(tiny["cache"], tiny["splits"], shuffle=False, seed=0)
        b = size_inputs_for(tiny["cache"], tiny["splits"], shuffle=True, seed=0)
        assert not np.array_equal(a, b)
        assert np.allclose(np.sort(a), np.sort(b))

    def test_the_shuffle_is_reproducible_from_the_seed(self, tiny):
        """학습 때 쓴 순열을 그대로 되살려야 같은 모델을 평가하는 것이다."""
        a = size_inputs_for(tiny["cache"], tiny["splits"], shuffle=True, seed=0)
        b = size_inputs_for(tiny["cache"], tiny["splits"], shuffle=True, seed=0)
        assert np.array_equal(a, b)

    def test_a_different_seed_gives_a_different_shuffle(self, tiny):
        a = size_inputs_for(tiny["cache"], tiny["splits"], shuffle=True, seed=0)
        b = size_inputs_for(tiny["cache"], tiny["splits"], shuffle=True, seed=1)
        assert not np.array_equal(a, b)


class TestCacheSizeLogits:
    def test_writes_logits_for_both_splits_in_the_shared_format(self, tiny):
        """a6 가 쓰는 형식과 같아야 앙상블 도구가 그대로 읽는다."""
        path = cache_size_logits(tiny["ckpt"], tiny["cache"], tiny["splits"],
                                 tiny["out"], "t", shuffle_size=False, seed=0)
        d = np.load(path)
        assert set(d.files) >= {"val_logits", "val_y", "test_logits", "test_y"}
        assert d["val_logits"].shape == (70, tiny["n_cls"])
        assert d["test_logits"].shape == (80, tiny["n_cls"])
        assert np.isfinite(d["test_logits"]).all()

    def test_matches_direct_inference_with_the_same_model(self, tiny, monkeypatch):
        """캐시된 로짓이 모델을 직접 돌린 것과 같아야 한다. 이것이 정확성의 근거다.

        비교 대상을 같은 장치(CPU)에 고정한다. GPU 와 CPU 는 float32 에서
        1e-4 규모로 다른 값을 내므로, 장치를 섞으면 배치 처리의 정확성이 아니라
        커널 차이를 재게 된다. GPU 경로의 검증은 a16 main 의 --expect 가
        학습 때 보고된 macro-F1 과 대조하는 것으로 한다.
        """
        import torch
        from a10_size_feature import SizeAwareNet
        from a3_train_wm811k_cls import to_onehot

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        path = cache_size_logits(tiny["ckpt"], tiny["cache"], tiny["splits"],
                                 tiny["out"], "t2", shuffle_size=False, seed=0)
        cached = np.load(path)["test_logits"]

        m = SizeAwareNet(num_classes=tiny["n_cls"])
        m.load_state_dict(torch.load(tiny["ckpt"], map_location="cpu", weights_only=True))
        m.eval()
        te = np.load(tiny["splits"])["test"]
        X = np.load(tiny["cache"])["X"]
        s = size_inputs_for(tiny["cache"], tiny["splits"], shuffle=False, seed=0)
        with torch.no_grad():
            direct = m(to_onehot(X[te]),
                       torch.as_tensor(s[te], dtype=torch.float32).unsqueeze(1)).numpy()
        assert np.allclose(cached, direct, atol=1e-4)

    def test_reuses_an_existing_cache_instead_of_recomputing(self, tiny):
        p1 = cache_size_logits(tiny["ckpt"], tiny["cache"], tiny["splits"],
                               tiny["out"], "t3", shuffle_size=False, seed=0)
        before = Path(p1).stat().st_mtime_ns
        p2 = cache_size_logits(tiny["ckpt"], tiny["cache"], tiny["splits"],
                               tiny["out"], "t3", shuffle_size=False, seed=0)
        assert p1 == p2 and Path(p2).stat().st_mtime_ns == before
