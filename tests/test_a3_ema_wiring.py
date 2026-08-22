"""EMA 는 **스텝마다** 갱신돼야 한다. 에폭마다면 평균할 표본이 40개뿐이다.

학습 궤적을 평균하는 것이 목적인데, 갱신 빈도가 낮으면 평균이라 부를 표본이 안 모인다.
43,223장 / batch 256 이면 에폭당 169 스텝이고 40 에폭이면 6,760 스텝이다.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a3_train_wm811k_cls import build_model, train_one_epoch


class CountingEMA:
    def __init__(self):
        self.calls = 0

    def update(self, model):
        self.calls += 1


class TestEmaWiring:
    def _run(self, ema, n=64, batch_size=16):
        rng = np.random.default_rng(0)
        X = rng.integers(0, 3, (n, 16, 16), dtype=np.uint8)
        y = (np.arange(n) % 3).astype(np.int64)
        m = build_model(num_classes=3)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        train_one_epoch(m, X, y, opt, batch_size=batch_size, device="cpu",
                        rng=np.random.default_rng(0), ema=ema)

    def test_updates_once_per_batch(self):
        ema = CountingEMA()
        self._run(ema, n=64, batch_size=16)
        assert ema.calls == 4, f"배치 4개인데 {ema.calls}번 갱신했다"

    def test_more_batches_means_more_updates(self):
        a, b = CountingEMA(), CountingEMA()
        self._run(a, n=64, batch_size=16)
        self._run(b, n=64, batch_size=8)
        assert b.calls > a.calls

    def test_training_still_works_without_an_ema(self):
        """기본 경로가 깨지면 안 된다. 기존 실행 전부가 ema 없이 돈다."""
        self._run(None)
