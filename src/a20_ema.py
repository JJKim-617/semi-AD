"""가중치의 지수이동평균(EMA).

이 프로젝트 최대 불안정 요인은 seed 분산이다 — pad 3 seed 범위 0.0495 이고
그 3/4 이 Scratch 붕괴 하나에서 나온다(F1 0.613 / **0.280** / 0.608).
자기지도 사전학습이 그 분산을 20배 줄였지만(범위 0.0025) 평균도 함께 떨어뜨려 채택하지 못했다.

EMA 는 학습 궤적 위의 여러 시점을 평균한다. 앙상블이 **여러 실행**을 평균하는 것이라면
EMA 는 **한 실행 안**에서 같은 일을 한다. 추가 forward 도 backward 도 없어 학습 비용이
사실상 0 이고, 이 프로젝트에서 통한 것이 전부 "평균내기" 계열이었다는 점과도 맞는다.

정확성의 핵심은 **정수 버퍼**다. BatchNorm 의 `num_batches_tracked` 는 int64 인데
거기에 소수 가중치를 곱하면 값이 망가진다. 실수 텐서만 평균하고 정수는 복사한다.
BN 의 running_mean, running_var 는 실수 버퍼이므로 평균 대상에 포함한다 —
빼면 EMA 가 반쪽이 되어 추론 시 통계가 가중치와 어긋난다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    """모델 가중치의 지수이동평균을 유지한다.

    shadow = decay * shadow + (1 - decay) * current
    decay 가 클수록 천천히 따라간다. 0 이면 현재 모델과 같고 1 이면 영원히 안 움직인다.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay 는 [0,1] 이어야 한다: {decay}")
        self.decay = float(decay)
        self.shadow = {}
        self._ints = {}
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = v.detach().clone().float()
            else:
                self._ints[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """현재 모델 쪽으로 한 걸음 옮긴다. 정수 버퍼는 그대로 따라간다.

        shadow 는 모델을 따라 장치를 옮긴다. a3 는 CPU 에서 모델을 만들고
        학습 루프 첫 스텝에서 `.to(device)` 하므로, 그 사이에 만들어진 EMA 는
        shadow 가 CPU 에 남는다. 첫 update 에서 맞춰준다.
        """
        sd = model.state_dict()
        d = self.decay
        for k, s in self.shadow.items():
            cur = sd[k].detach().float()
            if s.device != cur.device:
                s = s.to(cur.device)
                self.shadow[k] = s
            s.mul_(d).add_(cur, alpha=1.0 - d)
        for k in self._ints:
            self._ints[k] = sd[k].detach().clone()

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """평균낸 가중치를 모델에 싣는다. 원본 모델은 건드리지 않는다."""
        sd = model.state_dict()
        merged = {}
        for k, v in sd.items():
            if k in self.shadow:
                merged[k] = self.shadow[k].to(dtype=v.dtype, device=v.device)
            elif k in self._ints:
                merged[k] = self._ints[k].to(dtype=v.dtype, device=v.device)
            else:
                merged[k] = v
        model.load_state_dict(merged)

    def state_dict(self) -> dict:
        """체크포인트로 저장할 수 있는 형태."""
        return {**{k: v.clone() for k, v in self.shadow.items()},
                **{k: v.clone() for k, v in self._ints.items()}}
