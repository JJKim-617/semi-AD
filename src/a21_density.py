"""국소 불량 밀도 맵. **학습이 전혀 없다** — 순수 계산이다.

## 왜 이것을 입력 채널로 넣는가

모델의 입력은 지금까지 one-hot 3채널(다이없음/정상/불량)뿐이었다.
그러면 **"불량이 국소적으로 몰려 있는가" 를 conv 가 스스로 배워야 한다.**

옆 워크스트림(`Semi-AD-OOD`)이 이 신호 하나만으로 정상/결함을
**AUROC 0.9206 / AUPR 0.7143** 로 가른다는 것을 보였다. 학습 없이.
그리고 k=3 은 0.3845 로 전역 불량률과 다를 바 없고 **k=5, k=7 에서만 이득이 난다** —
창이 다이 몇 개를 덮느냐가 신호의 존재 여부를 정한다.

이유는 이 데이터셋의 성질에 있다. **정상 웨이퍼도 불량 다이가 중앙값 80개(약 10%) 있다.
결함은 개수가 아니라 배치로 정의된다.** Scratch 의 중앙 불량 비율은 0.0917 로
정상 0.0983 보다 오히려 낮아서, 개수로는 원리적으로 구분할 수 없다.

## 정규화는 창 넓이가 아니라 창 안 다이 개수로 한다

`num / den` 에서 분모가 kxk 가 아니라 **창 안에 실제로 존재하는 다이 수**다.
그래서 웨이퍼 가장자리에서 값이 희석되지 않고, 비율이므로 웨이퍼 크기에 자동으로 불변이다.
극좌표(E15b)가 웨이퍼 크기를 표현 자체에 새겨 넣어 못 본 크기에서 무너졌던 것과
정확히 반대의 성질이다.

## 등변성이 설계의 핵심이다

정사각 창 + 상수 0 패딩이라 이 연산은 rot90, 좌우반전, 평행이동과 **정확히 교환된다.**
따라서 (1) 증강 뒤에 계산해도 되고, (2) dihedral TTA 가 그대로 성립하며,
(3) 학습이 보는 것과 추론이 보는 것이 어긋날 여지가 없다.
`tests/test_a21_density.py` 가 이 성질을 직접 박아 둔다.

코드는 `Semi-AD-OOD/src/a24_ood_residual.local_fail_density_map` 과 같은 식이다
(그쪽 worktree 는 읽기만 했다).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def local_fail_density_map(x: np.ndarray, k: int = 7, chunk: int = 4096) -> np.ndarray:
    """(N,H,W) uint8 {0,1,2} -> (N,H,W) float64. 각 다이 위치의 kxk 창 안 불량 비율.

    다이가 없는 칸은 정확히 0 이다.
    """
    if k < 1 or k % 2 == 0:
        raise ValueError(f"창 크기는 홀수여야 한다(중심이 있어야 등변이다): {k}")
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3:
        raise ValueError(f"(N,H,W) 를 기대한다: {x.shape}")
    out = np.zeros(x.shape, np.float64)
    for a in range(0, len(x), chunk):
        xb = x[a:a + chunk]
        die = (xb > 0).astype(np.float64)
        fail = (xb == 2).astype(np.float64)
        num = ndimage.uniform_filter(fail, size=(1, k, k), mode="constant")
        den = ndimage.uniform_filter(die, size=(1, k, k), mode="constant")
        out[a:a + len(xb)] = np.where(xb > 0, num / np.maximum(den, 1e-12), 0.0)
    return out


def density_stack(x: np.ndarray, ks=(5, 7), chunk: int = 4096) -> np.ndarray:
    """(N,H,W) -> (N,len(ks),H,W) float32. 채널 순서는 `ks` 순서 그대로다."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[None]
    if not len(ks):
        raise ValueError("창 크기를 하나 이상 줘야 한다")
    return np.stack([local_fail_density_map(x, k=k, chunk=chunk) for k in ks],
                    axis=1).astype(np.float32)


def density_stack_torch(x, ks=(5, 7)):
    """위와 같은 값을 텐서로 계산한다. **정의는 scipy 쪽이고 이건 속도용 사본이다.**

    1 에폭 스모크에서 numpy 경로가 96s/epoch 였다(기준선 38s) — 밀도 계산이 학습보다
    비쌌다. `avg_pool2d(count_include_pad=True)` 는 창 합을 kxk 로 나누므로
    분자/분모에서 그 상수가 소거되고, 상수 0 패딩이라 `mode="constant"` 와 정확히 같다.

    두 구현이 같은 값을 낸다는 것을 `tests/test_a21_density.py` 가 직접 대조한다.
    """
    import torch
    import torch.nn.functional as F

    if x.ndim == 2:
        x = x[None]
    if not len(ks):
        raise ValueError("창 크기를 하나 이상 줘야 한다")
    die = (x > 0).float().unsqueeze(1)
    fail = (x == 2).float().unsqueeze(1)
    out = []
    for k in ks:
        if k < 1 or k % 2 == 0:
            raise ValueError(f"창 크기는 홀수여야 한다(중심이 있어야 등변이다): {k}")
        num = F.avg_pool2d(fail, k, stride=1, padding=k // 2, count_include_pad=True)
        den = F.avg_pool2d(die, k, stride=1, padding=k // 2, count_include_pad=True)
        out.append(torch.where(die > 0, num / den.clamp_min(1e-12),
                               torch.zeros_like(num)))
    return torch.cat(out, dim=1)
