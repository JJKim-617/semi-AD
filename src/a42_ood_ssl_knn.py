"""O2 — 자기지도 인코더 + kNN 패치 메모리 뱅크. **정상 웨이퍼만 쓴다.**

설계는 `docs/experiments/candidate/ood_o2_ssl_knn.md` §10 에 **한 줄도 쓰기 전에** 박았다.
여기 상수와 규칙은 그 문서를 그대로 옮긴 것이고, **결과를 보고 바꾸지 않는다.**

## 왜 수용영역이 7 인가

질문이 "학습된 표현이 창 통계를 넘는가" 이므로 **같은 창 안에서 겨루게** 해야 한다.
전역 수용영역(ResNet-18)으로 재면 이득이 학습에서 온 것인지 창을 키워서 온 것인지 못 가른다.
7 을 고른 근거는 실측이다 — 온전 창(창 안이 전부 다이)이 하나도 없는 train-none 웨이퍼가
11x11 에서 **7.07%**, 9x9 에서 1.24%, **7x7 에서 0.11%** 다.

**한계**: 요소 B(선 L=11)는 이 수용영역 밖이다. O2 가 fuse3 를 못 넘어도
"길이 11 구조를 못 봐서" 라는 설명이 남는다 — 그 설명을 사후에 꺼내지 않으려고 미리 적는다.

## 왜 온전 창만인가

정정 11 이다. 창이 웨이퍼 경계에서 잘리면 분모가 무너지고, 그것을 안 보면
**틀린 분모 → 틀린 확률 → 틀린 기작 → 사실 아닌 "사실"** 로 이어진다.
여기서는 분모가 없지만 같은 규율을 지킨다 — **뱅크도 질의도 온전 창 위치만.**
"""

from __future__ import annotations

import numpy as np

RF = 7                       # 수용영역. 3x3 conv 3층.
FEAT_DIM = 32
BANK_SIZE = 8192
KNN_K = 5
MASK_PATCH = 4
MASK_RATIO = 0.4
EPOCHS = 12
BATCH = 256
LR = 1e-3
NUM_CATEGORIES = 3


# --- 온전 창 -------------------------------------------------------------------

def valid_window_mask(x, k: int = RF) -> np.ndarray:
    """kxk 창의 k*k 칸이 **전부 다이**인 중심 위치. 정수 누적합으로 정확히 센다.

    `uniform_filter` 계열을 쓰지 않는다 — 반올림 잡음이 동점을 쪼개고
    동점은 인덱스 순서로 갈리는데 그 순서에 라벨이 샌다(정정 8).
    """
    x = np.asarray(x)
    die = (x > 0).astype(np.int64)
    b, h, w = die.shape
    c = np.zeros((b, h + 1, w + 1), np.int64)
    c[:, 1:, 1:] = die.cumsum(1).cumsum(2)
    s = c[:, k:, k:] - c[:, :-k, k:] - c[:, k:, :-k] + c[:, :-k, :-k]
    out = np.zeros((b, h, w), bool)
    r = (k - 1) // 2
    if s.shape[1] > 0 and s.shape[2] > 0:
        out[:, r:r + s.shape[1], r:r + s.shape[2]] = s == k * k
    return out


def wafer_max_score(loc_scores, valid) -> float:
    """유효 위치 점수의 max. **유효 위치가 없으면 전체의 max 로 되돌린다.**

    조용히 빠뜨리면 그 웨이퍼가 집계에서 사라진다. train-none 의 0.11% 가 여기 걸린다.
    """
    loc_scores = np.asarray(loc_scores, np.float64).ravel()
    valid = np.asarray(valid, bool).ravel()
    if valid.any():
        return float(loc_scores[valid].max())
    return float(loc_scores.max())


# --- kNN ------------------------------------------------------------------------

def build_bank(feats, n: int = BANK_SIZE, seed: int = 0) -> np.ndarray:
    """train-none 특징에서 **무작위로** n 개. coreset 이 아니다.

    coreset 선택은 그 자체가 손잡이이고, 그 손잡이를 test 로 안 골랐다는 것을
    증명하기 번거롭다. **무작위는 시드 하나로 끝난다.**
    """
    feats = np.asarray(feats)
    if n >= len(feats):
        return feats.copy()
    rng = np.random.default_rng(seed)
    return feats[rng.choice(len(feats), n, replace=False)].copy()


def knn_mean_distance(queries, bank, k: int = KNN_K, chunk: int = 16384,
                      device: str = "cpu") -> np.ndarray:
    """뱅크까지 k 최근접 **코사인 거리의 평균**. 특징은 L2 정규화돼 있다고 본다.

    torch 로 계산한다. `numpy.partition` 은 단일 스레드라 5e7 개 질의에서 병목이 되는데
    `torch.topk` 는 CPU 에서도 여러 스레드를 쓴다. **값은 정의 그대로**이고
    무차별 대입과 같다는 것을 테스트로 박았다(`test_knn_mean_distance_matches_brute_force`).

    `device="cuda"` 로 바꾸면 그대로 GPU 에서 돈다. 다만 **GPU 0번을 9-class 학습 큐가
    쓰고 있으면 쓰지 않는다** — 남의 학습을 OOM 으로 죽이지 않기 위해서다.
    """
    import torch
    q = torch.as_tensor(np.asarray(queries, np.float32), device=device)
    b = torch.as_tensor(np.asarray(bank, np.float32), device=device)
    k = min(k, b.shape[0])
    out = np.empty(len(q), np.float64)
    with torch.no_grad():
        for a in range(0, len(q), chunk):
            sim = q[a:a + chunk] @ b.T
            top = torch.topk(sim, k, dim=1, largest=True, sorted=False).values
            out[a:a + top.shape[0]] = (1.0 - top).mean(1).double().cpu().numpy()
    return out


# --- torch 부분 (import 는 지연시킨다. numpy 만 쓰는 테스트가 torch 없이도 돌게) ---

def to_onehot(x):
    """3 범주 one-hot. **웨이퍼 밖은 채널 0 이 1** 이다 — 범주가 셋이지 둘이 아니다."""
    import torch
    a = torch.as_tensor(np.asarray(x, np.int64))
    return torch.nn.functional.one_hot(a, NUM_CATEGORIES).permute(0, 3, 1, 2).float()


def _build_encoder():
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(NUM_CATEGORIES, 24, 3, padding=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(),
        nn.Conv2d(24, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, FEAT_DIM, 3, padding=1, bias=False), nn.BatchNorm2d(FEAT_DIM), nn.ReLU(),
    )


class _PatchEncoderImpl:
    pass


def _make_class():
    import torch
    import torch.nn as nn

    class PatchEncoder(nn.Module):
        """3x3 conv 3층. **수용영역 정확히 7x7.** 출력은 위치별 L2 정규화.

        정규화하는 이유: 코사인 거리를 내적으로 계산하기 위해서다.
        정규화하지 않으면 활성 크기가 큰 위치가 거리를 지배한다.
        """

        def __init__(self):
            super().__init__()
            self.body = _build_encoder()

        def forward(self, x):
            f = self.body(x)
            return f / f.pow(2).sum(1, keepdim=True).clamp_min(1e-12).sqrt()

    class MaskedHead(nn.Module):
        """인코더 + 1x1 분류 머리. 가린 자리의 범주를 문맥에서 맞힌다."""

        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Conv2d(FEAT_DIM, NUM_CATEGORIES, 1)

        def forward(self, x):
            return self.head(self.encoder(x))

    return PatchEncoder, MaskedHead


def __getattr__(name):
    if name in ("PatchEncoder", "MaskedHead"):
        pe, mh = _make_class()
        globals()["PatchEncoder"], globals()["MaskedHead"] = pe, mh
        return globals()[name]
    raise AttributeError(name)


def masked_valid_ce(logits, target, mask, valid):
    """**가려졌고 동시에 온전 창인 위치에서만** 교차 엔트로피.

    안 가린 곳은 입력에 그대로 있으므로 맞히는 데 정보가 필요 없다.
    온전 창 밖은 §10.2 가 점수를 안 낸다고 정한 곳이라 학습 신호에서도 뺀다.
    둘 다 아니면 0 을 준다(nan 방지).
    """
    import torch
    sel = mask & valid
    if not bool(sel.any()):
        return logits.sum() * 0.0
    ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    return ce[sel].mean()


def random_patch_mask(shape, patch: int, ratio: float, rng) -> np.ndarray:
    """정사각 패치 단위 랜덤 마스크. True 가 가려진 자리.

    픽셀 단위로 흩뿌리면 이웃에서 그대로 베낄 수 있어 과제가 너무 쉬워진다.
    """
    h, w = shape
    gh, gw = (h + patch - 1) // patch, (w + patch - 1) // patch
    n = gh * gw
    flat = np.zeros(n, bool)
    kk = int(round(n * ratio))
    if kk:
        flat[rng.choice(n, kk, replace=False)] = True
    return np.kron(flat.reshape(gh, gw), np.ones((patch, patch), bool))[:h, :w]


def dihedral(x, rot: int, flip: bool) -> np.ndarray:
    """90도 배수 회전 + 뒤집기만. **격자 정렬이 유지된다.**

    임의 각도 회전은 명목형 격자를 파괴한다(기획서 §7).
    """
    a = np.rot90(np.asarray(x), rot, axes=(1, 2))
    if flip:
        a = a[:, :, ::-1]
    return np.ascontiguousarray(a)
