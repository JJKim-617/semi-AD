"""WM-811K 9-class 분류 학습.

설계 근거는 `docs/research/encoder_selection/README.md` 에 있다. 요약하면,

- 백본은 **ResNet-18**. 이 데이터셋에서 백본 선택은 성능을 거의 가르지 않는다
  (동일 조건 7개 백본 비교에서 정확도 0.979~0.980). 표준적이고 수정이 쉬운 쪽을 택한다.
- **stem 을 3x3 stride 1 로 바꾸고 maxpool 을 제거**한다. 표준 stem 을 그대로 두면
  64x64 입력이 layer4 를 지나며 feature map 2x2 로 붕괴한다. 원 ResNet 논문이 CIFAR 용으로
  정의한 구성이고 SimCLR 도 저해상도 실험에서 같은 수정을 한다.
- 입력은 **3채널 one-hot**(배경/정상/불량). 셀 값 {0,1,2} 는 밝기가 아니라 명목형이라
  단일 채널 정수로 넣으면 없는 순서 관계를 주입한다.
- ImageNet 정규화는 적용하지 않는다. 입력이 이미 자연영상 통계와 무관하다.
- 사전학습은 플래그로 두고 scratch 와 **둘 다 측정**한다. 17만 장 규모에서는 전이 이득이
  최종 성능에는 거의 남지 않는다는 근거가 있어 단정하지 않고 실측한다.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

NUM_CATEGORIES = 3  # 0=die 없음, 1=정상, 2=불량


def to_onehot(x: np.ndarray) -> torch.Tensor:
    """명목형 웨이퍼맵 (N,H,W) uint8 {0,1,2} -> (N,3,H,W) float one-hot."""
    if x.min() < 0 or x.max() >= NUM_CATEGORIES:
        raise ValueError(
            f"셀 값은 0..{NUM_CATEGORIES - 1} 이어야 한다. 실제 범위: {x.min()}..{x.max()}"
        )
    t = torch.from_numpy(np.ascontiguousarray(x)).long()
    return nn.functional.one_hot(t, NUM_CATEGORIES).permute(0, 3, 1, 2).float()


def encode(x: np.ndarray, density_ks=(), shuffle_seed: int | None = None,
           line_ls=()) -> torch.Tensor:
    """모델 입력 텐서를 만든다. one-hot 3채널 뒤에 국소 밀도 채널을 이어붙인다.

    `density_ks=()` 면 `to_onehot` 과 완전히 같다. **기본값을 바꾸면 기존 체크포인트
    41개를 못 읽는다.**

    `shuffle_seed` 는 E20 의 대조군이다 — 밀도 채널을 **다른 웨이퍼에서 가져온다.**
    채널 수, 파라미터 수, 채널별 통계가 전부 보존되고 one-hot 과의 대응만 깨지므로,
    이득이 정보 때문인지 용량 때문인지를 가른다(E11 의 크기 값 셔플과 같은 설계).
    """
    t = to_onehot(x)
    if not len(density_ks) and not len(line_ls):
        return t
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from a21_density import density_stack
    from a22_line_filter import line_density_map

    src = np.asarray(x)
    if shuffle_seed is not None:
        src = src[np.random.default_rng(shuffle_seed).permutation(len(src))]
    parts = [t]
    if len(density_ks):
        parts.append(torch.from_numpy(density_stack(src, tuple(density_ks))))
    for L in line_ls:
        m = line_density_map(src, length=int(L), n_orient=8, min_dies=int(L))
        parts.append(torch.from_numpy(m.astype(np.float32)).unsqueeze(1))
    return torch.cat(parts, dim=1)


def encode_device(x: np.ndarray, density_ks=(), device="cuda",
                  shuffle_seed: int | None = None, line_ls=()) -> torch.Tensor:
    """`encode` 와 같은 값을 목표 장치에서 바로 만든다. 학습, 추론 경로는 이쪽을 쓴다.

    밀도를 CPU numpy 로 계산하면 96s/epoch 로 학습(38s)보다 비싸다. 같은 정의를
    `avg_pool2d` 로 옮겨 GPU 에서 계산한다. 두 구현이 같다는 것은 시험으로 박아 뒀다.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from a21_density import density_stack_torch
    from a22_line_filter import line_density_stack_torch

    t = torch.from_numpy(np.ascontiguousarray(x)).to(device).long()
    oh = nn.functional.one_hot(t, NUM_CATEGORIES).permute(0, 3, 1, 2).float()
    if not len(density_ks) and not len(line_ls):
        return oh
    src = t
    if shuffle_seed is not None:
        perm = np.random.default_rng(shuffle_seed).permutation(len(t))
        src = t[torch.from_numpy(perm).to(device)]
    parts = [oh]
    if len(density_ks):
        parts.append(density_stack_torch(src, tuple(density_ks)))
    if len(line_ls):
        parts.append(line_density_stack_torch(src, tuple(int(L) for L in line_ls)))
    return torch.cat(parts, dim=1)


PADDING_MODES = ("zeros", "reflect", "replicate", "circular")


def _apply_padding_mode(model: nn.Module, mode: str) -> nn.Module:
    """패딩이 있는 모든 Conv2d 의 패딩 방식을 바꾼다 (E24).

    **기본값 `zeros` 를 바꾸면 기존 체크포인트 42개의 결과를 되읽을 수 없다.**
    그래서 플래그로만 연다. 가중치와 파라미터 수는 전혀 바뀌지 않는다 —
    개입이 패딩 하나여야 교란이 없다.

    한 층이라도 빠지면 개입이 새어 실험이 무의미해지므로 **모든 층을 훑는다.**
    """
    if mode not in PADDING_MODES:
        raise ValueError(f"모르는 패딩 방식: {mode}. {PADDING_MODES} 중 하나여야 한다")
    if mode == "zeros":
        return model
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and any(q > 0 for q in m.padding):
            m.padding_mode = mode
    return model


def build_model(num_classes: int = 9, pretrained: bool = False,
                backbone: str = "resnet18", in_channels: int = 3,
                padding_mode: str = "zeros") -> nn.Module:
    """저해상도용으로 stem 을 고친 분류기.

    기본은 ResNet-18 이다. 기존 체크포인트가 전부 그것이고 호출부 여섯 곳이
    인자 없이 부르므로 **기본값을 바꾸면 과거 결과를 되읽을 수 없다.**

    `in_channels` 는 E20 의 국소 밀도 채널용이다. stem 의 입력 채널만 바뀌고
    나머지 구조는 그대로다. a19 백본들은 3채널 전제로 stem 이 수정돼 있으므로
    **조용히 어긋나느니 거절한다.**

    pretrained=True 면 layer1~4 는 ImageNet 가중치를 쓰고 stem 만 새로 초기화된다.
    stem 은 커널 형상이 달라 사전학습 가중치를 이어받을 수 없다.
    사전학습은 resnet18 에서만 지원한다 — 입력이 3채널 one-hot 명목형이라
    다른 백본까지 열어둘 근거가 없다.

    다른 백본은 `a19_backbones` 가 만든다. 전부 stem 을 고쳐 64x64 가
    붕괴하지 않게 돼 있다(최종 feature map 4x4~8x8).
    """
    if backbone != "resnet18":
        if pretrained:
            raise ValueError(f"{backbone} 는 사전학습을 지원하지 않는다")
        if in_channels != NUM_CATEGORIES:
            raise NotImplementedError(
                f"{backbone} 는 {NUM_CATEGORIES} 채널 입력만 지원한다 (요청 {in_channels})")
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from a19_backbones import build_backbone
        return _apply_padding_mode(
            build_backbone(backbone, num_classes=num_classes), padding_mode)

    weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.resnet18(weights=weights)
    m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return _apply_padding_mode(m, padding_mode)


class FocalLoss(nn.Module):
    """Focal loss. 잘 맞춘 샘플의 기여를 (1-p)^gamma 로 줄인다.

    역빈도 class weight 는 클래스 전체에 고정 배율을 걸어 극소수 클래스(Near-full 149장)에
    큰 가중치가 항상 붙고, 그 결과 gradient 분산이 커져 학습이 불안정해졌다(E4 는 ep4 에서
    accuracy 0.180 까지 붕괴). focal 은 배율이 샘플별이고 (1-p)^gamma 로 유계라 같은
    불균형 문제를 다루면서 그 부작용이 작다. 이 데이터셋은 93% 가 none 이라 쉬운 샘플이
    대부분이므로 눌러야 할 대상이 명확하다.

    alpha 를 주면 클래스별 가중치를 추가로 건다(focal 논문의 alpha-balanced 변형).
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma 는 0 이상이어야 한다: {gamma}")
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha if alpha is None else alpha.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = nn.functional.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - logp_t.exp()) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[target]
        return loss.mean()


def cosine_lr(epoch: int, total: int, base: float, warmup: int = 0) -> float:
    """warmup 후 코사인 감쇠. epoch 은 0부터 total 까지.

    15 epoch 고정 lr 에서 loss 가 계속 내려가는 중이었다(미수렴). 길게 학습하되
    후반에 lr 을 낮춰 진동을 줄인다. warmup 은 초기 gradient 스파이크 구간을 완만하게 만든다.
    """
    if total <= 0:
        raise ValueError(f"total 은 1 이상이어야 한다: {total}")
    if warmup and epoch < warmup:
        return base * (epoch + 1) / (warmup + 1)
    e = epoch - warmup
    t = max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(e, t) / t))


def should_snapshot(epoch: int, total: int, every: int) -> bool:
    """이 에폭의 체크포인트를 남길지 정한다.

    val 로는 중단 시점을 고를 수 없다는 것이 확인됐다(80 epoch 가 val 은 올랐는데
    test 는 0.018 떨어졌다). 어느 에폭이 실제로 좋았는지 사후에 확인하려면
    에폭별 체크포인트가 필요하다. every=0 이면 저장하지 않는다.
    """
    if every < 0:
        raise ValueError(f"every 는 0 이상이어야 한다: {every}")
    if every == 0:
        return False
    return epoch % every == 0 or epoch == total


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """역빈도 가중치. 등장하지 않는 클래스는 가중치 0 으로 두어 inf 를 피한다."""
    counts = np.bincount(np.asarray(y).ravel(), minlength=num_classes).astype(np.float64)
    w = np.zeros(num_classes, dtype=np.float64)
    seen = counts > 0
    w[seen] = counts[seen].sum() / (seen.sum() * counts[seen])
    return torch.tensor(w, dtype=torch.float32)


def dihedral(x: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """90도 배수 회전과 좌우 반전. 웨이퍼는 회전 대칭성이 있어 안전한 증강이다.

    격자 인덱스만 재배열하므로 명목형 셀 값이 보존된다. 임의 각도 회전은
    보간이 개입해 존재하지 않는 값을 만들므로 쓰지 않는다.
    """
    out = np.rot90(x, k=k, axes=(-2, -1))
    if flip:
        out = np.flip(out, axis=-1)
    return np.ascontiguousarray(out)


def angular_augment(x: np.ndarray, shift: int, flip: bool) -> np.ndarray:
    """극좌표용 증강. 행은 반지름, 열은 각도다.

    극좌표에서 유효한 대칭은 둘뿐이다. 웨이퍼의 회전은 **각도 축의 순환이동**이고
    반사는 **각도 축의 뒤집기**다. 반지름 축은 절대 뒤집지 않는다 — 중심과 가장자리가
    뒤바뀐다. dihedral 의 rot90 은 두 축을 맞바꾸므로 여기서 쓰면 안 된다.
    """
    out = np.roll(x, shift, axis=-1)
    if flip:
        out = np.flip(out, axis=-1)
    return np.ascontiguousarray(out)


def augment_batch(x: np.ndarray, mode: str, rng, extra=()) -> np.ndarray:
    """입력 표현에 맞는 증강을 고르고, 추가 증강이 있으면 이어서 적용한다.

    `dihedral` 은 카르테시안(pad, resize)용, `angular` 는 극좌표용이다.
    표현과 증강이 어긋나면 학습이 망가진다 — E15 에서 극좌표에 dihedral 을 걸었더니
    test macro-F1 이 0.5206 까지 떨어졌다.

    `extra` 는 a18 의 레시피 이름 목록이다(scale, translate, noise, dropout).
    scale 과 translate 는 캔버스에 여백이 있어야 의미가 있으므로 **pad 표현 전용**이다.
    """
    if mode == "dihedral":
        out = dihedral(x, k=int(rng.integers(4)), flip=bool(rng.integers(2)))
    elif mode == "angular":
        out = angular_augment(x, shift=int(rng.integers(x.shape[-1])),
                              flip=bool(rng.integers(2)))
    elif mode == "none":
        out = x
    else:
        raise ValueError(f"모르는 증강 방식: {mode}")

    if extra:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from a18_augment import apply_recipe
        out = apply_recipe(out, extra, rng)
    return out


def _batches(n: int, batch_size: int, shuffle: bool, rng=None):
    idx = np.arange(n)
    if shuffle:
        (rng or np.random.default_rng()).shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def train_one_epoch(model, X, y, optimizer, batch_size=256, device="cuda",
                    weight=None, augment=False, extra_augment=(), rng=None, criterion=None,
                    grad_clip=None, ema=None, density_ks=(), density_shuffle_seed=None,
                    line_ls=(), unlabeled=None, mu: int = 3, tau: float = 0.95,
                    lambda_u: float = 1.0) -> float:
    """한 에폭 학습하고 평균 손실을 반환한다.

    criterion 을 주면 그것을 쓰고, 없으면 weight 를 반영한 CrossEntropy 를 쓴다.
    grad_clip 을 주면 그 노름으로 gradient 를 자른다.

    `unlabeled` 를 주면 **E25 준지도(FixMatch 식 일관성)** 항을 더한다.
    약한 판(dihedral)의 예측이 `tau` 를 넘으면 그것을 의사 라벨로 삼아
    강한 판(dihedral + translate)에 교차엔트로피를 건다.
    라벨 1개당 미라벨 `mu` 개를 본다.

    **`unlabeled=None` 이면 이 함수는 예전과 완전히 같다.** 기존 체크포인트 42개의
    재현이 걸려 있으므로 시험으로 박아 뒀다(`tests/test_a3_semisup_wiring.py`).

    반환하는 평균 손실은 **라벨 손실만** 센다 — 일관성 항은 규모가 다르고
    에폭마다 통과 표본 수가 달라져서, 섞으면 학습 추이를 읽을 수 없다.
    """
    model.to(device).train()
    rng = rng or np.random.default_rng(0)
    crit = criterion if criterion is not None else nn.CrossEntropyLoss(
        weight=None if weight is None else weight.to(device))
    total, seen = 0.0, 0
    for b in _batches(len(X), batch_size, shuffle=True, rng=rng):
        xb = X[b]
        if augment:
            xb = augment_batch(xb, "dihedral" if augment is True else augment, rng,
                               extra=extra_augment)
        # 밀도는 **증강 뒤에** 계산한다. dihedral/translate 와 교환되므로 값은 같지만,
        # 증강 뒤 계산이 정의상 항상 옳다.
        inp = encode_device(xb, density_ks, device, density_shuffle_seed, line_ls)
        tgt = torch.as_tensor(y[b], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = crit(model(inp), tgt)
        loss_l = loss
        if unlabeled is not None and len(unlabeled) and mu > 0:
            import sys as _sys
            from pathlib import Path as _Path
            _sys.path.insert(0, str(_Path(__file__).resolve().parent))
            from a26_semisup import consistency_loss, pseudo_labels

            k = min(len(b) * mu, len(unlabeled))
            ui = np.sort(rng.choice(len(unlabeled), k, replace=False))
            xu = unlabeled[ui]
            # 약한 판 = dihedral 만. 강한 판 = 그 위에 translate.
            # **이 데이터에서 라벨을 보존하는 변환이 그 둘뿐**이라 강한 판이
            # 약한 판보다 아주 조금만 강하다 — 사전 등록에 적어 둔 구조적 약점이다.
            xw = augment_batch(xu, "dihedral", rng, extra=())
            xs = augment_batch(xw, "none", rng, extra=("translate",))
            iw = encode_device(xw, density_ks, device, density_shuffle_seed, line_ls)
            with torch.no_grad():
                pw = torch.softmax(model(iw).float(), 1)
            lab, mask = pseudo_labels(pw, tau)
            if bool(mask.any()):
                is_ = encode_device(xs, density_ks, device, density_shuffle_seed, line_ls)
                loss = loss + lambda_u * consistency_loss(model(is_), lab, mask)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total += loss_l.item() * len(b)      # 라벨 손실만 센다 (위 docstring)
        seen += len(b)
    return total / seen


@torch.no_grad()
def predict(model, X, batch_size=512, device="cuda",
            density_ks=(), density_shuffle_seed=None, line_ls=()) -> np.ndarray:
    """예측 라벨을 반환한다. 가중치를 바꾸지 않는다."""
    model.to(device).eval()
    out = np.empty(len(X), dtype=np.int64)
    for b in _batches(len(X), batch_size, shuffle=False):
        logits = model(encode_device(X[b], density_ks, device, density_shuffle_seed,
                                     line_ls))
        out[b] = logits.argmax(dim=1).cpu().numpy()
    return out


def main() -> None:
    import argparse
    import json
    import time
    from pathlib import Path

    import yaml

    p = argparse.ArgumentParser(description="WM-811K 9-class 분류 학습")
    p.add_argument("--config", type=Path)
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64.npz")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--out-dir", default="result/cls_baseline")
    p.add_argument("--tag", default="e1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--backbone", default="resnet18",
                   choices=["resnet18", "shufflenet_v2", "mobilenet_v3",
                            "convnext_tiny", "efficientnet_b0"],
                   help="백본 계열. 전부 64x64 용으로 stem 이 수정돼 있다.")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="가중치 EMA 감쇠. 0 이면 끈다. 켜면 EMA 체크포인트를 따로 저장한다.")
    p.add_argument("--extra-augment", nargs="*", default=[],
                   choices=["scale", "translate", "noise", "dropout", "rotate"],
                   help="a18 추가 증강. scale/translate 는 pad 표현 전용. "
                        "rotate 는 E23 의 자유 각도 회전(dihedral 의 연속 확장).")
    p.add_argument("--augment-mode", choices=["dihedral", "angular"], default="dihedral",
                   help="dihedral=카르테시안(pad, resize)용, angular=극좌표용. "
                        "표현과 어긋나면 학습이 망가진다.")
    p.add_argument("--density-ks", nargs="*", type=int, default=[],
                   help="E20 국소 밀도 입력 채널의 창 크기(홀수). 예: --density-ks 5 7. "
                        "비우면 기존과 동일한 one-hot 3채널이다.")
    p.add_argument("--line-ls", nargs="*", type=int, default=[],
                   help="E21 방향성 선 필터 채널의 길이(홀수). 예: --line-ls 11. "
                        "길이마다 min_dies=length 로 온전한 창만 본다.")
    p.add_argument("--density-shuffle", type=int, default=None,
                   help="대조군. 밀도 채널을 다른 웨이퍼에서 가져온다(순열 seed). "
                        "이득이 정보 때문인지 용량 때문인지를 가른다.")
    p.add_argument("--class-weight", action="store_true")
    p.add_argument("--loss", default="ce", choices=["ce", "focal"])
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--cosine", action="store_true")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--init-encoder", default=None,
                   help="자기지도 사전학습 인코더 가중치. fc 는 제외하고 싣는다.")
    p.add_argument("--snapshot-every", type=int, default=0,
                   help="N 에폭마다 체크포인트를 남긴다. 0 이면 저장하지 않는다.")
    p.add_argument("--unlabeled-cache", default=None,
                   help="E25 준지도. 미라벨이 섞인 캐시(예: wm811k_64pad_all.npz). "
                        "y < 0 인 행만 미라벨로 쓴다. 안 주면 기존과 완전히 동일하다.")
    p.add_argument("--mu", type=int, default=3,
                   help="E25. 라벨 1개당 미라벨 개수. batch 128 에서 mu=3 이 "
                        "24GB 상한이다(mu=7, batch 256 은 OOM).")
    p.add_argument("--tau", type=float, default=0.95,
                   help="E25. 의사 라벨 신뢰 임계값. 문헌 관례값 하나로 고정한다 — "
                        "여러 값을 시험해 고르는 것은 하이퍼파라미터 조정이다.")
    p.add_argument("--lambda-u", type=float, default=1.0,
                   help="E25. 일관성 항의 가중치.")
    p.add_argument("--padding-mode", default="zeros", choices=list(PADDING_MODES),
                   help="E24. 합성곱 패딩 방식. 기본 zeros 를 바꾸면 과거 체크포인트를 "
                        "못 읽으므로 플래그로만 연다.")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if a.config:
        cfg = yaml.safe_load(a.config.read_text()) or {}
        defaults = p.parse_args([])
        for k, v in cfg.items():
            k = k.replace("-", "_")
            if hasattr(a, k) and getattr(a, k) == getattr(defaults, k):
                setattr(a, k, v)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(a.cache)
    sp = np.load(a.splits)
    X, y, classes = d["X"], d["y"].astype(np.int64), [str(c) for c in d["classes"]]
    tr, va = sp["train"], sp["val"]
    print(f"[data] train {len(tr):,} | val {len(va):,} | classes {len(classes)} | {device}")

    dks = tuple(a.density_ks)
    lls = tuple(a.line_ls)
    in_ch = NUM_CATEGORIES + len(dks) + len(lls)
    if dks or lls:
        print(f"[channel] 밀도창 {list(dks)} 선길이 {list(lls)} -> 입력 {in_ch}채널"
              + (f"  (셔플 대조군 seed={a.density_shuffle})"
                 if a.density_shuffle is not None else ""))
    model = build_model(num_classes=len(classes), pretrained=a.pretrained,
                        backbone=a.backbone, in_channels=in_ch,
                        padding_mode=a.padding_mode)
    if a.init_encoder:
        sd = torch.load(a.init_encoder, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.startswith("fc.")]
        if bad:
            raise RuntimeError(f"사전학습 인코더에 없는 키가 fc 외에 있다: {bad[:5]}")
        print(f"[init] {a.init_encoder} 적재 (fc 는 새로 초기화)")
    unlabeled = None
    if a.unlabeled_cache:
        du = np.load(a.unlabeled_cache)
        yu = du["y"].astype(np.int64)
        sel = np.where(yu < 0)[0]
        if not len(sel):
            raise ValueError(f"{a.unlabeled_cache} 에 미라벨(y<0)이 없다")
        unlabeled = du["X"][sel]
        if unlabeled.shape[1:] != X.shape[1:]:
            raise ValueError(
                f"미라벨 표현이 라벨과 다르다: {unlabeled.shape[1:]} 대 {X.shape[1:]}")
        print(f"[semisup] 미라벨 {len(unlabeled):,}장 (라벨 학습의 "
              f"{len(unlabeled)/len(tr):.1f}배)  mu={a.mu} tau={a.tau} "
              f"lambda_u={a.lambda_u}")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    w = class_weights(y[tr], len(classes)) if a.class_weight else None
    criterion = FocalLoss(gamma=a.gamma, alpha=w) if a.loss == "focal" else None
    if criterion is not None:
        criterion = criterion.to(device)
        print(f"[loss] focal gamma={a.gamma} alpha={'class_weight' if w is not None else 'none'}")

    from a4_eval_wm811k_cls import evaluate

    ema = ema_model = None
    best_ema = -1.0
    if a.ema_decay > 0:
        from a20_ema import EMA
        ema = EMA(model, decay=a.ema_decay)
        ema_model = build_model(num_classes=len(classes), backbone=a.backbone,
                                in_channels=in_ch).to(device)
        print(f'[ema] decay={a.ema_decay} — EMA 가중치를 따로 저장한다')

    hist, best, t0 = [], -1.0, time.time()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(1, a.epochs + 1):
        if a.cosine:
            lr_now = cosine_lr(ep - 1, a.epochs, a.lr, a.warmup)
            for g in opt.param_groups:
                g["lr"] = lr_now
        loss = train_one_epoch(model, X[tr], y[tr], opt, a.batch_size, device,
                               weight=w, augment=(a.augment_mode if a.augment else False),
                               extra_augment=tuple(a.extra_augment), rng=rng,
                               criterion=criterion, grad_clip=a.grad_clip, ema=ema,
                               density_ks=dks, density_shuffle_seed=a.density_shuffle,
                               line_ls=lls, unlabeled=unlabeled, mu=a.mu, tau=a.tau,
                               lambda_u=a.lambda_u)
        m = evaluate(y[va], predict(model, X[va], a.batch_size * 2, device,
                                    density_ks=dks, line_ls=lls,
                                    density_shuffle_seed=a.density_shuffle), len(classes))
        hist.append({"epoch": ep, "loss": loss, "val_macro_f1": m["macro_f1"],
                     "val_accuracy": m["accuracy"]})
        if should_snapshot(ep, a.epochs, a.snapshot_every):
            snap = out_dir / "snapshots" / a.tag
            snap.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), snap / f"ep{ep:03d}.pt")
        if ema is not None:
            ema.copy_to(ema_model)
            me = evaluate(y[va], predict(ema_model, X[va], a.batch_size * 2, device,
                                         density_ks=dks, line_ls=lls,
                                         density_shuffle_seed=a.density_shuffle),
                          len(classes))
            hist[-1]["val_macro_f1_ema"] = me["macro_f1"]
            if me["macro_f1"] > best_ema:
                best_ema = me["macro_f1"]
                torch.save(ema_model.state_dict(), out_dir / f"{a.tag}_ema.pt")
        flag = ""
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(model.state_dict(), out_dir / f"{a.tag}_best.pt")
            flag = " *"
        extra = f"  ema {hist[-1]['val_macro_f1_ema']:.4f}" if ema is not None else ""
        print(f"  ep{ep:>3d}  loss {loss:.4f}  val macro-F1 {m['macro_f1']:.4f}  "
              f"acc {m['accuracy']:.4f}{extra}{flag}")

    (out_dir / f"{a.tag}_train.json").write_text(json.dumps({
        "config": vars(a) | {"config": str(a.config)},
        "device": device, "best_val_macro_f1": best,
        "backbone": a.backbone, "ema_decay": a.ema_decay,
        "best_val_macro_f1_ema": (best_ema if ema is not None else None),
        "seconds": round(time.time() - t0, 1), "history": hist,
    }, indent=2, default=str), encoding="utf-8")
    print(f"[done] best val macro-F1 {best:.4f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
