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


def build_model(num_classes: int = 9, pretrained: bool = False) -> nn.Module:
    """저해상도용으로 stem 을 고친 ResNet-18.

    pretrained=True 면 layer1~4 는 ImageNet 가중치를 쓰고 stem 만 새로 초기화된다.
    stem 은 커널 형상이 달라 사전학습 가중치를 이어받을 수 없다.
    """
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.resnet18(weights=weights)
    m.conv1 = nn.Conv2d(NUM_CATEGORIES, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


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


def _batches(n: int, batch_size: int, shuffle: bool, rng=None):
    idx = np.arange(n)
    if shuffle:
        (rng or np.random.default_rng()).shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def train_one_epoch(model, X, y, optimizer, batch_size=256, device="cuda",
                    weight=None, augment=False, rng=None, criterion=None,
                    grad_clip=None) -> float:
    """한 에폭 학습하고 평균 손실을 반환한다.

    criterion 을 주면 그것을 쓰고, 없으면 weight 를 반영한 CrossEntropy 를 쓴다.
    grad_clip 을 주면 그 노름으로 gradient 를 자른다.
    """
    model.to(device).train()
    rng = rng or np.random.default_rng(0)
    crit = criterion if criterion is not None else nn.CrossEntropyLoss(
        weight=None if weight is None else weight.to(device))
    total, seen = 0.0, 0
    for b in _batches(len(X), batch_size, shuffle=True, rng=rng):
        xb = X[b]
        if augment:
            xb = dihedral(xb, k=int(rng.integers(4)), flip=bool(rng.integers(2)))
        inp = to_onehot(xb).to(device)
        tgt = torch.as_tensor(y[b], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = crit(model(inp), tgt)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * len(b)
        seen += len(b)
    return total / seen


@torch.no_grad()
def predict(model, X, batch_size=512, device="cuda") -> np.ndarray:
    """예측 라벨을 반환한다. 가중치를 바꾸지 않는다."""
    model.to(device).eval()
    out = np.empty(len(X), dtype=np.int64)
    for b in _batches(len(X), batch_size, shuffle=False):
        logits = model(to_onehot(X[b]).to(device))
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
    p.add_argument("--class-weight", action="store_true")
    p.add_argument("--loss", default="ce", choices=["ce", "focal"])
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--grad-clip", type=float, default=None)
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

    model = build_model(num_classes=len(classes), pretrained=a.pretrained)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    w = class_weights(y[tr], len(classes)) if a.class_weight else None
    criterion = FocalLoss(gamma=a.gamma, alpha=w) if a.loss == "focal" else None
    if criterion is not None:
        criterion = criterion.to(device)
        print(f"[loss] focal gamma={a.gamma} alpha={'class_weight' if w is not None else 'none'}")

    from a4_eval_wm811k_cls import evaluate

    hist, best, t0 = [], -1.0, time.time()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(1, a.epochs + 1):
        loss = train_one_epoch(model, X[tr], y[tr], opt, a.batch_size, device,
                               weight=w, augment=a.augment, rng=rng,
                               criterion=criterion, grad_clip=a.grad_clip)
        m = evaluate(y[va], predict(model, X[va], a.batch_size * 2, device), len(classes))
        hist.append({"epoch": ep, "loss": loss, "val_macro_f1": m["macro_f1"],
                     "val_accuracy": m["accuracy"]})
        flag = ""
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(model.state_dict(), out_dir / f"{a.tag}_best.pt")
            flag = " *"
        print(f"  ep{ep:>3d}  loss {loss:.4f}  val macro-F1 {m['macro_f1']:.4f}  "
              f"acc {m['accuracy']:.4f}{flag}")

    (out_dir / f"{a.tag}_train.json").write_text(json.dumps({
        "config": vars(a) | {"config": str(a.config)},
        "device": device, "best_val_macro_f1": best,
        "seconds": round(time.time() - t0, 1), "history": hist,
    }, indent=2, default=str), encoding="utf-8")
    print(f"[done] best val macro-F1 {best:.4f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
