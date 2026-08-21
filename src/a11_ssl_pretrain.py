"""미라벨 웨이퍼맵으로 masked die prediction 사전학습.

811,457장 중 라벨은 172,950장뿐이고 638,507장(78.7%)을 쓰지 않고 있다.
일부 die 를 가리고 문맥에서 그 상태를 맞히게 해 결함의 공간적 구조를 배우게 한다.

회전 예측을 쓰지 않은 이유는 웨이퍼맵에 회전 대칭인 클래스가 있기 때문이다
(Edge-Ring 은 90도 돌려도 Edge-Ring 이다). 대칭 표본에서는 정답이 유일하지 않아
학습 신호가 잡음이 된다. 우리가 dihedral 증강을 쓰는 것 자체가 그 불변성을 인정한다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

NUM_CATEGORIES = 3


def random_patch_mask(shape: tuple[int, int], patch: int, ratio: float,
                      rng: np.random.Generator) -> np.ndarray:
    """정사각 패치 단위 랜덤 마스크. True 가 가려진 위치다.

    픽셀 단위로 흩뿌리면 이웃 값을 그대로 베낄 수 있어 과제가 너무 쉬워진다.
    패치 단위여야 문맥에서 추론하게 된다.
    """
    h, w = shape
    if h % patch or w % patch:
        raise ValueError(f"patch {patch} 가 shape {shape} 를 나누어떨어지게 하지 않는다")
    gh, gw = h // patch, w // patch
    n = gh * gw
    k = int(round(n * ratio))
    flat = np.zeros(n, dtype=bool)
    if k:
        flat[rng.choice(n, size=k, replace=False)] = True
    return np.kron(flat.reshape(gh, gw), np.ones((patch, patch), dtype=bool))


def masked_ce_loss(logits: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """가려진 위치에서만 교차 엔트로피를 계산한다.

    가리지 않은 곳은 입력에 그대로 있으므로 맞히는 데 정보가 필요 없다.
    마스크가 비면 0 을 반환한다(nan 방지).
    """
    if not mask.any():
        return logits.sum() * 0.0
    ce = nn.functional.cross_entropy(logits, target, reduction="none")
    return ce[mask].mean()


class MaskedWaferNet(nn.Module):
    """분류기와 같은 인코더 + 가벼운 디코더.

    사전학습이 끝나면 `encoder` 의 state_dict 를 분류 모델에 그대로 싣는다.
    """

    def __init__(self, width: int = 64):
        super().__init__()
        enc = tvm.resnet18(weights=None)
        enc.conv1 = nn.Conv2d(NUM_CATEGORIES, 64, 3, stride=1, padding=1, bias=False)
        enc.maxpool = nn.Identity()
        enc.avgpool = nn.Identity()
        enc.fc = nn.Identity()
        self.encoder = enc
        # layer1~4 를 지나면 64x64 -> 8x8, 채널 512. 세 번 올려 원해상도로 되돌린다.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, width * 4, 2, stride=2), nn.BatchNorm2d(width * 4), nn.ReLU(),
            nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.ConvTranspose2d(width * 2, width, 2, stride=2), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, NUM_CATEGORIES, 1),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encoder
        x = e.relu(e.bn1(e.conv1(x)))
        x = e.maxpool(x)
        return e.layer4(e.layer3(e.layer2(e.layer1(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.features(x))


def main() -> None:
    import argparse
    import json
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _runtime import setup
    setup("ssl")
    from a3_train_wm811k_cls import cosine_lr, dihedral, to_onehot

    p = argparse.ArgumentParser(description="masked die prediction 사전학습")
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64pad_all.npz")
    p.add_argument("--out-dir", default="result/ssl")
    p.add_argument("--tag", required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--labeled-only", action="store_true",
                   help="대조군. 라벨된 표본만 사전학습에 쓴다. 이득이 데이터 양 때문인지 가른다.")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(a.cache)
    X, y = d["X"], d["y"]
    idx = np.flatnonzero(y >= 0) if a.labeled_only else np.arange(len(X))
    print(f"[data] 사전학습 표본 {len(idx):,} / 전체 {len(X):,}"
          f"  (labeled_only={a.labeled_only})")

    model = MaskedWaferNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    hist, t0 = [], time.time()
    for ep in range(1, a.epochs + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(ep - 1, a.epochs, a.lr, warmup=1)
        model.train()
        perm = rng.permutation(len(idx))
        tot, seen = 0.0, 0
        for i in range(0, len(idx), a.batch_size):
            b = idx[perm[i:i + a.batch_size]]
            xb = dihedral(X[b], int(rng.integers(4)), bool(rng.integers(2)))
            tgt = torch.as_tensor(xb.astype(np.int64), device=device)
            mask_np = random_patch_mask(xb.shape[1:], a.patch, a.mask_ratio, rng)
            mask = torch.as_tensor(mask_np, device=device).expand(len(b), -1, -1)

            inp = to_onehot(xb).to(device)
            inp[:, :, mask_np] = 0.0          # 가린 자리는 어떤 범주도 아님으로 둔다
            loss = masked_ce_loss(model(inp), tgt, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            tot += loss.item() * len(b); seen += len(b)
        hist.append({"epoch": ep, "loss": tot / seen})
        print(f"  ep{ep:>3d}  masked CE {tot/seen:.4f}")

    torch.save(model.encoder.state_dict(), out / f"{a.tag}_encoder.pt")
    (out / f"{a.tag}.json").write_text(json.dumps({
        "tag": a.tag, "n_pretrain": int(len(idx)), "labeled_only": a.labeled_only,
        "epochs": a.epochs, "patch": a.patch, "mask_ratio": a.mask_ratio,
        "seconds": round(time.time() - t0, 1), "history": hist,
    }, indent=2), encoding="utf-8")
    print(f"[done] {out / f'{a.tag}_encoder.pt'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
