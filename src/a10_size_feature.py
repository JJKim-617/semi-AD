"""웨이퍼 크기를 보조 입력으로 쓰는 모델.

진단에서 클래스 조건부 분포가 split 간 다르다는 것이 확인됐다. 같은 Scratch 가
train 은 median 52x52, test 는 31x31 이다. pad 표현이 resize 를 이긴 것도(+0.0216)
크기 정보가 유용하다는 증거다. 다만 pad 는 크기를 암묵적으로 전달한다.
`dieSize` 는 캐시에 이미 있으므로 직접 줄 수 있다.

BN 통계 적응과 달리 이 정보는 **클래스 구성에 오염되지 않는다.** 웨이퍼 크기는
라벨 분포와 무관한 표본 고유 속성이다. BN 적응이 실패한 이유가 타겟 집합의
클래스 구성(none 93.3%)이 통계를 오염시킨 것이었으므로, 이 구분이 중요하다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

NUM_CATEGORIES = 3


def standardize_size(size: np.ndarray, stats: tuple[float, float] | None = None
                     ) -> tuple[np.ndarray, tuple[float, float]]:
    """크기를 표준화한다.

    stats 를 주면 그것을 그대로 적용한다(train 통계를 test 에 적용하는 경로).
    주지 않으면 입력에서 계산한다. **test 에서 통계를 다시 계산하면 누수다.**
    상수 입력이면 표준편차가 0 이므로 1 로 대체한다.
    """
    x = np.asarray(size, dtype=np.float64).ravel()
    if stats is None:
        mu, sd = float(x.mean()), float(x.std())
        if sd == 0.0:
            sd = 1.0
        stats = (mu, sd)
    mu, sd = stats
    return (x - mu) / sd, stats


class SizeAwareNet(nn.Module):
    """저해상도용 ResNet-18 에 크기 스칼라를 분류 헤드에서 결합한다.

    크기는 작은 MLP 를 거쳐 이미지 특징과 concat 된다. 스칼라 하나를 512차원 특징에
    그대로 붙이면 묻히므로 차원을 키워 준다.
    """

    def __init__(self, num_classes: int = 9, size_dim: int = 16):
        super().__init__()
        self.backbone = tvm.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(NUM_CATEGORIES, 64, 3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.size_mlp = nn.Sequential(
            nn.Linear(1, size_dim), nn.ReLU(), nn.Linear(size_dim, size_dim), nn.ReLU())
        self.head = nn.Linear(feat_dim + size_dim, num_classes)

    def forward(self, x: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        s = self.size_mlp(size)
        return self.head(torch.cat([f, s], dim=1))


def main() -> None:
    import argparse
    import json
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _runtime import setup
    setup("size")
    from a3_train_wm811k_cls import dihedral, to_onehot
    from a4_eval_wm811k_cls import evaluate

    p = argparse.ArgumentParser(description="크기 보조 입력 학습")
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64pad.npz")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--out-dir", default="result/cls_baseline")
    p.add_argument("--tag", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-size", action="store_true",
                   help="대조군. 크기 값을 무작위로 섞어 정보를 파괴한다. 용량은 동일하다.")
    a = p.parse_args()

    from a3_train_wm811k_cls import cosine_lr
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d, sp = np.load(a.cache), np.load(a.splits)
    classes = [str(c) for c in d["classes"]]
    n = len(classes)
    X, y = d["X"], d["y"].astype(np.int64)
    size_raw = d["die_size"].astype(np.float64)
    tr, va, te = sp["train"], sp["val"], sp["test"]

    _, stats = standardize_size(size_raw[tr])          # train 통계만 쓴다
    size_all, _ = standardize_size(size_raw, stats=stats)
    if a.shuffle_size:
        size_all = size_all[rng.permutation(len(size_all))]
        print("[대조군] 크기 값을 셔플했다. 정보는 없고 용량만 동일하다.")

    model = SizeAwareNet(num_classes=n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    crit = nn.CrossEntropyLoss()

    def run_eval(idx):
        model.eval()
        out = np.empty(len(idx), dtype=np.int64)
        with torch.no_grad():
            for i in range(0, len(idx), 512):
                b = idx[i:i + 512]
                s = torch.as_tensor(size_all[b], dtype=torch.float32,
                                    device=device).unsqueeze(1)
                out[i:i + len(b)] = model(to_onehot(X[b]).to(device), s).argmax(1).cpu().numpy()
        return evaluate(y[idx], out, n)

    hist, best, t0 = [], -1.0, time.time()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(1, a.epochs + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(ep - 1, a.epochs, a.lr, a.warmup)
        model.train()
        perm = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(tr), a.batch_size):
            b = tr[perm[i:i + a.batch_size]]
            xb = dihedral(X[b], int(rng.integers(4)), bool(rng.integers(2)))
            s = torch.as_tensor(size_all[b], dtype=torch.float32, device=device).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(to_onehot(xb).to(device), s),
                        torch.as_tensor(y[b], dtype=torch.long, device=device))
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        m = run_eval(va)
        hist.append({"epoch": ep, "loss": tot / len(tr), "val_macro_f1": m["macro_f1"]})
        flag = ""
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(model.state_dict(), out_dir / f"{a.tag}_best.pt")
            flag = " *"
        print(f"  ep{ep:>3d}  loss {tot/len(tr):.4f}  val macro-F1 {m['macro_f1']:.4f}{flag}")

    model.load_state_dict(torch.load(out_dir / f"{a.tag}_best.pt", weights_only=True))
    t = run_eval(te)
    (out_dir / f"{a.tag}_test.json").write_text(json.dumps({
        "tag": a.tag, "shuffle_size": a.shuffle_size, "classes": classes,
        "best_val_macro_f1": best, "seconds": round(time.time() - t0, 1), **t,
    }, indent=2), encoding="utf-8")
    print(f"[done] test macro-F1 {t['macro_f1']:.4f}  acc {t['accuracy']:.4f}")


if __name__ == "__main__":
    main()
