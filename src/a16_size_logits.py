"""SizeAwareNet 체크포인트의 로짓을 앙상블용 형식으로 캐시한다.

E14 까지 시험한 다양성 축은 입력 표현 하나뿐이다. E11 이 남긴 SizeAwareNet 두 개는
백본은 같지만 헤드가 다른 모델이라 **구조 다양성의 첫 표본**이 된다.
E11 이 기각됐다는 것은 여기에 상관없다 — 단독 성능과 앙상블 기여는 별개임이
E14 에서 확인됐고, 실제로 가장 나쁜 모델(0.6991)이 resize96(0.7320)보다 더 기여했다.

a6.cache_logits 는 plain ResNet 을 만들므로 이 체크포인트를 못 읽는다.
크기 입력은 학습 때와 **똑같이** 재현해야 한다 — train split 통계로만 표준화하고,
대조군은 같은 seed 의 같은 순열로 셔플한다. 그러지 않으면 학습된 것과 다른 모델을 재게 된다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def size_inputs_for(cache: str, splits: str, shuffle: bool, seed: int) -> np.ndarray:
    """학습 때 모델이 본 크기 입력을 그대로 되살린다.

    a10 의 main 과 같은 순서로 계산한다. train 통계로 표준화한 뒤,
    셔플 대조군이면 default_rng(seed) 의 첫 permutation 을 적용한다.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from a10_size_feature import standardize_size

    d, sp = np.load(cache), np.load(splits)
    raw = d["die_size"].astype(np.float64)
    _, stats = standardize_size(raw[sp["train"]])       # train 통계만
    out, _ = standardize_size(raw, stats=stats)
    if shuffle:
        out = out[np.random.default_rng(seed).permutation(len(out))]
    return out


def cache_size_logits(ckpt: str, cache: str, splits: str, out_dir: str, tag: str,
                      shuffle_size: bool = False, seed: int = 0,
                      batch_size: int = 512) -> str:
    """val, test 로짓을 a6 와 같은 npz 형식으로 저장한다. 있으면 재사용한다."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _runtime import setup
    setup("logits")
    import torch
    from a10_size_feature import SizeAwareNet
    from a3_train_wm811k_cls import to_onehot

    path = Path(out_dir) / f"{tag}_logits.npz"
    if path.exists():
        return str(path)

    d, sp = np.load(cache), np.load(splits)
    n_cls = len(d["classes"])
    # npz 는 지연 로딩이라 배치 루프 안에서 읽으면 매번 통째로 압축 해제된다.
    X, y_all = d["X"], d["y"]
    size_all = size_inputs_for(cache, splits, shuffle_size, seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SizeAwareNet(num_classes=n_cls)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()

    store = {}
    with torch.no_grad():
        for split in ("val", "test"):
            idx = sp[split]
            z = np.empty((len(idx), n_cls), dtype=np.float32)
            for i in range(0, len(idx), batch_size):
                b = idx[i:i + batch_size]
                s = torch.as_tensor(size_all[b], dtype=torch.float32, device=device).unsqueeze(1)
                z[i:i + len(b)] = model(to_onehot(X[b]).to(device), s).cpu().numpy()
            store[f"{split}_logits"] = z
            store[f"{split}_y"] = y_all[idx].astype(np.int64)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **store)
    return str(path)


def main() -> None:
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from a4_eval_wm811k_cls import evaluate

    p = argparse.ArgumentParser(description="SizeAwareNet 로짓 캐시")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64pad.npz")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--out-dir", default="result/posthoc")
    p.add_argument("--shuffle-size", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--expect", type=float, default=None,
                   help="학습 때 보고된 test macro-F1. 주면 대조해서 재현을 검증한다.")
    a = p.parse_args()

    path = cache_size_logits(a.ckpt, a.cache, a.splits, a.out_dir, a.tag,
                             a.shuffle_size, a.seed)
    d = np.load(path)
    n_cls = d["test_logits"].shape[1]
    m = evaluate(d["test_y"], d["test_logits"].argmax(1), n_cls)
    print(f"  {a.tag}  test macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}")
    if a.expect is not None:
        gap = abs(m["macro_f1"] - a.expect)
        verdict = "일치" if gap < 1e-4 else f"불일치 (차이 {gap:.4f})"
        print(f"  학습 때 보고값 {a.expect:.4f} 과 {verdict}")
        if gap >= 1e-4:
            raise SystemExit("크기 입력 재현이 틀렸다. 학습 때와 다른 모델을 평가하고 있다.")


if __name__ == "__main__":
    main()
