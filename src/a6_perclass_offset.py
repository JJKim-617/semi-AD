"""클래스별 logit 오프셋을 좌표 상승법으로 최적화한다.

a5 의 스칼라 tau 는 오프셋을 log(prior) 방향 한 축으로만 움직여 이득이 +0.0007 에 그쳤다.
macro-F1 은 표본별로 분해되지 않는 지표라 최적 결정규칙이 단순 argmax 가 아니고,
클래스마다 독립적인 임계값이 필요하다. 여기서는 9차원 오프셋 벡터를 직접 찾는다.

주의. 검증셋에서 9개 자유 파라미터를 맞추므로 과적합 위험이 있고, 우리 val 은 train 분포
(none 67.4%)를 물려받는데 test 는 93.3% 라 전이가 보장되지 않는다. val 로 고른 오프셋의
test 성능과, test 로 직접 고른 oracle 을 함께 보고해 그 격차를 드러낸다.
"""

from __future__ import annotations

import numpy as np

from a4_eval_wm811k_cls import evaluate


def apply_offsets(logits: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """클래스별 오프셋을 logit 에 더한다."""
    logits = np.asarray(logits, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.shape[0] != logits.shape[1]:
        raise ValueError(f"오프셋 길이 {offsets.shape[0]} 와 클래스 수 {logits.shape[1]} 가 다르다")
    return logits + offsets[None, :]


def optimize_offsets(logits: np.ndarray, y: np.ndarray, num_classes: int,
                     rounds: int = 3, lo: float = -4.0, hi: float = 4.0,
                     steps: int = 33, metric: str = "macro_f1",
                     verbose: bool = False) -> np.ndarray:
    """좌표 상승법. 0 벡터에서 시작해 한 클래스씩 최적 오프셋으로 갱신한다.

    각 단계에서 지표가 떨어지지 않는 값만 채택하므로 결과가 기준선보다 나쁠 수 없고,
    라운드를 늘려도 단조 비감소다. 결정론적이다(무작위 요소 없음).
    """
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y).ravel()
    off = np.zeros(num_classes)
    grid = np.linspace(lo, hi, steps)

    def score(o):
        return evaluate(y, apply_offsets(logits, o).argmax(1), num_classes)[metric]

    best = score(off)
    for r in range(rounds):
        improved = False
        for c in range(num_classes):
            cand, cur = off.copy(), off[c]
            for g in grid:
                cand[c] = g
                s = score(cand)
                if s > best + 1e-12:
                    best, cur, improved = s, g, True
            off[c] = cur
        if verbose:
            print(f"    round {r + 1}: {metric} {best:.4f}")
        if not improved:
            break
    return off


def cache_logits(ckpt: str, cache: str, splits: str, out_dir: str, tag: str) -> str:
    """체크포인트의 val, test logit 을 한 번 계산해 저장한다. 이후 탐색은 이 파일만 쓴다."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _runtime import setup
    setup("logits")
    import torch
    from a3_train_wm811k_cls import build_model, to_onehot

    path = Path(out_dir) / f"{tag}_logits.npz"
    if path.exists():
        return str(path)
    d, sp = np.load(cache), np.load(splits)
    n_cls = len(d["classes"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(num_classes=n_cls, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()

    store = {}
    with torch.no_grad():
        for split in ("val", "test"):
            idx = sp[split]
            z = np.empty((len(idx), n_cls), dtype=np.float32)
            for i in range(0, len(idx), 512):
                b = idx[i:i + 512]
                z[i:i + len(b)] = model(to_onehot(d["X"][b]).to(device)).cpu().numpy()
            store[f"{split}_logits"] = z
            store[f"{split}_y"] = d["y"][idx].astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **store)
    return str(path)


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(description="클래스별 logit 오프셋 최적화")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64.npz")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--out-dir", default="result/posthoc")
    p.add_argument("--tag", default=None)
    p.add_argument("--rounds", type=int, default=4)
    a = p.parse_args()
    tag = a.tag or Path(a.ckpt).stem

    lp = cache_logits(a.ckpt, a.cache, a.splits, a.out_dir, tag)
    L = np.load(lp)
    classes = [str(c) for c in np.load(a.cache)["classes"]]
    n = len(classes)

    base_val = evaluate(L["val_y"], L["val_logits"].argmax(1), n)
    base_test = evaluate(L["test_y"], L["test_logits"].argmax(1), n)

    print("[val 에서 오프셋 탐색]")
    off_val = optimize_offsets(L["val_logits"], L["val_y"], n, rounds=a.rounds, verbose=True)
    sel = evaluate(L["test_y"], apply_offsets(L["test_logits"], off_val).argmax(1), n)

    print("[test 에서 직접 탐색 — oracle 상한]")
    off_te = optimize_offsets(L["test_logits"], L["test_y"], n, rounds=a.rounds, verbose=True)
    orc = evaluate(L["test_y"], apply_offsets(L["test_logits"], off_te).argmax(1), n)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}_offsets.json").write_text(json.dumps({
        "ckpt": a.ckpt, "classes": classes,
        "offsets_from_val": off_val.tolist(), "offsets_from_test_oracle": off_te.tolist(),
        "val_macro_f1_base": base_val["macro_f1"],
        "test_macro_f1_base": base_test["macro_f1"],
        "test_accuracy_base": base_test["accuracy"],
        "test_macro_f1_val_selected": sel["macro_f1"],
        "test_accuracy_val_selected": sel["accuracy"],
        "test_per_class_f1_val_selected": sel["per_class_f1"],
        "test_macro_f1_oracle": orc["macro_f1"],
    }, indent=2), encoding="utf-8")

    print()
    print(f"  기준선                macro-F1 {base_test['macro_f1']:.4f}  acc {base_test['accuracy']:.4f}")
    print(f"  val 선택 오프셋       macro-F1 {sel['macro_f1']:.4f}  acc {sel['accuracy']:.4f}"
          f"   ({sel['macro_f1'] - base_test['macro_f1']:+.4f})")
    print(f"  oracle (test 로 선택) macro-F1 {orc['macro_f1']:.4f}   (상한, 참고용)")
    print("  오프셋:", ", ".join(f"{c}={v:+.2f}" for c, v in zip(classes, off_val)))


if __name__ == "__main__":
    main()
