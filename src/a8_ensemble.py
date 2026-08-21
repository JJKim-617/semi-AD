"""수렴된 모델들을 재학습 없이 합친다.

seed 만 다른 두 모델의 test macro-F1 이 0.7251 과 0.7340 으로 갈리고, 클래스별로는
Edge-Loc 이 ±0.049, Scratch 가 ±0.029 흔들린다. 서로 다른 실수를 한다는 뜻이므로
합치면 분산이 줄어든다. logit 은 이미 캐시돼 있어 추가 학습이 필요 없다.

확률 평균을 기본으로 한다. logit 평균은 한 모델이 극단적으로 확신할 때 그쪽으로
결과가 끌려간다.
"""

from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def combine(logits_list: list[np.ndarray], mode: str = "prob",
            weights: list[float] | None = None) -> np.ndarray:
    """여러 모델의 logit 을 합친다.

    Args:
        logits_list: 모델별 (N, C) logit.
        mode: "prob" 이면 softmax 후 평균(기본), "logit" 이면 logit 자체를 평균.
        weights: 모델별 가중치. 없으면 균등.

    Returns:
        mode="prob" 이면 확률 (행 합 1), "logit" 이면 평균 logit.
    """
    if not logits_list:
        raise ValueError("최소 한 개의 logit 배열이 필요하다")
    arrs = [np.asarray(z, dtype=np.float64) for z in logits_list]
    shape = arrs[0].shape
    for i, z in enumerate(arrs):
        if z.shape != shape:
            raise ValueError(f"{i}번째 배열 shape {z.shape} 가 {shape} 와 다르다")
    if weights is None:
        w = np.ones(len(arrs))
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != len(arrs):
            raise ValueError(f"가중치 {w.shape[0]}개 와 모델 {len(arrs)}개 가 맞지 않는다")
    w = w / w.sum()

    if mode == "prob":
        return sum(wi * _softmax(z) for wi, z in zip(w, arrs))
    if mode == "logit":
        return sum(wi * z for wi, z in zip(w, arrs))
    raise ValueError(f"mode 는 'prob' 또는 'logit' 이어야 한다: {mode}")


def main() -> None:
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from a4_eval_wm811k_cls import evaluate
    from a6_perclass_offset import apply_offsets, cache_logits, optimize_offsets
    from a7_reweighted_offset import estimate_target_prior_cc, importance_weights

    p = argparse.ArgumentParser(description="앙상블 + 재가중 오프셋")
    p.add_argument("--tags", nargs="+", required=True,
                   help="result/cls_baseline/<tag>_best.pt 의 tag 들")
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64.npz")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--out-dir", default="result/posthoc")
    p.add_argument("--mode", default="prob", choices=["prob", "logit"])
    p.add_argument("--rounds", type=int, default=4)
    a = p.parse_args()

    classes = [str(c) for c in np.load(a.cache)["classes"]]
    n = len(classes)

    val_list, test_list = [], []
    for t in a.tags:
        L = np.load(cache_logits(f"result/cls_baseline/{t}_best.pt", a.cache,
                                 a.splits, a.out_dir, t))
        val_list.append(L["val_logits"])
        test_list.append(L["test_logits"])
        y_val, y_test = L["val_y"], L["test_y"]
        print(f"  [{t}] 단독 test macro-F1 "
              f"{evaluate(y_test, L['test_logits'].argmax(1), n)['macro_f1']:.4f}")

    ens_val = combine(val_list, mode=a.mode)
    ens_test = combine(test_list, mode=a.mode)
    # 확률을 log 로 옮겨 오프셋을 더하기 좋게 만든다.
    zv = np.log(np.clip(ens_val, 1e-12, None)) if a.mode == "prob" else ens_val
    zt = np.log(np.clip(ens_test, 1e-12, None)) if a.mode == "prob" else ens_test

    ens = evaluate(y_test, zt.argmax(1), n)
    q = estimate_target_prior_cc(zt, n)
    w = importance_weights(y_val, q, n)
    off = optimize_offsets(zv, y_val, n, rounds=a.rounds, sample_weight=w)
    final = evaluate(y_test, apply_offsets(zt, off).argmax(1), n)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    name = "ens_" + "_".join(a.tags)
    (out / f"{name}.json").write_text(json.dumps({
        "tags": a.tags, "mode": a.mode, "classes": classes,
        "offsets": off.tolist(), "target_prior_cc": q.tolist(),
        "test_macro_f1_ensemble": ens["macro_f1"],
        "test_accuracy_ensemble": ens["accuracy"],
        "test_macro_f1_ensemble_adjusted": final["macro_f1"],
        "test_accuracy_ensemble_adjusted": final["accuracy"],
        "test_per_class_f1_adjusted": final["per_class_f1"],
    }, indent=2), encoding="utf-8")

    print(f"  앙상블({a.mode})        macro-F1 {ens['macro_f1']:.4f}  acc {ens['accuracy']:.4f}")
    print(f"  + 재가중 오프셋      macro-F1 {final['macro_f1']:.4f}  acc {final['accuracy']:.4f}")


if __name__ == "__main__":
    main()
