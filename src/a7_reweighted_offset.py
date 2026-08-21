"""타겟 분포로 재가중한 뒤 클래스별 오프셋을 튜닝한다.

a6 에서 val 로 고른 오프셋이 test 를 악화시켰다(0.6705 -> 0.6554). 이론적 이유가 있다.
test 의 93.3% 가 none 이라 none 에서 오탐률이 조금만 생겨도 결함 클래스의 precision 이
붕괴한다. 따라서 test 에서 macro-F1 을 최대화하는 결정 규칙은 val 에서보다 결함 예측에
**더 보수적**이어야 하는데, val 에서 튜닝하면 정반대 방향으로 밀린다.

해법은 val 표본에 w_y = q_target(y) / p_val(y) 를 곱해 "타겟 분포에서의 macro-F1" 을
val 라벨만으로 근사하는 것이다. 타겟 사전확률은 모델이 라벨 없는 test 에 매긴 예측 빈도로
추정한다(classify-and-count). test 라벨은 쓰지 않는다.

한계. CC 추정은 분류기 편향을 그대로 물려받고, 이 데이터셋은 클래스 조건부 분포도
일부 변해서(같은 Scratch 가 train 은 median 52x52, test 는 31x31) 순수 label shift 가정이
깨져 있다. 그래서 이 경로는 만능이 아니라 "val 튜닝보다 나은가" 를 묻는 실험이다.
"""

from __future__ import annotations

import numpy as np


def estimate_target_prior_cc(logits: np.ndarray, num_classes: int,
                             floor: float = 0.5) -> np.ndarray:
    """classify-and-count. 라벨 없이 예측 빈도로 타겟 사전확률을 추정한다.

    라벨을 인자로 받지 않는다. 서명 자체가 test 라벨 미사용을 보장한다.
    """
    pred = np.asarray(logits).argmax(1)
    counts = np.bincount(pred, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = floor
    return counts / counts.sum()


def importance_weights(y: np.ndarray, target_prior: np.ndarray,
                       num_classes: int) -> np.ndarray:
    """표본별 가중치 w_i = q(y_i) / p_source(y_i).

    가중치를 적용한 뒤 클래스별 질량 비율이 target_prior 와 일치한다.
    """
    y = np.asarray(y).ravel()
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    source = np.where(counts > 0, counts, 1.0)
    source = source / source.sum()
    w = np.asarray(target_prior, dtype=np.float64) / source
    return w[y]


def main() -> None:
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from a4_eval_wm811k_cls import evaluate
    from a6_perclass_offset import apply_offsets, cache_logits, optimize_offsets

    p = argparse.ArgumentParser(description="타겟 재가중 오프셋 튜닝")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64.npz")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--out-dir", default="result/posthoc")
    p.add_argument("--tag", default=None)
    p.add_argument("--rounds", type=int, default=4)
    a = p.parse_args()
    tag = a.tag or Path(a.ckpt).stem

    L = np.load(cache_logits(a.ckpt, a.cache, a.splits, a.out_dir, tag))
    classes = [str(c) for c in np.load(a.cache)["classes"]]
    n = len(classes)

    # 타겟 사전확률을 라벨 없이 추정한다.
    q = estimate_target_prior_cc(L["test_logits"], n)
    w = importance_weights(L["val_y"], q, n)

    base = evaluate(L["test_y"], L["test_logits"].argmax(1), n)
    off_plain = optimize_offsets(L["val_logits"], L["val_y"], n, rounds=a.rounds)
    off_rw = optimize_offsets(L["val_logits"], L["val_y"], n, rounds=a.rounds, sample_weight=w)

    m_plain = evaluate(L["test_y"], apply_offsets(L["test_logits"], off_plain).argmax(1), n)
    m_rw = evaluate(L["test_y"], apply_offsets(L["test_logits"], off_rw).argmax(1), n)

    # 참고용: 실제 test 사전확률(라벨 사용). 추정이 얼마나 맞았는지 보기 위해서만 쓴다.
    true_q = np.bincount(L["test_y"], minlength=n) / len(L["test_y"])

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}_reweighted.json").write_text(json.dumps({
        "ckpt": a.ckpt, "classes": classes,
        "target_prior_estimated_cc": q.tolist(),
        "target_prior_true_for_reference": true_q.tolist(),
        "offsets_plain_val": off_plain.tolist(),
        "offsets_reweighted_val": off_rw.tolist(),
        "test_macro_f1_base": base["macro_f1"], "test_accuracy_base": base["accuracy"],
        "test_macro_f1_plain_val": m_plain["macro_f1"],
        "test_macro_f1_reweighted": m_rw["macro_f1"],
        "test_accuracy_reweighted": m_rw["accuracy"],
        "test_per_class_f1_reweighted": m_rw["per_class_f1"],
    }, indent=2), encoding="utf-8")

    print(f"  타겟 사전확률 추정(CC) none={q[0]:.3f}  실제 none={true_q[0]:.3f}"
          f"  (오차 {abs(q[0]-true_q[0]):.3f})")
    print(f"  기준선            macro-F1 {base['macro_f1']:.4f}  acc {base['accuracy']:.4f}")
    print(f"  val 오프셋        macro-F1 {m_plain['macro_f1']:.4f}"
          f"   ({m_plain['macro_f1']-base['macro_f1']:+.4f})")
    print(f"  재가중 오프셋     macro-F1 {m_rw['macro_f1']:.4f}  acc {m_rw['accuracy']:.4f}"
          f"   ({m_rw['macro_f1']-base['macro_f1']:+.4f})")


if __name__ == "__main__":
    main()
