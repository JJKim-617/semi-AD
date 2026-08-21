"""BatchNorm 통계를 타겟 분포로 다시 잰다.

BN 의 running mean, var 은 학습 분포의 통계다. 타겟 분포가 다르면 각 층의 정규화가
어긋난 채로 추론하게 된다. 라벨 없는 타겟 데이터를 흘려보내 이 통계만 다시 재면
정규화가 타겟에 맞춰진다.

**가중치는 건드리지 않고 라벨도 쓰지 않는다.** 사후 오프셋 튜닝이 마지막 결정 경계만
옮기는 것과 달리, 이 방법은 모든 층의 표현을 타겟 통계에 맞춘다.

transductive 설정임에 주의한다. 테스트 데이터를 배치로 한 번에 볼 수 있어야 한다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def count_bn_layers(model: nn.Module) -> int:
    """모델 안의 BatchNorm 층 수. 적응이 의미 있는지 먼저 확인할 때 쓴다."""
    return sum(1 for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))


@torch.no_grad()
def adapt_bn_stats(model: nn.Module, X: np.ndarray, batch_size: int = 256,
                   device: str = "cuda", momentum: float | None = None) -> nn.Module:
    """타겟 데이터로 BN running statistics 를 다시 잰다.

    Args:
        model: 적응할 모델. 제자리에서 수정된다.
        X: 타겟 입력 (N, H, W) uint8. 라벨은 필요 없다.
        momentum: None 이면 누적 평균으로 완전히 새로 잰다(기존 통계를 버린다).
            0~1 값을 주면 그 momentum 으로 기존 통계와 섞는다.

    Returns:
        같은 모델. eval 모드로 돌려놓는다.
    """
    from a3_train_wm811k_cls import to_onehot

    X = np.asarray(X)
    if len(X) == 0:
        raise ValueError("타겟 데이터가 비어 있다")

    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return model.eval()

    saved = []
    for bn in bns:
        saved.append(bn.momentum)
        bn.reset_running_stats() if momentum is None else None
        # momentum=None 이면 PyTorch 가 누적 평균을 쓴다(num_batches_tracked 기반).
        bn.momentum = momentum

    model.to(device)
    model.eval()          # 드롭아웃 등은 끄고
    for bn in bns:
        bn.train()        # BN 만 통계를 갱신하게 둔다

    for i in range(0, len(X), batch_size):
        model(to_onehot(X[i:i + batch_size]).to(device))

    for bn, mom in zip(bns, saved):
        bn.momentum = mom
    return model.eval()


def main() -> None:
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _runtime import setup
    setup("bnadapt")
    from a3_train_wm811k_cls import build_model, predict
    from a4_eval_wm811k_cls import evaluate

    p = argparse.ArgumentParser(description="BN 통계 타겟 적응")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64pad.npz")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--out-dir", default="result/bn_adapt")
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    tag = a.tag or Path(a.ckpt).stem

    d, sp = np.load(a.cache), np.load(a.splits)
    classes = [str(c) for c in d["classes"]]
    n = len(classes)
    y = d["y"].astype(np.int64)
    te, va = sp["test"], sp["val"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def fresh():
        m = build_model(num_classes=n, pretrained=False)
        m.load_state_dict(torch.load(a.ckpt, map_location=device, weights_only=True))
        return m

    base_model = fresh()
    print(f"  BN 층 {count_bn_layers(base_model)}개")
    base = evaluate(y[te], predict(base_model, d["X"][te], 512, device), n)

    # 본 실험: test 로 BN 재추정
    m_test = adapt_bn_stats(fresh(), d["X"][te], 512, device)
    r_test = evaluate(y[te], predict(m_test, d["X"][te], 512, device), n)

    # 대조군: val 로 재추정. val 은 학습 분포라 이득이 없어야 한다.
    # BN 은 val 로 재추정하되 평가는 test 로 한다. 적응 대상만 다르고 평가는 동일해야
    # 비교가 성립한다.
    m_val = adapt_bn_stats(fresh(), d["X"][va], 512, device)
    r_val = evaluate(y[te], predict(m_val, d["X"][te], 512, device), n)

    # 혼합: 기존 통계를 일부 유지
    m_mix = adapt_bn_stats(fresh(), d["X"][te], 512, device, momentum=0.05)
    r_mix = evaluate(y[te], predict(m_mix, d["X"][te], 512, device), n)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}_bn.json").write_text(json.dumps({
        "ckpt": a.ckpt, "classes": classes,
        "base": {k: base[k] for k in ("macro_f1", "accuracy")},
        "bn_test": {k: r_test[k] for k in ("macro_f1", "accuracy")},
        "bn_val_control": {k: r_val[k] for k in ("macro_f1", "accuracy")},
        "bn_mix_momentum005": {k: r_mix[k] for k in ("macro_f1", "accuracy")},
        "bn_test_per_class_f1": r_test["per_class_f1"],
    }, indent=2), encoding="utf-8")

    print(f"  기준선              macro-F1 {base['macro_f1']:.4f}  acc {base['accuracy']:.4f}")
    print(f"  BN <- test          macro-F1 {r_test['macro_f1']:.4f}  acc {r_test['accuracy']:.4f}"
          f"   ({r_test['macro_f1']-base['macro_f1']:+.4f})")
    print(f"  BN <- val (대조군)  macro-F1 {r_val['macro_f1']:.4f}"
          f"   ({r_val['macro_f1']-base['macro_f1']:+.4f})")
    print(f"  BN <- test (혼합)   macro-F1 {r_mix['macro_f1']:.4f}"
          f"   ({r_mix['macro_f1']-base['macro_f1']:+.4f})")


if __name__ == "__main__":
    main()
