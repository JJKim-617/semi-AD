"""준지도 학습 — FixMatch 식 일관성 정칙화 (E25).

**18차는 이것을 돌리지 않는다.** 마감 안에 3 seed 가 안 들어간다.
구현, 시험, 에폭당 비용 측정까지만 하고 다음 사이클에 넘긴다.
반증 조건은 `docs/experiments/candidate/semi_supervised.md` 에 **실행 전에** 박았다.

## 왜 이 축인가 (18차 요약)

- 앙상블이 포화했고(k=4 0.7903, k=25 0.7957) 운영점은 이미 최적이다.
- 이진 판별은 AUROC 0.983 이고 남은 것은 꼬리 2,457장이다.
- **미검출 1,554장은 어떤 조리법으로도 ±13% 밖으로 안 움직인다.**
- 손으로 만든 신호(밀도, 선, lot, 크기, 극좌표)는 전부 닫혔다.
- **안 쓴 자원이 하나 있다 — 미라벨 638,507장(라벨 학습의 14.8배).**

## 이 데이터의 제약

**라벨을 보존하는 변환이 dihedral 과 정수 이동 둘뿐이다**(18차가 확인).
그래서 약한/강한 증강의 차이가 작다. **이 실험의 최대 구조적 약점**이고
후보 문서에 미리 적어 뒀다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pseudo_labels(probs: torch.Tensor, tau: float = 0.95):
    """약한 증강 판의 **확률**에서 의사 라벨과 신뢰 마스크를 만든다.

    `probs` 는 확률이어야 한다 — 로짓을 넘기면 거절한다. 조용히 통과시키면
    임계 `tau` 가 아무 뜻도 없어진다.

    **의사 라벨에는 기울기가 흐르지 않는다.** 흐르면 자기강화가 폭주한다.
    경계값은 **포함**한다(`>= tau`).
    """
    if probs.dim() != 2:
        raise ValueError(f"(N, C) 확률이어야 한다: {tuple(probs.shape)}")
    if probs.min() < -1e-6 or not torch.allclose(
            probs.sum(1), torch.ones(len(probs), device=probs.device), atol=1e-3):
        raise ValueError("행이 확률 분포가 아니다 (로짓을 넘겼나?)")
    conf, lab = probs.detach().max(1)
    return lab.detach(), (conf >= tau)


def consistency_loss(logits_strong: torch.Tensor, labels: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """강한 증강 판에 거는 교차엔트로피. **분모는 배치 크기다.**

    남은 개수로 나누면 초기(신뢰 표본이 한둘일 때)에 그 한둘이 손실을 지배한다.
    FixMatch 의 정의가 배치 크기로 나누는 것이고, 이 선택이 안정성을 가른다.
    """
    if mask.sum() == 0:
        return logits_strong.sum() * 0.0        # 그래프를 끊지 않으면서 0
    ce = F.cross_entropy(logits_strong, labels, reduction="none")
    return (ce * mask.float()).sum() / mask.numel()


def main():
    """비용 측정 전용. **학습하지 않는다** — 에폭당 시간만 잰다.

    후보 문서가 "비용을 모른다, 스모크로 재고 큐를 걸어라" 로 남겨 뒀으므로
    그 숫자를 여기서 만들어 넘긴다.
    """
    import argparse
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _runtime import setup
    setup("train")
    import numpy as np

    from a3_train_wm811k_cls import build_model, encode_device
    from a18_augment import apply_recipe

    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64pad.npz")
    p.add_argument("--all-cache", default="data/wm811k/cache/wm811k_64pad_all.npz")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--mu", type=int, default=7, help="라벨 1개당 미라벨 개수")
    p.add_argument("--tau", type=float, default=0.95)
    p.add_argument("--steps", type=int, default=30, help="비용 측정용 스텝 수")
    a = p.parse_args()

    d = np.load(a.cache); sp = np.load(a.splits)
    X, y = d["X"], d["y"].astype(np.int64)
    tr = sp["train"]
    da = np.load(a.all_cache)
    ya = da["y"].astype(np.int64)
    unl = np.where(ya < 0)[0]
    print(f"라벨 학습 {len(tr):,}  미라벨 {len(unl):,}  (배율 {len(unl)/len(tr):.1f})")

    Xa = da["X"]
    model = build_model(num_classes=9).cuda().train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)

    t0 = time.time(); kept = 0
    for step in range(a.steps):
        li = rng.choice(tr, a.batch, replace=False)
        ui = rng.choice(unl, a.batch * a.mu, replace=False)
        xl = apply_recipe(X[li], ["translate"], rng)
        xu = Xa[np.sort(ui)]
        xw = xu                                   # 약한 증강: dihedral 은 아래에서
        xs = apply_recipe(xu, ["translate"], rng)  # 강한 증강
        tl = encode_device(xl, device="cuda")
        tw = encode_device(xw, device="cuda")
        ts = encode_device(xs, device="cuda")
        out_l = model(tl)
        loss_l = F.cross_entropy(out_l, torch.from_numpy(y[li]).cuda())
        with torch.no_grad():
            pw = torch.softmax(model(tw).float(), 1)
        lab, mask = pseudo_labels(pw, a.tau)
        kept += int(mask.sum())
        loss_u = consistency_loss(model(ts), lab, mask)
        loss = loss_l + loss_u
        opt.zero_grad(); loss.backward(); opt.step()
    dt = time.time() - t0
    per_step = dt / a.steps
    steps_per_epoch = len(tr) // a.batch
    print(f"스텝당 {per_step*1000:.0f} ms  (배치 {a.batch}, mu={a.mu})")
    print(f"에폭당 {per_step*steps_per_epoch:.0f} 초  ({steps_per_epoch} 스텝)")
    print(f"40 에폭 = {per_step*steps_per_epoch*40/3600:.2f} 시간  "
          f"-> 3 seed = {per_step*steps_per_epoch*40*3/3600:.2f} 시간")
    print(f"tau={a.tau} 를 넘은 미라벨 비율 "
          f"{kept/(a.steps*a.batch*a.mu):.1%}  (**초기 모델 기준이라 낮다**)")


if __name__ == "__main__":
    main()
