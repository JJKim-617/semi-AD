"""사후 logit 보정 (post-hoc logit adjustment).

Menon et al., "Long-tail learning via logit adjustment" (ICLR 2021), 식 (9):

    argmax_y [ f_y(x) - tau * log(pi_y) ]

학습된 모델을 그대로 두고 추론 시 logit 에 클래스별 오프셋만 더한다. 재학습이 없어
체크포인트 하나로 tau 를 수십 개 평가할 수 있다.

**우리 상황의 특수성.** 표준 long-tail 연구는 test 가 균형잡혀 있다고 보고 tau > 0 으로
경계를 희소 클래스 쪽으로 민다. 그런데 우리 test 는 none 이 93.3% 로 train(67.4%)보다
더 쏠려 있다. 방향이 반대일 수 있으므로 tau 를 음수까지 열어 스윕한다.

진단에서 나온 근거: E4b 결함 오류의 63.7% 가 none 흡수이고, 4개 실험의 클래스별 최고 F1 만
모으면 macro 0.693(현재 0.6705)이 된다. 경계 위치가 지렛대라는 뜻이다.
"""

from __future__ import annotations

import numpy as np

from a4_eval_wm811k_cls import evaluate


def estimate_prior(y: np.ndarray, num_classes: int, floor: float = 0.5) -> np.ndarray:
    """라벨에서 클래스 사전확률을 추정한다.

    실제 빈도를 그대로 쓰되, 등장하지 않는 클래스만 `floor` 로 바닥을 깐다.
    add-one 스무딩처럼 모든 클래스를 건드리면 사전확률 자체가 왜곡되는데,
    보정량이 log(prior) 에 비례하므로 그 왜곡이 그대로 경계 위치로 넘어간다.
    log(0) 만 피하면 되므로 0인 칸만 손댄다.
    """
    counts = np.bincount(np.asarray(y).ravel(), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = floor
    return counts / counts.sum()


def adjust_logits(logits: np.ndarray, prior: np.ndarray, tau: float) -> np.ndarray:
    """logit 에서 tau * log(prior) 를 뺀다. tau>0 이면 희소 클래스가 유리해진다."""
    logits = np.asarray(logits, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    if prior.shape[0] != logits.shape[1]:
        raise ValueError(f"prior 길이 {prior.shape[0]} 와 클래스 수 {logits.shape[1]} 가 다르다")
    return logits - tau * np.log(prior)[None, :]


def sweep_tau(logits: np.ndarray, y: np.ndarray, num_classes: int,
              prior: np.ndarray, taus: np.ndarray,
              metric: str = "macro_f1") -> dict:
    """tau 를 훑으며 지표를 계산하고 최고점을 찾는다."""
    taus = np.asarray(taus, dtype=np.float64)
    scores, accs = [], []
    for t in taus:
        m = evaluate(y, adjust_logits(logits, prior, t).argmax(1), num_classes)
        scores.append(m[metric])
        accs.append(m["accuracy"])
    scores = np.asarray(scores)
    i = int(scores.argmax())
    return {"taus": taus, "scores": scores, "accuracies": np.asarray(accs),
            "best_tau": float(taus[i]), "best_score": float(scores[i]),
            "best_accuracy": float(accs[i])}


def main() -> None:
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _runtime import setup
    setup("adjust")
    import torch
    from a3_train_wm811k_cls import build_model, to_onehot

    p = argparse.ArgumentParser(description="사후 logit 보정 tau 스윕")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64.npz")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--out-dir", default="result/posthoc")
    p.add_argument("--tag", default=None)
    p.add_argument("--tau-min", type=float, default=-2.0)
    p.add_argument("--tau-max", type=float, default=2.0)
    p.add_argument("--n-tau", type=int, default=81)
    a = p.parse_args()
    tag = a.tag or Path(a.ckpt).stem

    d, sp = np.load(a.cache), np.load(a.splits)
    classes = [str(c) for c in d["classes"]]
    y_all = d["y"].astype(np.int64)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(a.ckpt, map_location=device, weights_only=True))
    model.to(device).eval()

    @torch.no_grad()
    def logits_for(idx):
        out = np.empty((len(idx), len(classes)), dtype=np.float32)
        for i in range(0, len(idx), 512):
            b = idx[i:i + 512]
            out[i:i + len(b)] = model(to_onehot(d["X"][b]).to(device)).cpu().numpy()
        return out

    # 사전확률은 학습셋에서 추정한다. test 라벨은 쓰지 않는다.
    prior = estimate_prior(y_all[sp["train"]], len(classes))
    taus = np.linspace(a.tau_min, a.tau_max, a.n_tau)

    res = {}
    for split in ("val", "test"):
        idx = sp[split]
        z = logits_for(idx)
        res[split] = sweep_tau(z, y_all[idx], len(classes), prior, taus)
        print(f"[{split}] n={len(idx):,}  tau=0 macro-F1 "
              f"{res[split]['scores'][np.argmin(np.abs(taus))]:.4f}  "
              f"-> best {res[split]['best_score']:.4f} @ tau={res[split]['best_tau']:+.2f}")

    # 정직한 선택: val 에서 tau 를 고르고 test 에 적용한다.
    tv = res["val"]["best_tau"]
    j = int(np.argmin(np.abs(taus - tv)))
    sel_score, sel_acc = res["test"]["scores"][j], res["test"]["accuracies"][j]
    base_j = int(np.argmin(np.abs(taus)))
    base_score, base_acc = res["test"]["scores"][base_j], res["test"]["accuracies"][base_j]

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}_tau_sweep.json").write_text(json.dumps({
        "ckpt": a.ckpt, "prior_source": "train split",
        "prior": prior.tolist(), "classes": classes,
        "taus": taus.tolist(),
        "val_scores": res["val"]["scores"].tolist(),
        "test_scores": res["test"]["scores"].tolist(),
        "test_accuracies": res["test"]["accuracies"].tolist(),
        "tau_selected_on_val": tv,
        "test_macro_f1_at_selected_tau": float(sel_score),
        "test_accuracy_at_selected_tau": float(sel_acc),
        "test_macro_f1_at_tau0": float(base_score),
        "test_accuracy_at_tau0": float(base_acc),
        "oracle_test_best_tau": res["test"]["best_tau"],
        "oracle_test_best_macro_f1": res["test"]["best_score"],
    }, indent=2), encoding="utf-8")

    print()
    print(f"  기준선 (tau=0)          macro-F1 {base_score:.4f}  acc {base_acc:.4f}")
    print(f"  val 선택 tau={tv:+.2f}      macro-F1 {sel_score:.4f}  acc {sel_acc:.4f}"
          f"   ({sel_score - base_score:+.4f})")
    print(f"  oracle tau={res['test']['best_tau']:+.2f}       macro-F1 "
          f"{res['test']['best_score']:.4f}  (상한, 선택에 test 를 썼으므로 참고용)")


if __name__ == "__main__":
    main()
