"""WM-811K 분류 평가 지표.

이 데이터셋은 none 이 라벨의 85%(원본 Test 기준 93%)라 **accuracy 는 단독으로 무의미하다**.
none 만 찍어도 85% 가 나온다. 주 지표는 macro-F1 이고, 소수 클래스(Donut 555장,
Near-full 149장)의 recall 을 별도로 본다.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int,
             sample_weight: np.ndarray | None = None) -> dict:
    """분류 결과를 지표 묶음으로 만든다.

    Returns:
        accuracy, macro_f1, per_class_recall(클래스 인덱스 -> recall), per_class_f1, confusion.
        분할에 등장하지 않는 클래스도 키를 유지한다. 소수 클래스가 특정 분할에
        아예 없을 수 있어서다.

        sample_weight 를 주면 표본별 가중치를 반영한다. 검증셋을 타겟 분포로 재가중해
        "타겟 분포에서의 macro-F1" 을 근사할 때 쓴다.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"길이가 다르다: {y_true.shape} vs {y_pred.shape}")

    labels = list(range(num_classes))
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()
        if sample_weight.shape != y_true.shape:
            raise ValueError(f"가중치 길이 {sample_weight.shape} 가 표본 수 {y_true.shape} 와 다르다")
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0,
                          sample_weight=sample_weight)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0,
                  sample_weight=sample_weight)

    # macro 는 실제 등장한 클래스에 대해서만 평균낸다. 없는 클래스의 0 이 섞이면
    # 분할마다 지표가 달라져 비교가 깨진다.
    present = np.unique(np.concatenate([y_true, y_pred]))
    macro_f1 = float(np.mean([f1[c] for c in present]))

    return {
        "accuracy": float(np.average(y_true == y_pred, weights=sample_weight)),
        "macro_f1": macro_f1,
        "per_class_recall": {c: float(recall[c]) for c in labels},
        "per_class_f1": {c: float(f1[c]) for c in labels},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels,
                                      sample_weight=sample_weight).tolist(),
        "support": {c: int((y_true == c).sum()) for c in labels},
    }


def format_report(m: dict, classes: list[str], title: str) -> str:
    """사람이 읽는 md 요약. accuracy 단독 해석을 막기 위해 support 를 함께 보인다."""
    lines = [f"## {title}", "",
             f"- macro-F1 **{m['macro_f1']:.4f}**",
             f"- accuracy {m['accuracy']:.4f} (none 비중이 커서 단독 해석 금지)", "",
             "| 클래스 | support | recall | F1 |", "|---|---:|---:|---:|"]
    for i, c in enumerate(classes):
        lines.append(f"| {c} | {m['support'][i]:,} | "
                     f"{m['per_class_recall'][i]:.3f} | {m['per_class_f1'][i]:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _runtime import setup
    setup("eval")
    import torch
    from a3_train_wm811k_cls import build_model, predict

    p = argparse.ArgumentParser(description="WM-811K 분류 평가")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64.npz")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out-dir", default="result/cls_baseline")
    p.add_argument("--tag", default=None)
    p.add_argument("--pretrained", action="store_true",
                   help="체크포인트를 만든 모델 구조와 맞추기 위한 플래그(가중치는 ckpt 로 덮인다)")
    a = p.parse_args()
    tag = a.tag or Path(a.ckpt).stem

    d, sp = np.load(a.cache), np.load(a.splits)
    classes = [str(c) for c in d["classes"]]
    idx = sp[a.split]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(a.ckpt, map_location=device, weights_only=True))
    pred = predict(model, d["X"][idx], 512, device)
    m = evaluate(d["y"][idx].astype(np.int64), pred, len(classes))

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}_{a.split}.json").write_text(
        json.dumps({"ckpt": a.ckpt, "split": a.split, "n": len(idx), **m},
                   indent=2), encoding="utf-8")
    report = format_report(m, classes, f"{tag} — {a.split} ({len(idx):,}장)")
    (out / f"{tag}_{a.split}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
