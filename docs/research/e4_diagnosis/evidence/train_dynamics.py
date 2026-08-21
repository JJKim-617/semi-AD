"""학습 곡선과 class weight 분석.

*_train.json 의 epoch history 를 정독하고, 실제 역빈도 class weight 를
train split 라벨로 재계산한다.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from a3_train_wm811k_cls import class_weights  # noqa: E402

RES = Path("result/cls_baseline")
CLASSES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Loc", "Random", "Scratch", "Near-full"]
EXPS = {
    "E1": "e1_scratch_train.json",
    "E3": "e3_aug_train.json",
    "E4": "e4_cw_train.json",
    "E4b": "e4_aug_cw_train.json",
}


def main():
    for tag, fname in EXPS.items():
        d = json.loads((RES / fname).read_text())
        print(f"\n== {tag} ({fname}) best_val_macro_f1={d['best_val_macro_f1']:.4f} "
              f"seconds={d.get('seconds')}")
        print(f"{'ep':>3s} {'loss':>9s} {'d_loss':>9s} {'val_mF1':>8s} {'val_acc':>8s}")
        prev = None
        for h in d["history"]:
            dl = "" if prev is None else f"{h['loss']-prev:+.4f}"
            print(f"{h['epoch']:3d} {h['loss']:9.4f} {dl:>9s} "
                  f"{h['val_macro_f1']:8.4f} {h['val_accuracy']:8.4f}")
            prev = h["loss"]
        # 수렴 판단 근거: 마지막 3 epoch loss 변화율
        losses = [h["loss"] for h in d["history"]]
        f1s = [h["val_macro_f1"] for h in d["history"]]
        best_ep = int(np.argmax(f1s)) + 1
        print(f"   best val epoch: {best_ep} / {len(f1s)}")
        print(f"   마지막 3ep loss: {losses[-3:]}, 총 감소 {losses[0]-losses[-1]:.4f}, "
              f"마지막 구간 감소 {losses[-4]-losses[-1]:.4f}")

    # class weight 재계산
    print("\n\n== 역빈도 class weight (train split 실측) ==")
    d = np.load("data/wm811k/cache/wm811k_64.npz")
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    y = d["y"].astype(np.int64)
    tr = sp["train"]
    w = class_weights(y[tr], 9).numpy()
    counts = np.bincount(y[tr], minlength=9)
    print(f"{'class':10s} {'n_train':>8s} {'weight':>10s} {'w/w_none':>9s}")
    for i, c in enumerate(CLASSES):
        print(f"{c:10s} {counts[i]:8d} {w[i]:10.4f} {w[i]/w[0]:9.1f}")
    print(f"weight 합(기대 샘플가중 평균 대비): max/min = {w.max()/w[w>0].min():.1f}")
    # 배치 관점: 배치 256 에서 Near-full 등장 확률과 등장 시 손실 기여
    p = counts / counts.sum()
    print(f"\n{'class':10s} {'배치256 기대수':>12s} {'미등장확률':>10s} {'가중손실 점유율(균형시)':>14s}")
    share = w * counts / (w * counts).sum()
    for i, c in enumerate(CLASSES):
        miss = (1 - p[i]) ** 256
        print(f"{c:10s} {p[i]*256:12.2f} {miss:10.4f} {share[i]:14.4f}")


if __name__ == "__main__":
    main()
