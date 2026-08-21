"""WM-811K 분할 인덱스 생성.

원본 `trianTestLabel`(데이터셋 자체 오타) 을 그대로 존중한다. Test 행은 학습에
절대 들어가지 않는다. 검증셋은 Training 안에서 **lot 단위**로 떼어낸다.
웨이퍼 단위 랜덤 분할을 쓰면 같은 lot 의 거의 동일한 웨이퍼맵이 train 과 val 에
나뉘어 들어가 검증 수치가 낙관적으로 왜곡된다.

실측 결과 원본 split 자체에는 lot 겹침이 없다(Training 5,809 lot, Test 4,953 lot).
누수는 우리가 새로 만드는 분할에서 생기므로 여기서 막는다.
"""

from __future__ import annotations

import numpy as np


def make_lot_split(
    lot_name: np.ndarray,
    split_label: np.ndarray,
    val_frac: float = 0.2,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """공식 split 을 유지한 채 Training 을 lot 단위로 train/val 로 나눈다.

    Args:
        lot_name: 웨이퍼별 lot 이름. shape (N,)
        split_label: 'Training' 또는 'Test'. shape (N,)
        val_frac: Training 의 **lot 중** 검증셋으로 뗄 비율. 0 이면 val 이 비어 있다.
        seed: lot 셔플 시드.

    Returns:
        {'train': idx, 'val': idx, 'test': idx} — 전부 원본 배열에 대한 정수 인덱스.
    """
    if not 0.0 <= val_frac <= 1.0:
        raise ValueError(f"val_frac 은 0 과 1 사이여야 한다: {val_frac}")

    is_test = split_label == "Test"
    test_idx = np.flatnonzero(is_test)
    train_pool = np.flatnonzero(~is_test)

    lots = np.unique(lot_name[train_pool])
    rng = np.random.default_rng(seed)
    rng.shuffle(lots)

    n_val = int(round(len(lots) * val_frac))
    val_lots = set(lots[:n_val].tolist())

    in_val = np.array([lot_name[i] in val_lots for i in train_pool], dtype=bool)
    return {
        "train": train_pool[~in_val],
        "val": train_pool[in_val],
        "test": test_idx,
    }


def main() -> None:
    import argparse
    import json

    p = argparse.ArgumentParser(description="WM-811K split 인덱스 생성")
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64.npz")
    p.add_argument("--out", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    d = np.load(a.cache)
    s = make_lot_split(d["lot_name"], d["split"], a.val_frac, a.seed)
    np.savez_compressed(a.out, **s, seed=a.seed, val_frac=a.val_frac)

    y = d["y"]
    print(f"[save] {a.out}")
    for k in ("train", "val", "test"):
        idx = s[k]
        lots = len(set(d["lot_name"][idx].tolist()))
        print(f"  {k:6s} {len(idx):>7,}  lots {lots:>6,}  "
              f"none {int((y[idx] == 0).sum()):>7,}  결함 {int((y[idx] > 0).sum()):>6,}")
    ov = set(d["lot_name"][s['train']]) & set(d["lot_name"][s['val']])
    print(f"  train/val lot 겹침: {len(ov)}")


if __name__ == "__main__":
    main()
