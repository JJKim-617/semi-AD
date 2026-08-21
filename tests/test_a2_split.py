"""a2 split 로직 테스트.

검증하는 성질:
- 공식 trianTestLabel 을 존중한다 (Test 행은 학습에 절대 들어가지 않는다)
- train/val 은 lot 단위로 갈린다 (같은 lot 이 양쪽에 걸치면 누수)
- 같은 시드면 같은 분할이 나온다
- 인덱스가 유실되거나 중복되지 않는다
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a2_split_wm811k import make_lot_split


def toy():
    """lot 6개, 웨이퍼 12장. Training lot 4개, Test lot 2개."""
    lot = np.array(["l1", "l1", "l2", "l2", "l3", "l3",
                    "l4", "l4", "l5", "l5", "l6", "l6"])
    split = np.array(["Training"] * 8 + ["Test"] * 4)
    return lot, split


def test_test_rows_never_enter_train_or_val():
    lot, split = toy()
    s = make_lot_split(lot, split, val_frac=0.25, seed=0)
    test_rows = set(np.flatnonzero(split == "Test").tolist())
    assert not (set(s["train"].tolist()) & test_rows)
    assert not (set(s["val"].tolist()) & test_rows)


def test_official_test_split_is_preserved_exactly():
    lot, split = toy()
    s = make_lot_split(lot, split, val_frac=0.25, seed=0)
    assert set(s["test"].tolist()) == set(np.flatnonzero(split == "Test").tolist())


def test_no_lot_overlap_between_train_and_val():
    lot, split = toy()
    s = make_lot_split(lot, split, val_frac=0.25, seed=0)
    assert not (set(lot[s["train"]]) & set(lot[s["val"]]))


def test_train_and_val_together_cover_all_training_rows():
    lot, split = toy()
    s = make_lot_split(lot, split, val_frac=0.25, seed=0)
    covered = np.concatenate([s["train"], s["val"]])
    assert sorted(covered.tolist()) == sorted(np.flatnonzero(split == "Training").tolist())


def test_same_seed_gives_same_split():
    lot, split = toy()
    a = make_lot_split(lot, split, val_frac=0.25, seed=7)
    b = make_lot_split(lot, split, val_frac=0.25, seed=7)
    for k in ("train", "val", "test"):
        assert np.array_equal(a[k], b[k])


def test_different_seed_can_give_different_split():
    lot, split = toy()
    a = make_lot_split(lot, split, val_frac=0.25, seed=1)
    b = make_lot_split(lot, split, val_frac=0.25, seed=2)
    assert not np.array_equal(a["val"], b["val"])


def test_val_frac_zero_yields_empty_val():
    lot, split = toy()
    s = make_lot_split(lot, split, val_frac=0.0, seed=0)
    assert s["val"].size == 0


def test_rejects_val_frac_out_of_range():
    lot, split = toy()
    with pytest.raises(ValueError):
        make_lot_split(lot, split, val_frac=1.5, seed=0)
