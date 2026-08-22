"""봉인 홀드아웃 가드 — **가드 자체를 테스트한다.**

문서 규약만으로는 봉인이 안 지켜진다(9차 감사가 그것을 실측했다).
그래서 두 방향을 다 박는다: **가드가 실제로 막는가**, **명시적 플래그로 열리는가**.
사양은 `docs/experiments/candidate/ood_sealed_holdout.md` §4 에 실행 전에 적었다.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import a38_sealed_holdout as h  # noqa: E402


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    """실제 분할로 레지스트리를 만들되 산출물은 tmp 에 쓴다."""
    mp = pytest.MonkeyPatch()
    mp.setattr(h, "HOLDOUT_DIR", tmp_path_factory.mktemp("holdout"))
    r = h.build_registry()
    yield r
    mp.undo()


# --- 분할 자체 ---------------------------------------------------------------

def test_partitions_are_deterministic(reg, tmp_path, monkeypatch):
    """같은 salt, 같은 문턱이면 처음부터 다시 지어도 같은 lot 이 나온다."""
    monkeypatch.setattr(h, "HOLDOUT_DIR", tmp_path)
    again = h.build_registry()
    for k in ("test_sealed", "test_dev", "val_unseen"):
        assert np.array_equal(reg[k], again[k])


def test_sealed_and_dev_partition_the_official_test_exactly(reg):
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    te = sp["test"]
    assert len(np.intersect1d(reg["test_sealed"], reg["test_dev"])) == 0
    assert np.array_equal(np.sort(np.concatenate([reg["test_sealed"], reg["test_dev"]])),
                          np.sort(te))


def test_split_is_by_lot_not_by_wafer(reg):
    """웨이퍼 단위로 자르면 같은 lot 의 쌍둥이가 양쪽에 들어가 무의미해진다."""
    lots = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)["lot_name"]
    assert len(np.intersect1d(np.unique(lots[reg["test_sealed"]]),
                              np.unique(lots[reg["test_dev"]]))) == 0


def test_no_lot_leaks_from_train(reg):
    lots = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)["lot_name"]
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    tr_lots = np.unique(lots[sp["train"]])
    for k in ("test_sealed", "val_unseen"):
        assert len(np.intersect1d(tr_lots, np.unique(lots[reg[k]]))) == 0


def test_sealed_lot_fraction_is_the_preregistered_25pct(reg):
    """크기는 결과를 보고 바꾸지 못한다. 문턱 상수 자체를 테스트로 박는다."""
    assert h.SALT == "ood_sealed_v1"
    assert h.THRESHOLD == 2500
    n_sealed = len(np.unique(reg["sealed_lots"]))
    n_all = n_sealed + len(np.unique(reg["dev_lots"]))
    assert 0.23 <= n_sealed / n_all <= 0.27


def test_val_unseen_is_the_official_val_split(reg):
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    assert np.array_equal(np.sort(reg["val_unseen"]), np.sort(sp["val"]))


def test_both_classes_present_in_every_partition(reg):
    y = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)["y"]
    for k in ("test_sealed", "test_dev", "val_unseen"):
        yy = y[reg[k]]
        assert (yy == 0).sum() > 0 and (yy != 0).sum() > 0


# --- 가드가 막는가 -----------------------------------------------------------

def test_dev_partition_loads_without_any_flag(reg):
    idx = h.load_partition("test_dev")
    assert len(idx) == len(reg["test_dev"])


@pytest.mark.parametrize("name", ["test_sealed", "val_unseen"])
def test_sealed_partition_refuses_to_load_by_default(reg, name):
    with pytest.raises(h.SealedHoldoutError):
        h.load_partition(name)


@pytest.mark.parametrize("bad", [1, "yes", [1], 1.0])
def test_truthy_is_not_enough_only_the_literal_flag_opens_it(reg, bad):
    """`if unseal:` 로 쓰면 실수로 열린다. `is True` 여야 한다."""
    with pytest.raises(h.SealedHoldoutError):
        h.load_partition("test_sealed", unseal=bad, reason="사유")


@pytest.mark.parametrize("bad", [None, "", "   "])
def test_unsealing_without_a_reason_is_refused(reg, bad):
    with pytest.raises(h.SealedHoldoutError):
        h.load_partition("test_sealed", unseal=True, reason=bad)


def test_unknown_partition_name_is_an_error(reg):
    with pytest.raises(ValueError):
        h.load_partition("test_whatever", unseal=True, reason="사유")


# --- 플래그로 열리는가, 그리고 흔적이 남는가 ---------------------------------

def test_explicit_flag_and_reason_open_it(reg):
    idx = h.load_partition("test_sealed", unseal=True, reason="사전등록 §5 일회 평가")
    assert np.array_equal(np.sort(idx), np.sort(reg["test_sealed"]))


def test_every_unseal_appends_an_audit_record(reg):
    log = h.HOLDOUT_DIR / h.UNSEAL_LOG
    before = len(log.read_text().splitlines()) if log.exists() else 0
    h.load_partition("val_unseen", unseal=True, reason="사유 A")
    h.load_partition("val_unseen", unseal=True, reason="사유 B")
    lines = log.read_text().splitlines()
    assert len(lines) == before + 2
    rec = json.loads(lines[-1])
    assert rec["partition"] == "val_unseen" and rec["reason"] == "사유 B"
    assert rec["n"] == len(reg["val_unseen"]) and "time" in rec


# --- 파티션 API 를 우회해도 막는가 -------------------------------------------

def test_assert_not_sealed_blocks_a_contaminated_index_array(reg):
    contaminated = np.concatenate([reg["test_dev"][:100], reg["test_sealed"][:1]])
    with pytest.raises(h.SealedHoldoutError):
        h.assert_not_sealed(contaminated)


def test_assert_not_sealed_allows_a_clean_index_array(reg):
    h.assert_not_sealed(reg["test_dev"][:100])
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    h.assert_not_sealed(sp["train"])


def test_assert_not_sealed_also_guards_the_val_partition(reg):
    with pytest.raises(h.SealedHoldoutError):
        h.assert_not_sealed(reg["val_unseen"][:5])
