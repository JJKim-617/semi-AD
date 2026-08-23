"""O2 파이프라인의 one-class 성질 — **비트 단위로** 박는다.

## 왜 뒤늦게 쓰는가

O2 사전등록 §5-5 가 요구한 것이다:

> **train 결함 픽셀을 망가뜨려도 test 점수가 비트 단위로 같아야 하고,
> train 정상을 망가뜨리면 달라져야 한다. 진입점 라벨 검사만으로는 부족하다.**

**17차 사이클을 돌릴 때 이 시험을 안 썼다.** `assert_one_class` 로 진입점만 막고
"라벨 가드 통과" 라고 적었는데, 그건 **라벨이 안 샜다는 증명이지 픽셀이 안 샜다는 증명이 아니다.**
`tests/test_a37_one_class_endtoend.py` 가 O1 파이프라인에 대해 정확히 이 구분을 세워 놨는데
O2 에는 옮기지 않았다. 여기서 갚는다.

## 스레드 수까지 고정해야 비트 재현이 성립한다

학습의 역전파 축약 순서가 스레드 수에 의존한다 —
같은 시드로 4/18/28 스레드에서 학습하면 **가중치가 최대 4e-3 까지 갈린다**(18차 사이클 실측).
그래서 이 시험은 **한 프로세스 안에서 스레드 수를 고정하고** 비교한다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from a43_ood_ssl_eval import o2_scores, select_train_none  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _fixed_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def toy():
    """train 12장(정상 8 + 결함 4) + test 6장. 전부 7x7 온전 창이 있는 크기."""
    rng = np.random.default_rng(0)
    n = 18
    X = np.zeros((n, 24, 24), np.uint8)
    X[:, 4:20, 4:20] = 1
    for i in range(n):
        m = rng.random((16, 16)) < 0.12
        X[i, 4:20, 4:20][m] = 2
    y = np.zeros(n, np.int64)
    y[8:12] = 7                      # train 쪽 결함 4장
    y[12:] = np.array([0, 0, 0, 7, 5, 1])
    train_idx = np.arange(12)
    test_idx = np.arange(12, 18)
    return X, y, train_idx, test_idx


def run(X, y, seed=0):
    _, _, tr, te = toy()
    return o2_scores(X, y, tr, te, seed=seed, epochs=1, bank=32, device="cpu")


# --- 라벨 선택 -------------------------------------------------------------------

def test_select_train_none_keeps_only_normals():
    X, y, tr, te = toy()
    sel = select_train_none(y, tr)
    assert set(sel.tolist()) == set(range(8))


def test_select_train_none_rejects_a_contaminated_selection():
    """정상만 남기지 않으면 `assert_one_class` 가 터져야 한다."""
    X, y, tr, te = toy()
    y2 = y.copy()
    y2[0] = 3
    sel = select_train_none(y2, tr)
    assert 0 not in sel.tolist()


# --- 핵심 성질 -------------------------------------------------------------------

def test_corrupting_train_defect_pixels_leaves_test_scores_bit_identical():
    """**어딘가에서 y==0 필터를 빠뜨렸다면 여기서 깨진다.**"""
    X, y, tr, te = toy()
    base = run(X, y)
    X2 = X.copy()
    rng = np.random.default_rng(99)
    X2[8:12] = rng.integers(0, 3, X2[8:12].shape).astype(np.uint8)   # 결함 4장을 통째로
    assert np.array_equal(base, run(X2, y))


def test_deleting_train_defect_rows_leaves_test_scores_bit_identical():
    """행을 지워도 같아야 한다 — 결함이 개수로도 안 새는지 본다."""
    X, y, tr, te = toy()
    base = run(X, y)
    keep = np.array([i for i in range(18) if i not in (8, 9, 10, 11)])
    X2, y2 = X[keep], y[keep]
    tr2 = np.arange(8)
    te2 = np.arange(8, 14)
    got = o2_scores(X2, y2, tr2, te2, seed=0, epochs=1, bank=32, device="cpu")
    assert np.array_equal(base, got)


def test_corrupting_train_normal_pixels_does_change_test_scores():
    """대조. 안 바뀌면 그건 보장이 아니라 **상수 채점기**다."""
    X, y, tr, te = toy()
    base = run(X, y)
    X2 = X.copy()
    rng = np.random.default_rng(7)
    X2[0:3] = rng.integers(0, 3, X2[0:3].shape).astype(np.uint8)
    assert not np.array_equal(base, run(X2, y))


def test_same_seed_and_thread_count_reproduces_bitwise():
    X, y, tr, te = toy()
    assert np.array_equal(run(X, y), run(X, y))


def test_different_seed_changes_the_scores():
    X, y, tr, te = toy()
    assert not np.array_equal(run(X, y, seed=0), run(X, y, seed=1))


def test_test_labels_never_touch_the_pipeline():
    """test 라벨을 통째로 바꿔도 점수가 같아야 한다 — 채점에 라벨이 들어가면 안 된다."""
    X, y, tr, te = toy()
    base = run(X, y)
    y2 = y.copy()
    y2[12:] = np.array([7, 7, 7, 0, 0, 0])
    assert np.array_equal(base, run(X, y2))
