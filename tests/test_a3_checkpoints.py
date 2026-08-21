"""주기적 체크포인트 저장 규칙 테스트.

80 epoch 가 40 보다 나빴는데(test -0.018) val 은 반대로 올랐다. 즉 val 로는 중단 시점을
고를 수 없다. 어느 에폭이 실제로 좋았는지 알려면 에폭별 체크포인트가 필요하다.
현재는 val 최고 시점 하나만 저장해 소급 검증이 불가능하다.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a3_train_wm811k_cls import should_snapshot


def test_saves_on_the_interval():
    assert should_snapshot(epoch=5, total=40, every=5)


def test_skips_between_intervals():
    assert not should_snapshot(epoch=4, total=40, every=5)


def test_always_saves_the_final_epoch():
    """마지막은 간격에 안 맞아도 남긴다. 곡선의 끝점이 없으면 해석이 어렵다."""
    assert should_snapshot(epoch=40, total=40, every=7)


def test_never_saves_when_disabled():
    assert not should_snapshot(epoch=10, total=40, every=0)
    assert not should_snapshot(epoch=40, total=40, every=0)


def test_first_epoch_is_not_special():
    assert not should_snapshot(epoch=1, total=40, every=5)


def test_rejects_negative_interval():
    with pytest.raises(ValueError):
        should_snapshot(epoch=1, total=40, every=-1)
