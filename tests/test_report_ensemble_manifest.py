"""앙상블 구성원 매니페스트를 새 실행으로 갱신하는 도구 (18차).

`docs/experiments/ensemble_members.json` 은 `a17_tta` 와 `bench_ensemble` 이 함께 읽는
단일 목록이다. E22, E23 구성원을 손으로 넣으면 `backbone` 을 빠뜨려 **3채널 resnet18 로
잘못 읽는 사고**가 난다(`bench_ensemble` 주석에 이미 적혀 있는 함정이다).
그래서 태그에서 항목을 만드는 규칙을 시험으로 박는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from report_ensemble_manifest import entry_for_tag, merge_entries  # noqa: E402


def test_e22_entry_carries_the_backbone():
    """백본을 빠뜨리면 체크포인트를 못 읽는다. 이것이 이 시험의 이유다."""
    e = entry_for_tag("e22_shufflenet_v2_tr_s0")
    assert e["backbone"] == "shufflenet_v2"
    assert e["architecture"] == "shufflenet_v2"
    assert e["augment"] == "translate"
    assert e["representation"] == "pad64"
    assert e["ckpt"].endswith("e22_shufflenet_v2_tr_s0_best.pt")


def test_e22_handles_a_backbone_name_containing_digits():
    """`efficientnet_b0` 는 이름 안에 숫자와 밑줄이 있다. 순진한 분해가 깨지는 자리다."""
    assert entry_for_tag("e22_efficientnet_b0_tr_s2")["backbone"] == "efficientnet_b0"


def test_e23_is_resnet18_and_records_its_recipe():
    assert entry_for_tag("e23_rot_s1")["augment"] == "rotate"
    assert entry_for_tag("e23_rot_s1")["backbone"] == "resnet18"
    assert entry_for_tag("e23_rottr_s0")["augment"] == "rotate+translate"


def test_e24_entry_carries_the_padding_mode():
    """`padding_mode` 가 빠지면 평가가 zeros 모델로 돌아가 **조용히 틀린다.**

    가중치가 아니라서 `load_state_dict` 는 성공한다. 그래서 시험으로 박는다.
    """
    e = entry_for_tag("e24_pad_s0")
    assert e["padding_mode"] == "reflect"
    assert e["augment"] == ""
    assert entry_for_tag("e24_padtr_s2")["padding_mode"] == "reflect"
    assert entry_for_tag("e24_padtr_s2")["augment"] == "translate"


def test_unknown_tag_is_refused_not_guessed():
    """모르는 태그를 resnet18 기본값으로 조용히 만들면 사고가 난다."""
    with pytest.raises(ValueError):
        entry_for_tag("e99_mystery_s0")


def test_merge_keeps_existing_entries_and_their_order():
    """기존 목록을 다시 쓰면 과거 결과의 재현이 깨진다. 뒤에 붙이기만 한다."""
    old = [{"tag": "e8_pad"}, {"tag": "e17_translate_s0"}]
    out = merge_entries(old, ["e22_shufflenet_v2_tr_s0"])
    assert [e["tag"] for e in out[:2]] == ["e8_pad", "e17_translate_s0"]
    assert out[2]["tag"] == "e22_shufflenet_v2_tr_s0"


def test_merge_is_idempotent():
    """두 번 돌려도 중복이 생기지 않아야 한다 — 사이클마다 돌린다."""
    old = [{"tag": "e22_shufflenet_v2_tr_s0", "ckpt": "손으로 고친 값"}]
    out = merge_entries(old, ["e22_shufflenet_v2_tr_s0"])
    assert len(out) == 1
    assert out[0]["ckpt"] == "손으로 고친 값"
