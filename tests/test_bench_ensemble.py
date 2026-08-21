"""앙상블 구성원의 기여를 조합 전수로 잰다.

E14 에서 단독 성능과 앙상블 기여가 다르다는 것이 확인됐다(resize96 은 단독 +0.005 로
잡음 안인데 상위 조합에 전부 들어간다). 따라서 새 구성원을 평가할 때 단독 점수를
보면 안 되고, **그 구성원을 포함한 조합과 포함하지 않은 같은 크기 조합**을 비교해야 한다.

여기 두 함수는 그 비교의 순수 계산부다. 로짓도 GPU 도 필요 없다.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench_ensemble import group_by_representation_count, leave_one_out_delta


class TestGroupByRepresentationCount:
    def test_groups_a_single_representation_combo_under_one(self):
        reps = {"a": "pad", "b": "pad", "c": "pad"}
        got = group_by_representation_count([("a", "b", "c")], reps)
        assert list(got) == [1]

    def test_counts_distinct_representations_not_members(self):
        reps = {"a": "pad", "b": "pad", "c": "polar"}
        got = group_by_representation_count([("a", "b", "c")], reps)
        assert list(got) == [2]

    def test_keeps_every_combo_in_exactly_one_group(self):
        reps = {"a": "pad", "b": "resize", "c": "polar", "d": "pad"}
        combos = [("a", "b"), ("a", "d"), ("b", "c"), ("a", "b", "c")]
        got = group_by_representation_count(combos, reps)
        assert sum(len(v) for v in got.values()) == len(combos)

    def test_raises_when_a_member_has_no_representation(self):
        with pytest.raises(KeyError):
            group_by_representation_count([("a", "z")], {"a": "pad"})


class TestLeaveOneOutDelta:
    def test_reports_the_gap_between_combos_with_and_without_a_member(self):
        scores = {
            ("a", "b"): 0.70, ("a", "c"): 0.72,   # a 포함
            ("b", "c"): 0.60,                      # a 미포함
        }
        d = leave_one_out_delta(scores, "a")
        assert d[2]["with_mean"] == pytest.approx(0.71)
        assert d[2]["without_mean"] == pytest.approx(0.60)
        assert d[2]["delta"] == pytest.approx(0.11)

    def test_separates_the_comparison_by_combo_size(self):
        """크기가 다른 조합을 섞어 비교하면 앙상블 크기 효과가 기여로 오인된다."""
        scores = {
            ("a", "b"): 0.70, ("b", "c"): 0.60,
            ("a", "b", "c"): 0.80, ("b", "c", "d"): 0.75,
        }
        d = leave_one_out_delta(scores, "a")
        assert set(d) == {2, 3}
        assert d[2]["delta"] == pytest.approx(0.10)
        assert d[3]["delta"] == pytest.approx(0.05)

    def test_skips_sizes_where_one_side_is_empty(self):
        """비교 상대가 없으면 숫자를 만들지 않는다."""
        scores = {("a", "b"): 0.70}
        assert leave_one_out_delta(scores, "a") == {}

    def test_counts_how_many_combos_went_into_each_side(self):
        scores = {("a", "b"): 0.7, ("a", "c"): 0.8, ("b", "c"): 0.6}
        d = leave_one_out_delta(scores, "a")
        assert d[2]["n_with"] == 2 and d[2]["n_without"] == 1

    def test_member_order_inside_a_combo_does_not_matter(self):
        scores = {("b", "a"): 0.70, ("b", "c"): 0.60}
        d = leave_one_out_delta(scores, "a")
        assert d[2]["n_with"] == 1 and d[2]["delta"] == pytest.approx(0.10)
