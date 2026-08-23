"""저장된 결과 파일의 대표 수치를 못 박는다 — 문서와 결과가 조용히 갈리는 것을 막는다.

## 왜 필요한가

`docs/reports/OOD.md` 는 수십 개 수치를 인용한다. 스크립트를 고치고 다시 돌리면
그 수치가 바뀔 수 있고, **문서는 안 바뀐다.** 그러면 보고서가 조용히 틀려진다.

이 워크스트림은 이미 **바닥 수치를 테스트로 박아 뒀다**
(`test_a21_ood_baseline_pin.py`, `test_a24_local_density_pin.py`).
여기서는 그 관례를 **오늘 만든 결과들**로 넓힌다.

## 왜 파이프라인을 다시 안 돌리나

채택 arm 을 한 번 재는 데 5분이 걸린다. 단위 시험에 넣을 수 없다.
그래서 **저장된 결과 파일**을 검사한다 — 결과를 다시 만들면 이 시험이 잡는다.

`result/` 는 대용량 저장소 심링크라 git 에 없다. **없으면 건너뛴다.**
건너뛴 것이 통과로 보이지 않게 이유를 남긴다.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULT = ROOT / "result" / "ood"


def load(rel):
    p = RESULT / rel
    if not p.exists():
        pytest.skip("결과 파일이 없다(대용량 저장소 미연결): %s" % p)
    return json.loads(p.read_text())


# --- 채택 arm — 공식 test 전수 ---------------------------------------------------

def test_adopted_fuse3_on_full_test():
    """보고서의 대표 수치. 바뀌면 보고서 전체를 다시 읽어야 한다."""
    m = load("o1_fuse3/fuse3_metrics.json")["주 fuse3 (k2k5+반경 / 선L11 / k2k3)"]
    assert m["auroc"] == pytest.approx(0.9698, abs=5e-5)
    assert m["aupr_blocked"] == pytest.approx(0.8325, abs=5e-5)
    assert m["fpr_at_95tpr"] == pytest.approx(0.1719, abs=5e-5)
    assert m["n_unique"] == 3953


def test_adopted_fuse3_per_class_on_full_test():
    pc = load("o1_fuse3/fuse3_metrics.json")["주 fuse3 (k2k5+반경 / 선L11 / k2k3)"]["per_class"]
    for name, want in (("Scratch", 0.9251), ("Center", 0.9412), ("Loc", 0.9830),
                       ("Edge-Ring", 0.9540), ("Near-full", 1.0000)):
        assert pc[name]["auroc"] == pytest.approx(want, abs=5e-5), name


# --- 개발 기준선 — `test_dev` ----------------------------------------------------

def test_fuse3_on_test_dev_is_the_bar_new_arms_must_clear():
    """11차 사이클 이후 새 arm 이 넘어야 하는 값. 오늘 네 스크립트가 독립으로 재현했다."""
    m = load("o1_weighting/weighting_metrics.json")["metrics"]["채택 fuse3 (동일 가중 Fisher)"]
    assert m["aupr_blocked"] == pytest.approx(0.8304, abs=5e-5)
    assert m["fpr_at_95tpr"] == pytest.approx(0.1877, abs=5e-5)


# --- 14차: 요소 가중은 기각됐다 ----------------------------------------------------

def test_component_weighting_lost():
    r = load("o1_weighting/weighting_metrics.json")
    corr = r["corr"]
    assert corr[0][2] == pytest.approx(0.800, abs=5e-4), "A-C 상관이 가장 높다"
    assert corr[0][1] == pytest.approx(0.758, abs=5e-4)
    assert corr[1][2] == pytest.approx(0.677, abs=5e-4)
    w1 = r["metrics"]["W1 주 (S^-1 가중 probit 합)"]["aupr_blocked"]
    assert w1 < 0.8304, "졌다는 것이 판정이다"


# --- 15차: 합 규칙 안에서 꼬리 강조에 단조 -------------------------------------------

def test_sum_rules_are_monotone_in_tail_emphasis():
    """**이 성질이 15차 사이클의 결론이다.** 깨지면 결론을 다시 읽어야 한다."""
    m = load("o1_tail/tail_metrics.json")["metrics"]
    edg = m["2 Edgington 합"]["aupr_blocked"]
    sto = m["3 Stouffer probit 합"]["aupr_blocked"]
    fis = m["4 Fisher (채택)"]["aupr_blocked"]
    assert edg < sto < fis
    assert (edg, sto, fis) == pytest.approx((0.7800, 0.8187, 0.8304), abs=5e-5)


def test_tippett_cannot_choose_an_operating_point():
    """Tippett 을 기각시킨 것은 꼬리가 아니라 눈금이다."""
    m = load("o1_tail/tail_metrics.json")["metrics"]["5 Tippett 최대 (이접)"]
    assert m["n_unique"] == 204
    tied95 = m["operating"][2]["n_tied_at_threshold"]
    assert tied95 == 8776, "95%% 재현율 문턱에서 동점인 웨이퍼 수"


# --- 17차/20차: O2 -----------------------------------------------------------------

def test_o2_loses_to_the_window_statistics_in_every_seed():
    m = load("o2_ssl_knn/o2_metrics.json")["metrics"]
    for s, want in ((0, 0.3283), (1, 0.5524), (2, 0.3307)):
        got = m["O2_trained_s%d" % s]["aupr_blocked"]
        assert got == pytest.approx(want, abs=5e-5)
        assert got < 0.8304


def test_the_epoch_curve_peaks_at_three_in_every_seed():
    """**정정 16 의 근거.** 12 epoch 은 과학습이었다."""
    for s in (0, 1, 2):
        c = load("o2_ssl_knn/epoch_curve_s%d.json" % s)
        vals = {int(k[2:]): v["aupr_blocked"] for k, v in c.items()}
        assert max(vals, key=vals.get) == 3, "seed %d 의 봉우리" % s
        assert vals[3] > vals[0], "학습이 무작위 초기화를 넘는다"


def test_collapse_ratio_decreases_with_depth_in_every_seed():
    """19차의 기작 주장. 깊이 2 는 보충으로 잰 것이다."""
    a = load("o2_ssl_knn/layer_depth_d13.json")
    b = load("o2_ssl_knn/layer_depth_d2.json")
    r = {**a, **b}
    for s in (0, 1, 2):
        ratio = [r["trained_d%d_s%d" % (d, s)]["eff_dim"]
                 / r["random_d%d_s%d" % (d, s)]["eff_dim"] for d in (1, 2, 3)]
        assert ratio[0] > ratio[1] > ratio[2], "seed %d: %s" % (s, ratio)


# --- 21차/22차: 융합해도 안 움직인다 -------------------------------------------------

def test_fuse4_does_not_move_the_primary_metric():
    r = load("o2_fuse4/fuse4_metrics.json")
    assert abs(r["fuse4_minus_fuse3_mean"]) < 0.001, "평균 이득이 사실상 0 이다"
    assert r["fuse4_seed_range"] > 10 * abs(r["fuse4_minus_fuse3_mean"]), \
        "seed 폭이 이득보다 훨씬 크다 — 이것이 21차의 판정 근거다"


def test_seed_ensemble_helps_o2_alone_but_not_the_fusion():
    """22차의 문장: 앙상블은 잡음을 줄이지만 체계적 약점을 못 고친다."""
    m = load("o2_ensemble/ensemble_metrics.json")["metrics"]
    ens_alone = m["O2 앙상블 단독 (참고)"]["aupr_blocked"]
    best_single = max(m["fuse4 s%d (참고)" % s]["aupr_blocked"] for s in (0, 1, 2))
    assert ens_alone > 0.6474, "앙상블이 최고 단일 seed 의 O2 보다 낫다"
    r = load("o2_ensemble/ensemble_metrics.json")["paired_vs_fuse3"]
    assert r["lo"] < 0 < r["hi"], "무승부여야 한다"
    assert best_single > 0.83


def test_large_wafers_are_flagged_more_often():
    """18차. 결함 라벨 없이 잰 것이다."""
    b = load("o1_size_fa/size_falsealarm.json")["bins"]["0.95"]
    assert b[">1600"]["fa_rate"] > b["562-776"]["fa_rate"] * 3
    assert b[">1600"]["ci95"][0] > b["1334-1600"]["ci95"][1], "CI 가 안 겹친다"
