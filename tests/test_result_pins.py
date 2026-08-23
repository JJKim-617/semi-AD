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


# --- 12차: Scratch 경고등 -----------------------------------------------------------

def test_scratch_cross_decomposition():
    """2x2 교차 분해. 보고서가 인용하는 네 숫자를 그대로 박는다."""
    c = load("o1_scratch_shift/scratch_shift.json")["cross"]["fuse3"]["Scratch"]
    assert c["devpos_devneg"] == pytest.approx(0.9197, abs=5e-5)
    assert c["devpos_valneg"] == pytest.approx(0.8434, abs=5e-5)
    assert c["valpos_devneg"] == pytest.approx(0.8521, abs=5e-5)
    assert c["valpos_valneg"] == pytest.approx(0.7835, abs=5e-5)
    assert c["n_val"] == 92 and c["n_dev"] == 517


def test_scratch_is_the_only_class_where_val_positives_are_harder():
    """**이것이 12차의 판정 근거다.** 다른 클래스는 val 양성이 더 쉬워진다."""
    cross = load("o1_scratch_shift/scratch_shift.json")["cross"]["fuse3"]
    harder = [c for c, r in cross.items()
              if r["valpos_devneg"] < r["devpos_devneg"] - 1e-9]
    assert harder == ["Scratch"], harder


def test_scratch_null_excludes_the_observation():
    """dev 부표집 귀무 2,000회 중 관측값 아래가 0회."""
    n = load("o1_scratch_shift/scratch_shift.json")["subsample_null"]["fuse3"]["Scratch"]
    assert n["n"] == 92 and n["n_rep"] == 2000
    assert n["ci95"][0] == pytest.approx(0.8928, abs=5e-4)
    assert n["frac_below_observed"] == 0.0


# --- 정정 17: 표와 문장이 어긋나면 여기서 깨진다 ---------------------------------------

def test_correction_17_train_none_and_dev_have_different_size_distributions():
    """**정정 17 은 "숫자는 표에 이미 있었는데 읽기가 틀렸다" 는 종류였다.**

    그 종류를 막으려면 **문장이 인용하는 비율을 원자료에서 다시 세는 시험**이 있어야 한다.
    여기서는 결과 파일이 아니라 **캐시에서 직접 센다** — 표가 아니라 사실을 박는 것이다.
    """
    import numpy as np
    cache = ROOT / "data" / "wm811k" / "cache"
    if not (cache / "wm811k_64pad.npz").exists():
        pytest.skip("데이터 캐시 미연결")
    sys.path.insert(0, str(ROOT / "src"))
    from a38_sealed_holdout import load_partition
    from a23_ood_template_eval import BINS

    d = np.load(cache / "wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    size = d["die_size"].astype(np.float64)
    sp = np.load(cache / "splits_v1.npz")
    trn = sp["train"][y[sp["train"]] == 0]
    dev = load_partition("test_dev")
    devn = dev[y[dev] == 0]

    edges = np.array(BINS[1:-1], np.float64)
    bt = np.searchsorted(edges, size[trn], side="right")
    bd = np.searchsorted(edges, size[devn], side="right")
    # 구간 1 = 400-562
    assert (bt == 1).mean() == pytest.approx(0.773, abs=0.002), "train-none 의 400-562 비중"
    assert (bd == 1).mean() == pytest.approx(0.138, abs=0.002), "dev 정상의 400-562 비중"
    assert np.median(size[trn]) == pytest.approx(518.0)
    assert np.median(size[devn]) == pytest.approx(844.0)
    # 문장의 핵심: 참조가 한 구간에 몰려 있고 평가는 퍼져 있다
    assert (bt == 1).mean() > 5 * (bd == 1).mean()


def test_correction_13_dev_has_almost_no_large_wafer_scratches():
    """정정 13 이 인용하는 6장 / 517장."""
    import numpy as np
    cache = ROOT / "data" / "wm811k" / "cache"
    if not (cache / "wm811k_64pad.npz").exists():
        pytest.skip("데이터 캐시 미연결")
    sys.path.insert(0, str(ROOT / "src"))
    from a38_sealed_holdout import load_partition
    d = np.load(cache / "wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    size = d["die_size"].astype(np.float64)
    dev = load_partition("test_dev")
    scr = dev[y[dev] == 7]
    assert len(scr) == 517
    assert int((size[scr] >= 1600).sum()) == 6


# --- 24차: 고재현율 목적함수 -----------------------------------------------------------

def test_high_recall_verdict_nothing_is_adopted():
    """**세 arm 다 무승부이거나 기각이다.** 하나라도 통과하면 판정을 다시 써야 한다."""
    r = load("o2_high_recall/high_recall_metrics.json")
    # H4 앙상블: CI 는 0 을 배제하는데 이득이 seed 폭보다 작다
    assert r["h4_gain_vs_range"]["exceeds"] is False
    assert r["paired_fpr"]["H4 fuse5 앙상블"]["hi"] < 0
    # H1: 세 seed 점추정은 전부 대조보다 낮지만 평균 이득 < seed 폭
    assert r["h1_worst"] < r["metrics"]["H3 fuse3 (대조)"]["fpr_at_95tpr"]
    h1 = [r["metrics"]["H1 fuse4 s%d" % s]["fpr_at_95tpr"] for s in (0, 1, 2)]
    gain = r["metrics"]["H3 fuse3 (대조)"]["fpr_at_95tpr"] - sum(h1) / 3
    assert gain < r["seed_range_21"], "이득이 seed 폭보다 작다는 것이 판정 근거다"


def test_tippett_wins_the_objective_but_cannot_be_operated_there():
    """**24차에서 가장 날카로운 줄.** 이기는데 그 운영점에서 문턱을 못 고른다."""
    r = load("o2_high_recall/high_recall_metrics.json")
    t = r["metrics"]["H2 Tippett 최대"]
    f = r["metrics"]["H3 fuse3 (대조)"]
    assert r["paired_fpr"]["H2 Tippett 최대"]["hi"] < 0, "선언된 목적함수를 이긴다"
    assert t["n_tied_at_95"] == 8776 and f["n_tied_at_95"] == 23
    assert t["n_tied_at_95"] > r["tie_limit"]
    assert r["aupr_drop"]["H2 Tippett 최대"] > r["aupr_drop_limit"]


def test_one_fuse4_seed_is_not_distinguishable_from_the_control():
    """사전등록의 약점을 박아 둔다 — 조건 1 은 점추정 규칙이라 이걸 못 잡았다."""
    r = load("o2_high_recall/high_recall_metrics.json")["paired_fpr"]["H1 fuse4 s0"]
    assert r["lo"] < 0 < r["hi"], "seed 0 의 CI 는 0 을 포함한다"


# --- 25차: 미라벨 참조의 전제 -------------------------------------------------------

def test_unlabeled_is_closer_to_the_evaluation_distribution():
    """25차의 전제. 이게 깨지면 그 방향 전체가 무의미해진다."""
    r = load("o1_unlabeled_size/unlabeled_size.json")
    assert r["unlabeled_is_closer"] is True
    tv = r["tv_to_dev_normals"]
    assert tv["미라벨 638K"] < tv["train-none (현행 참조)"]
    assert tv["train-none (현행 참조)"] / tv["미라벨 638K"] > 1.9


def test_unlabeled_is_skewed_the_other_way():
    """**예측 밖의 사실.** 그냥 갈아 끼우면 반대 방향으로 틀린다."""
    h = load("o1_unlabeled_size/unlabeled_size.json")["hist"]
    assert h["미라벨 638K"][6] > 0.30, ">1600 비중이 크다"
    assert h["test_dev 정상 (평가 대상)"][6] < 0.01, "dev 정상은 거의 없다"


def test_the_two_coverage_holes_have_plenty_of_unlabeled_wafers():
    """U3 이 자료 부족으로 막히지 않는다는 것."""
    c = load("o1_unlabeled_size/unlabeled_size.json")["per_bin_counts"]
    for lbl, lo in (("562-776", 100), ("1090-1334", 100)):
        assert c[lbl]["train_none"] < 400
        assert c[lbl]["unlabeled"] > lo * c[lbl]["train_none"]
