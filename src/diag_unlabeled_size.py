"""미라벨 638K 의 크기 분포 — 참조를 바꿀 이유가 있는지 **라벨 없이** 확인한다.

반증 조건은 `candidate/ood_unlabeled_reference.md` §5 에 실행 전에 박았다.

## 이 스크립트가 읽는 것

**`die_size` 와 `y` 의 부호뿐이다**(`y < 0` 이 미라벨). **픽셀을 안 연다.**
미라벨의 결함 여부는 애초에 라벨이 없어 알 수 없고, 알아내려 하지도 않는다.

## 왜 이것부터인가

638K 를 쓰는 **유일한 동기**가 "참조가 평가 분포를 더 잘 대표한다" 인데
**그 전제부터 확인이 안 됐다.** 전제가 틀리면 나머지가 무의미하다.
`die_size` 는 라벨이 아니라 **공짜로 확인된다.**
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a23_ood_template_eval import BINS, BIN_LBL  # noqa: E402
from a38_sealed_holdout import load_partition  # noqa: E402
from a41_shape_stats import mannwhitney_u  # noqa: E402

OUT = Path("result/ood/o1_unlabeled_size")
EVID = Path("docs/research/ood_scratch_shift/evidence")


def hist(size, edges):
    b = np.searchsorted(edges, size, side="right")
    return np.array([(b == i).mean() for i in range(len(BIN_LBL))])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    EVID.mkdir(parents=True, exist_ok=True)
    alld = np.load("data/wm811k/cache/wm811k_64pad_all.npz", allow_pickle=True)
    y_all = alld["y"]
    size_all = alld["die_size"].astype(np.float64)
    unl = size_all[y_all < 0]                      # 미라벨. **픽셀을 안 읽는다.**

    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    size = d["die_size"].astype(np.float64)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    trn = sp["train"][y[sp["train"]] == 0]
    dev = load_partition("test_dev")
    devn = dev[y[dev] == 0]

    edges = np.array(BINS[1:-1], np.float64)
    groups = {"미라벨 638K": unl, "train-none (현행 참조)": size[trn],
              "test_dev 정상 (평가 대상)": size[devn]}
    rep = {"n": {k: int(len(v)) for k, v in groups.items()},
           "median": {k: float(np.median(v)) for k, v in groups.items()},
           "hist": {k: hist(v, edges).tolist() for k, v in groups.items()}}

    print("%-26s %9s %10s" % ("집단", "장수", "die 중앙값"))
    for k, v in groups.items():
        print("%-26s %9d %10.0f" % (k, len(v), np.median(v)))

    print("\n%-26s " % "집단" + " ".join("%10s" % b for b in BIN_LBL))
    for k, v in groups.items():
        print("%-26s " % k + " ".join("%9.1f%%" % (100 * x) for x in hist(v, edges)))

    tgt = np.median(size[devn])
    d_unl = abs(np.median(unl) - tgt)
    d_trn = abs(np.median(size[trn]) - tgt)
    rep["dist_to_dev_median"] = {"미라벨": float(d_unl), "train-none": float(d_trn)}
    print("\n== 반증 조건 1: dev 정상 중앙값(%.0f)까지의 거리 ==" % tgt)
    print("  미라벨      %7.1f" % d_unl)
    print("  train-none  %7.1f" % d_trn)
    closer = d_unl < d_trn
    rep["unlabeled_is_closer"] = bool(closer)
    print("  -> %s" % ("미라벨이 더 가깝다 — 전제 성립" if closer
                       else "**train-none 이 더 가깝다 — 반증 조건 1 발동, 방향을 닫는다**"))

    u, p = mannwhitney_u(unl, size[trn])
    rep["mannwhitney_unlabeled_vs_trainnone_p"] = p
    print("\n== 반증 조건 2: 미라벨 대 train-none 크기 분포 ==")
    print("  Mann-Whitney p = %.3e  -> %s"
          % (p, "구분된다" if p < 0.05 else "**구분 안 된다 — 반증 조건 2 발동**"))

    # 층별로 얼마나 늘어나는가 — U3(층 균형)이 실제로 가능한지
    bt = np.searchsorted(edges, size[trn], side="right")
    bu = np.searchsorted(edges, unl, side="right")
    print("\n== 참고: 층별 장수 (U3 이 가능한가) ==")
    print("%-12s %12s %14s %10s" % ("구간", "train-none", "미라벨", "배수"))
    rep["per_bin_counts"] = {}
    for i, lbl in enumerate(BIN_LBL):
        a, b = int((bt == i).sum()), int((bu == i).sum())
        rep["per_bin_counts"][lbl] = {"train_none": a, "unlabeled": b}
        print("%-12s %12d %14d %10s" % (lbl, a, b, "%.1fx" % (b / a) if a else "-"))

    (OUT / "unlabeled_size.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    (EVID / "unlabeled_size.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print("\n저장 완료 → %s" % OUT)
