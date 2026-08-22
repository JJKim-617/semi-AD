"""val 홀드아웃이 왜 **더 쉬운가** — 유병률을 맞춰도 남는 차이를 분해한다.

실행 전 예측 3, 4 가 틀렸다(사전등록 §7). "분포가 다르면 어려울 것" 이라 적었는데
val 은 유병률을 맞춰도 blocked AUPR 0.9291 로 test 의 0.83 보다 **높다.**

가설 두 개가 있고 여기서 가른다.

1. **클래스 구성** — val 결함의 50.0% 가 Edge-Ring 이다(dev 는 14.8%). 쉬운 쪽이 많다.
2. **정상의 성질** — val 정상의 불량 다이 비율 중앙값이 0.1544 로 dev 의 0.0877 보다 훨씬 높다.
   결함 중앙값 대비 비가 1.02 라 **개수 축에는 정보가 거의 없다.**

그래서 **dev 의 클래스 구성을 val 에 강제**해 다시 잰다. 라벨은 표본 추출에만 쓴다 —
점수 계산에는 안 들어간다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import aupr_blocked, auroc, fail_ratio  # noqa: E402
from a22_ood_template import assert_one_class  # noqa: E402
from a23_ood_template_eval import NAMES, ecdf_percentile, fisher  # noqa: E402
from a24_ood_residual import local_fail_count_map, local_fail_count_max, local_fail_density_max  # noqa: E402
from a27_line_filter import line_density_max  # noqa: E402
from a29_radial_calibration import band_index, calibrate_map, fit_band_reference  # noqa: E402
from a38_sealed_holdout import load_partition  # noqa: E402

OUT = Path("result/ood/holdout")
CHUNK, N_BANDS, L_LINE = 4000, 32, 11
TARGET_PREV, N_REP = 0.0666, 200
REASON = "예측 3, 4 오답 분해 (사후 진단, 채택 판정에 쓰지 않는다)"
T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    X = d["X"]
    trn = sp["train"][y[sp["train"]] == 0]
    assert_one_class(y[trn])
    pad_tr = np.ascontiguousarray(X[trn])
    dev, val = load_partition("test_dev"), load_partition("val_unseen", unseal=True, reason=REASON)
    pad_va = np.ascontiguousarray(X[val])
    del X, d

    v_tr = local_fail_count_map(pad_tr, k=5, chunk=CHUNK).astype(np.float32)
    ref = fit_band_reference(v_tr, pad_tr, y=y[trn], n_bands=N_BANDS, seed=0)
    del v_tr

    def a_score(x):
        out = np.empty(len(x))
        for a in range(0, len(x), CHUNK):
            xb = x[a:a + CHUNK]
            u = calibrate_map(local_fail_count_map(xb, k=5, chunk=CHUNK),
                              band_index(xb, N_BANDS), ref)
            out[a:a + len(xb)] = u.reshape(len(xb), -1).max(1)
        return out

    def comps(x):
        return {"A": a_score(x),
                "B": line_density_max(x, length=L_LINE, n_orient=8, chunk=CHUNK, min_dies=L_LINE),
                "C": local_fail_count_max(x, k=3, chunk=CHUNK)}

    ctr, cva = comps(pad_tr), comps(pad_va)
    log("요소 완료")
    scores = {
        "fuse3": fisher([ecdf_percentile(ctr[k], cva[k]) for k in ("A", "B", "C")]),
        "die k=7": local_fail_density_max(pad_va, k=7, chunk=CHUNK),
        "E0 불량 다이 비율": fail_ratio(pad_va),
    }

    yv = y[val]
    mix = np.array([(y[dev] == c).sum() for c in range(1, 9)], float)
    mix /= mix.sum()
    quota = np.floor(mix * 541).astype(int)
    quota[int(np.argmax(mix))] += 541 - quota.sum()
    pools = {c: np.flatnonzero(yv == c) for c in range(1, 9)}
    neg = np.flatnonzero(yv == 0)
    log("dev 클래스 구성 강제: " + ", ".join("%s %d/%d" % (NAMES[c], quota[c - 1], len(pools[c]))
                                        for c in range(1, 9)))
    for c in range(1, 9):
        if quota[c - 1] > len(pools[c]):
            raise ValueError("val 에 %s 가 모자라다" % NAMES[c])

    def matched(score, by_class):
        v = np.empty(N_REP)
        for r in range(N_REP):
            rng = np.random.default_rng(r)
            if by_class:
                take = np.concatenate([neg] + [rng.choice(pools[c], quota[c - 1], replace=False)
                                               for c in range(1, 9)])
            else:
                pos = np.flatnonzero(yv != 0)
                take = np.concatenate([neg, rng.choice(pos, 541, replace=False)])
            v[r] = aupr_blocked(score[take], (yv[take] != 0).astype(np.int64))
        return {"mean": float(v.mean()),
                "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}

    rep = {"quota": {NAMES[c]: int(quota[c - 1]) for c in range(1, 9)},
           "dev_class_mix": {NAMES[c]: float(mix[c - 1]) for c in range(1, 9)}, "arms": {}}
    print("\n%-18s %10s %26s %26s" % ("arm", "val AUROC", "유병률만 맞춤", "유병률+클래스구성 맞춤"))
    for n, s in scores.items():
        a = matched(s, False)
        b = matched(s, True)
        rep["arms"][n] = {"auroc_val": auroc(s, (yv != 0).astype(np.int64)),
                          "prevalence_matched": a, "prevalence_and_class_matched": b}
        print("%-18s %10.4f  %.4f [%.4f, %.4f]  %.4f [%.4f, %.4f]"
              % (n, rep["arms"][n]["auroc_val"], a["mean"], *a["ci95"], b["mean"], *b["ci95"]))

    fr = fail_ratio(pad_va)
    rep["fail_ratio_median"] = {"val_normal": float(np.median(fr[yv == 0])),
                               "val_defect": float(np.median(fr[yv != 0]))}
    (OUT / "holdout_shift.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log("저장 완료 → %s" % (OUT / "holdout_shift.json"))
