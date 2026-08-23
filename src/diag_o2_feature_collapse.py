"""O2 가 **무작위 초기화보다 나빠진** 이유 — 표현이 무너졌는가.

## 왜 이걸 재는가

14차… 아니 16차 사이클(O2)에서 자기지도 학습이 표현을 **악화**시켰다.
`test_dev` blocked AUPR: 무작위 초기화 0.4986 → 학습 후 **0.3283**.
그리고 kNN 거리의 **척도 자체가 무너졌다** — 평균 0.0884 → **0.0021**(42배), 표준편차 10배.

"학습이 나빴다" 로 끝내면 기작을 모른 채 넘어가는 것이다.
이 워크스트림의 정정 6, 11 이 정확히 그런 실수였다(**틀린 기작 → 사실 아닌 "사실"**).
그래서 **표현이 실제로 무너졌는지**를 직접 잰다.

## 무엇을 재는가

같은 시드(결정론)로 같은 인코더를 다시 만들어 **무작위 초기화**와 **학습 후**를 나란히 본다.

- **유효 차원**(공분산 고윳값의 참여비 `(sum l)^2 / sum l^2`) — 32차원 중 몇 차원을 실제로 쓰는가
- **평균 쌍거리** — 특징들이 서로 얼마나 떨어져 있는가
- **결함 패치와 정상 패치의 거리 분리도** — 라벨은 **기술용으로만** 쓴다(채점에 안 들어간다)

**이것은 진단이지 arm 이 아니다.** 여기서 나온 것으로 O2 를 고치지 않는다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from _runtime import setup  # noqa: E402

setup("feat")
import torch  # noqa: E402

from a22_ood_template import assert_one_class  # noqa: E402
from a38_sealed_holdout import assert_not_sealed, load_partition  # noqa: E402
import a42_ood_ssl_knn as M  # noqa: E402
from a43_ood_ssl_eval import encode_chunk, train_encoder  # noqa: E402

OUT = Path("result/ood/o2_ssl_knn")
N_SAMPLE = 3000
T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


def patch_feats(enc, X, idx, per_wafer, seed, device="cpu"):
    """웨이퍼마다 유효 위치에서 몇 개씩 뽑아 모은다."""
    rng = np.random.default_rng(seed)
    out = []
    for a in range(0, len(idx), 256):
        b = idx[a:a + 256]
        xb = X[b]
        f = encode_chunk(enc, xb, device)
        v = M.valid_window_mask(xb, M.RF)
        for i in range(len(b)):
            sel = f[i][v[i]] if v[i].any() else f[i].reshape(-1, M.FEAT_DIM)
            if len(sel) > per_wafer:
                sel = sel[rng.choice(len(sel), per_wafer, replace=False)]
            out.append(sel)
    return np.concatenate(out)


def effective_dim(f):
    """참여비. 32 이면 모든 방향을 고르게 쓰고, 1 이면 한 방향으로 무너진 것이다."""
    lam = np.linalg.eigvalsh(np.cov(np.asarray(f, np.float64), rowvar=False))
    lam = np.clip(lam, 0, None)
    return float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-30))


def mean_pair_dist(f, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    g = f[rng.choice(len(f), min(n, len(f)), replace=False)].astype(np.float32)
    d = 1.0 - g @ g.T
    iu = np.triu_indices(len(g), 1)
    return float(d[iu].mean()), float(d[iu].std())


if __name__ == "__main__":
    torch.set_num_threads(10)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    X = d["X"]
    trn = sp["train"][y[sp["train"]] == 0]
    assert_one_class(y[trn])
    assert_not_sealed(trn, "train-none")
    dev = load_partition("test_dev")
    rng = np.random.default_rng(0)
    tr_s = trn[rng.choice(len(trn), N_SAMPLE, replace=False)]
    y_dev = y[dev]
    def_s = dev[y_dev != 0][rng.choice(int((y_dev != 0).sum()), N_SAMPLE, replace=False)]
    log("표본: train-none %d, dev 결함 %d" % (len(tr_s), len(def_s)))

    rep = {}
    for arm in ("random", "trained"):
        torch.manual_seed(0)
        enc = M.PatchEncoder()
        if arm == "trained":
            train_encoder(enc, X, trn, 0, M.EPOCHS, "cpu")
        enc.eval()
        f_tr = patch_feats(enc, X, tr_s, 8, 1)
        f_df = patch_feats(enc, X, def_s, 8, 2)
        ed_tr, ed_df = effective_dim(f_tr), effective_dim(f_df)
        m_tr, s_tr = mean_pair_dist(f_tr)
        m_df, s_df = mean_pair_dist(f_df)
        # 결함 패치가 정상 패치 분포에서 얼마나 떨어져 있는가
        g = f_tr[np.random.default_rng(3).choice(len(f_tr), 4000, replace=False)].astype(np.float32)
        cross = 1.0 - f_df[:4000].astype(np.float32) @ g.T
        rep[arm] = {
            "eff_dim_train_none": ed_tr, "eff_dim_defect": ed_df,
            "mean_pair_dist_train_none": m_tr, "std_pair_dist_train_none": s_tr,
            "mean_pair_dist_defect": m_df, "std_pair_dist_defect": s_df,
            "defect_to_normal_mean": float(cross.mean()),
            "defect_to_normal_min_mean": float(cross.min(1).mean()),
            "n_patch_train_none": int(len(f_tr)), "n_patch_defect": int(len(f_df)),
        }
        log("%-8s 유효차원 %5.2f (정상) / %5.2f (결함)  평균쌍거리 %.5f  결함-정상 최근접 %.6f"
            % (arm, ed_tr, ed_df, m_tr, rep[arm]["defect_to_normal_min_mean"]))

    print("\n%-28s %12s %12s %10s" % ("", "무작위 초기화", "학습 후", "비율"))
    for k in ("eff_dim_train_none", "eff_dim_defect", "mean_pair_dist_train_none",
              "mean_pair_dist_defect", "defect_to_normal_mean", "defect_to_normal_min_mean"):
        a, b = rep["random"][k], rep["trained"][k]
        print("%-28s %12.5f %12.5f %10.3f" % (k, a, b, b / a if a else float("nan")))

    (OUT / "feature_collapse.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log("저장 완료")
