"""깊이별 붕괴 — 17차 사이클의 기작 주장을 직접 검정한다.

반증 조건은 `candidate/ood_o2_layer_depth.md` 에 실행 전에 박았다.
**성능이 아니라 유효 차원의 깊이 의존성으로 판정한다**(§3 의 교란 때문에).
`test_dev` 만 쓴다. **어떤 깊이도 채택 arm 으로 올리지 않는다.**
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from _runtime import setup  # noqa: E402

setup("feat")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from a21_ood_metrics import aupr_blocked, auroc, fpr_at_tpr  # noqa: E402
from a22_ood_template import assert_one_class  # noqa: E402
from a23_ood_template_eval import NAMES, _fast_auroc  # noqa: E402
from a38_sealed_holdout import assert_not_sealed, load_partition  # noqa: E402
import a42_ood_ssl_knn as M  # noqa: E402
from a43_ood_ssl_eval import gather_bank, score_partition, train_encoder  # noqa: E402
from diag_o2_feature_collapse import effective_dim, patch_feats  # noqa: E402

OUT = Path("result/ood/o2_ssl_knn")
# nn.Sequential 안에서 각 깊이가 끝나는 위치. 한 층 = conv + BN + ReLU 세 개다.
DEPTH_BLOCKS = {d: 3 * d for d in (1, 2, 3)}
T0 = time.time()


def log(m):
    print("[%7.1fs] %s" % (time.time() - T0, m), flush=True)


class DepthEncoder(nn.Module):
    """같은 인코더의 **앞부분만** 쓴다. 출력은 위치별 L2 정규화(O2 와 같은 규약)."""

    def __init__(self, enc: nn.Module, depth: int):
        super().__init__()
        self.body = nn.Sequential(*list(enc.body.children())[:DEPTH_BLOCKS[depth]])

    def forward(self, x):
        f = self.body(x)
        return f / f.pow(2).sum(1, keepdim=True).clamp_min(1e-12).sqrt()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--depths", default="1,3")
    p.add_argument("--threads", type=int, default=20)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    seeds = [int(s) for s in a.seeds.split(",")]
    depths = [int(s) for s in a.depths.split(",")]

    sp = np.load("data/wm811k/cache/splits_v1.npz")
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    X = d["X"]
    trn = sp["train"][y[sp["train"]] == 0]
    assert_one_class(y[trn])
    assert_not_sealed(trn, "train-none")
    dev = load_partition("test_dev")
    assert_not_sealed(dev, "test_dev")
    y_dev = y[dev]
    isd = (y_dev != 0).astype(np.int64)
    rng = np.random.default_rng(0)
    tr_s = trn[rng.choice(len(trn), 3000, replace=False)]
    log("train-none %d / test_dev %d (결함 %d)" % (len(trn), len(dev), int(isd.sum())))

    rep = {}
    for seed in seeds:
        for arm in ("random", "trained"):
            torch.manual_seed(seed)
            enc = M.PatchEncoder()
            if arm == "trained":
                train_encoder(enc, X, trn, seed, M.EPOCHS, "cpu")
            enc.eval()
            for dep in depths:
                t1 = time.time()
                sub = DepthEncoder(enc, dep).eval()
                f = patch_feats(sub, X, tr_s, 8, 1)
                ed = effective_dim(f)
                bank, cnt, tot = gather_bank(sub, X, trn, M.BANK_SIZE, seed, "cpu")
                s, fb = score_partition(sub, X, dev, bank, "cpu")
                ns = s[y_dev == 0]
                pc = {NAMES[c]: _fast_auroc(
                        np.concatenate([ns, s[y_dev == c]]),
                        np.concatenate([np.zeros(len(ns), int),
                                        np.ones(int((y_dev == c).sum()), int)]))
                      for c in range(1, 9) if (y_dev == c).sum()}
                key = "%s_d%d_s%d" % (arm, dep, seed)
                rep[key] = {"arm": arm, "depth": dep, "seed": seed,
                            "feat_dim": int(f.shape[1]), "eff_dim": ed,
                            "aupr_blocked": aupr_blocked(s, isd), "auroc": auroc(s, isd),
                            "fpr_at_95tpr": fpr_at_tpr(s, isd, 0.95),
                            "n_unique": int(len(np.unique(s))),
                            "knn_mean": float(s.mean()), "per_class": pc,
                            "seconds": round(time.time() - t1, 1)}
                log("%-18s 차원 %2d  유효차원 %5.2f  AUPRblk %.4f  AUROC %.4f  거리평균 %.5f"
                    % (key, f.shape[1], ed, rep[key]["aupr_blocked"],
                       rep[key]["auroc"], rep[key]["knn_mean"]))
                (OUT / ("layer_depth_d%s.json" % a.depths.replace(",", ""))).write_text(
                    json.dumps(rep, ensure_ascii=False, indent=2))

    # 깊이 집합을 골라 돌릴 수 있으므로 **있는 깊이만** 표에 낸다.
    # (처음 판은 깊이 1과 3이 있다고 가정해서 `--depths 2` 만 돌렸을 때 터졌다.)
    print("\n== 반증 조건 1: 붕괴 비율 (학습 유효차원 / 무작위 유효차원) ==")
    print("%6s " % "seed" + " ".join(
        "%16s" % ("깊이%d (RF %dx%d)" % (d, 2 * d + 1, 2 * d + 1)) for d in depths))
    ratios = {}
    for seed in seeds:
        row = []
        for dep in depths:
            kt = "trained_d%d_s%d" % (dep, seed)
            kr = "random_d%d_s%d" % (dep, seed)
            if kt in rep and kr in rep:
                ratios[(seed, dep)] = rep[kt]["eff_dim"] / rep[kr]["eff_dim"]
                row.append("%16.4f" % ratios[(seed, dep)])
            else:
                row.append("%16s" % "-")
        print("%6d " % seed + " ".join(row))
    if len(depths) >= 2:
        lo, hi = min(depths), max(depths)
        ok = [ratios.get((s, lo), 0.0) > ratios.get((s, hi), 1.0) for s in seeds]
        verdict = "예 — 기작 주장 유지" if all(ok) else "아니오 — **기작 주장 기각**"
        print("  얕은 층(%d)이 깊은 층(%d)보다 덜 무너지나 — 세 seed 전부: %s"
              % (lo, hi, verdict))
    rep["_collapse_ratios"] = {"s%d_d%d" % (s, d): v for (s, d), v in ratios.items()}

    print("\n== 성능 (판정에 쓰지 않는다. §3 의 교란 때문) ==")
    print("%-18s %10s %10s %10s %10s" % ("arm", "유효차원", "AUPR블록", "AUROC", "FPR@95"))
    for k, v in rep.items():
        print("%-18s %10.2f %10.4f %10.4f %10.4f"
              % (k, v["eff_dim"], v["aupr_blocked"], v["auroc"], v["fpr_at_95tpr"]))
    (OUT / ("layer_depth_d%s.json" % a.depths.replace(",", ""))).write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log("저장 완료")
