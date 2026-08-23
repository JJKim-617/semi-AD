"""학습량 곡선 — epoch 수만 바꿔 O2 를 다시 잰다.

반증 조건은 `candidate/ood_o2_training_amount.md` 에 실행 전에 박았다.
**여기서 나온 어떤 epoch 값도 채택하지 않는다** — 산출물은 곡선의 모양 하나다.
`test_dev` 만 쓴다.
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

from a21_ood_metrics import aupr_blocked, auroc, fpr_at_tpr, operating_points  # noqa: E402
from a22_ood_template import assert_one_class  # noqa: E402
from a23_ood_template_eval import NAMES, _fast_auroc  # noqa: E402
from a38_sealed_holdout import assert_not_sealed, load_partition  # noqa: E402
import a42_ood_ssl_knn as M  # noqa: E402
from a43_ood_ssl_eval import gather_bank, score_partition, train_encoder  # noqa: E402
from diag_o2_feature_collapse import effective_dim, patch_feats  # noqa: E402

OUT = Path("result/ood/o2_ssl_knn")
T0 = time.time()


def log(m):
    print("[%7.1fs] %s" % (time.time() - T0, m), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", default="0,1,3,6,12")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=28)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    eps = [int(e) for e in a.epochs.split(",")]

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
    for ep in eps:
        t1 = time.time()
        torch.manual_seed(a.seed)
        enc = M.PatchEncoder()
        hist = train_encoder(enc, X, trn, a.seed, ep, "cpu") if ep > 0 else []
        enc.eval()
        f = patch_feats(enc, X, tr_s, 8, 1)
        ed = effective_dim(f)
        bank, cnt, tot = gather_bank(enc, X, trn, M.BANK_SIZE, a.seed, "cpu")
        s, fb = score_partition(enc, X, dev, bank, "cpu")
        pc = {}
        ns = s[y_dev == 0]
        for c in range(1, 9):
            m = y_dev == c
            if m.sum():
                pc[NAMES[c]] = _fast_auroc(
                    np.concatenate([ns, s[m]]),
                    np.concatenate([np.zeros(len(ns), int), np.ones(int(m.sum()), int)]))
        rep["ep%d" % ep] = {
            "epochs": ep, "final_masked_ce": hist[-1]["masked_ce"] if hist else None,
            "aupr_blocked": aupr_blocked(s, isd), "auroc": auroc(s, isd),
            "fpr_at_95tpr": fpr_at_tpr(s, isd, 0.95), "n_unique": int(len(np.unique(s))),
            "knn_mean": float(s.mean()), "knn_std": float(s.std()),
            "eff_dim": ed, "per_class": pc, "history": hist,
            "operating": operating_points(s, isd, (0.5, 0.8, 0.95)),
            "seconds": round(time.time() - t1, 1),
        }
        r = rep["ep%d" % ep]
        log("ep=%2d  AUPRblk %.4f  AUROC %.4f  FPR@95 %.4f  유효차원 %5.2f  거리평균 %.5f  CE %s"
            % (ep, r["aupr_blocked"], r["auroc"], r["fpr_at_95tpr"], ed, r["knn_mean"],
               "%.4f" % r["final_masked_ce"] if r["final_masked_ce"] else "-"))
        (OUT / "epoch_curve.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))

    print("\n%6s %8s %10s %9s %9s %10s %11s %9s"
          % ("epoch", "maskedCE", "AUPR블록", "AUROC", "FPR@95", "유효차원", "kNN거리평균", "고유값"))
    for ep in eps:
        r = rep["ep%d" % ep]
        print("%6d %8s %10.4f %9.4f %9.4f %10.2f %11.5f %9d"
              % (ep, "%.4f" % r["final_masked_ce"] if r["final_masked_ce"] else "-",
                 r["aupr_blocked"], r["auroc"], r["fpr_at_95tpr"], r["eff_dim"],
                 r["knn_mean"], r["n_unique"]))
    print("\n%6s " % "epoch" + " ".join("%10s" % c for c in NAMES[1:]))
    for ep in eps:
        r = rep["ep%d" % ep]
        print("%6d " % ep + " ".join("%10.4f" % r["per_class"][c] for c in NAMES[1:]))
    log("저장 완료 → %s" % (OUT / "epoch_curve.json"))
