"""O2 실행 — 자기지도 인코더 + kNN 을 `test_dev` 에서 잰다.

반증 조건 8개와 실행 설계는 `docs/experiments/candidate/ood_o2_ssl_knn.md` §5, §10 에
**한 줄도 실행하기 전에** 박혀 있다. 이 스크립트는 그것을 그대로 수행한다.

## 반드시 같이 도는 것 — 무작위 초기화 대조군

**학습 arm 만 돌리면 판정이 무효다**(§10.5). "자기지도 학습이 이겼다" 와
"임의의 비선형 필터뱅크 + kNN 이면 아무거나 이긴다" 를 못 가르기 때문이다.
그래서 seed 마다 **학습 arm 과 무작위 초기화 arm 을 짝지어** 돌린다.

## 봉인

**`test_dev` 만 쓴다.** `load_partition("test_sealed")` 는 실패한다. 실패하는 게 정상이다.
적합 자료(train-none)에 봉인이 섞이지 않았는지 `assert_not_sealed` 로 확인한다.
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

setup("feat")                       # GPU 0 고정 + 프로세스 이름 중립화. torch import 보다 먼저.

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from a21_ood_metrics import aupr, aupr_blocked, auroc, fpr_at_tpr, operating_points  # noqa: E402
from a22_ood_template import assert_one_class  # noqa: E402
from a23_ood_template_eval import (  # noqa: E402
    BIN_LBL, BINS, NAMES, _fast_auroc, bootstrap_auroc_ci, ecdf_percentile, fisher,
    paired_aupr_blocked_diff_ci,
)
from a24_ood_residual import local_fail_count_map, local_fail_count_max  # noqa: E402
from a27_line_filter import line_density_max  # noqa: E402
from a29_radial_calibration import band_index, calibrate_map, fit_band_reference  # noqa: E402
from a38_sealed_holdout import assert_not_sealed, load_partition  # noqa: E402
import a42_ood_ssl_knn as M  # noqa: E402

OUT = Path("result/ood/o2_ssl_knn")
CHUNK_ENC = 512
T0 = time.time()


def log(m):
    print("[%7.1fs] %s" % (time.time() - T0, m), flush=True)


# --- 특징 ------------------------------------------------------------------------

def encode_chunk(enc, xb, device):
    with torch.no_grad():
        h = M.to_onehot(xb).to(device)
        return enc(h).permute(0, 2, 3, 1).cpu().numpy()      # (B, H, W, C)


def valid_counts(X, idx, k):
    out = np.empty(len(idx), np.int64)
    for a in range(0, len(idx), 4096):
        b = idx[a:a + 4096]
        out[a:a + len(b)] = M.valid_window_mask(X[b], k).reshape(len(b), -1).sum(1)
    return out


def gather_bank(enc, X, idx, n, seed, device):
    """**두 번 훑어 정확히 뽑는다.** 전체 유효 위치를 메모리에 올리지 않는다.

    1) 웨이퍼별 유효 위치 개수를 센다(인코더 없이, 정수 연산).
    2) 전역 유효 위치 중 n 개를 균등하게 고른다.
    3) 그 위치가 든 웨이퍼만 인코딩해 해당 벡터만 꺼낸다.

    이러면 "train-none 의 유효 위치에서 무작위 n 개" 라는 §10.4 의 정의가
    근사가 아니라 **정확히** 지켜진다.
    """
    cnt = valid_counts(X, idx, M.RF)
    tot = int(cnt.sum())
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(tot, min(n, tot), replace=False))
    starts = np.concatenate([[0], np.cumsum(cnt)])
    wafer_of = np.searchsorted(starts, take, side="right") - 1
    local = take - starts[wafer_of]
    feats = np.empty((len(take), M.FEAT_DIM), np.float32)
    for w in np.unique(wafer_of):
        sel = wafer_of == w
        f = encode_chunk(enc, X[idx[w:w + 1]], device)[0]
        v = M.valid_window_mask(X[idx[w:w + 1]], M.RF)[0]
        feats[sel] = f[v][local[sel]]
    return feats, cnt, tot


def score_partition(enc, X, idx, bank, device, k=M.KNN_K):
    """웨이퍼별 점수 = 유효 위치의 kNN 거리 max. 유효 위치가 없으면 전체 max(§10.2)."""
    out = np.empty(len(idx), np.float64)
    n_fallback = 0
    for a in range(0, len(idx), CHUNK_ENC):
        b = idx[a:a + CHUNK_ENC]
        xb = X[b]
        f = encode_chunk(enc, xb, device)
        v = M.valid_window_mask(xb, M.RF)
        q, owner, fb = [], [], []
        for i in range(len(b)):
            if v[i].any():
                sel = f[i][v[i]]
            else:
                sel = f[i].reshape(-1, M.FEAT_DIM)
                fb.append(i)
            q.append(sel)
            owner.append(np.full(len(sel), i))
        n_fallback += len(fb)
        q = np.concatenate(q)
        owner = np.concatenate(owner)
        d = M.knn_mean_distance(q, bank, k=k, device=device)
        o = np.full(len(b), -np.inf)
        np.maximum.at(o, owner, d)
        out[a:a + len(b)] = o
    return out, n_fallback


# --- 학습 -------------------------------------------------------------------------

def train_encoder(enc, X, idx, seed, epochs, device, batch=M.BATCH, lr=M.LR):
    """masked die prediction. **train-none 만.** 모든 난수를 시드에 묶는다(§10.9)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = M.MaskedHead(enc).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    hist = []
    for ep in range(1, epochs + 1):
        for g in opt.param_groups:
            g["lr"] = lr * (0.5 * (1 + np.cos(np.pi * (ep - 1) / epochs)))
        head.train()
        perm = rng.permutation(len(idx))
        tot, seen = 0.0, 0
        for i in range(0, len(idx), batch):
            b = idx[perm[i:i + batch]]
            xb = M.dihedral(X[b], int(rng.integers(4)), bool(rng.integers(2)))
            tgt = torch.as_tensor(xb.astype(np.int64), device=device)
            valid = torch.as_tensor(M.valid_window_mask(xb, M.RF), device=device)
            mk = M.random_patch_mask(xb.shape[1:], M.MASK_PATCH, M.MASK_RATIO, rng)
            inp = M.to_onehot(xb).to(device)
            inp[:, :, mk] = 0.0
            mask = torch.as_tensor(mk, device=device).expand(len(b), -1, -1)
            loss = M.masked_valid_ce(head(inp), tgt, mask, valid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
            seen += len(b)
        hist.append({"epoch": ep, "masked_ce": tot / seen})
        log("    ep%2d  masked CE %.4f" % (ep, tot / seen))
    return hist


# --- 지표 -------------------------------------------------------------------------

def blocked_ci(s, l, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    v = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(l), len(l))
        v[b] = aupr_blocked(s[i], l[i])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def per_class(score, yy, ci_for=("Scratch", "Center", "Loc")):
    ns = score[yy == 0]
    z = np.zeros(len(ns), np.int64)
    r = {}
    for c in range(1, 9):
        m = yy == c
        if not m.sum():
            continue
        s = np.concatenate([ns, score[m]])
        l = np.concatenate([z, np.ones(int(m.sum()), np.int64)])
        row = {"n": int(m.sum()), "auroc": _fast_auroc(s, l)}
        if NAMES[c] in ci_for:
            row["auroc_ci"] = bootstrap_auroc_ci(s, l, n_boot=200, seed=c)
        r[NAMES[c]] = row
    return r


def per_size(score, isd, size):
    o = {}
    for i, lbl in enumerate(BIN_LBL):
        m = (size >= BINS[i]) & (size < BINS[i + 1])
        if m.sum() and isd[m].sum() not in (0, m.sum()):
            o[lbl] = aupr_blocked(score[m], isd[m])
    return o


def full_metrics(score, yy, size, si=0):
    isd = (yy != 0).astype(np.int64)
    return {
        "auroc": auroc(score, isd), "aupr_blocked": aupr_blocked(score, isd),
        "aupr_order": aupr(score, isd), "fpr_at_95tpr": fpr_at_tpr(score, isd, 0.95),
        "n_unique": int(len(np.unique(score))),
        "aupr_blocked_ci95": blocked_ci(score, isd, 300, 9000 + si),
        "per_class": per_class(score, yy), "per_size_bin": per_size(score, isd, size),
        "operating": operating_points(score, isd, (0.5, 0.8, 0.95)),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=M.EPOCHS)
    p.add_argument("--bank", type=int, default=M.BANK_SIZE)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=32)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    seeds = [int(s) for s in a.seeds.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    sp = np.load("data/wm811k/cache/splits_v1.npz")
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    size = d["die_size"].astype(np.float64)
    X = d["X"]
    trn = sp["train"][y[sp["train"]] == 0]
    assert_one_class(y[trn])                 # 라벨 가드
    assert_not_sealed(trn, "train-none")     # 봉인 가드
    dev = load_partition("test_dev")
    assert_not_sealed(dev, "test_dev")
    y_dev, size_dev = y[dev], size[dev]
    log("train-none %d / test_dev %d (결함 %d)" % (len(trn), len(dev), int((y_dev != 0).sum())))

    # 기준선 fuse3 — 같은 파티션에서 다시 잰다(저장된 전체 test 수치를 쓰지 않는다)
    pad_tr = np.ascontiguousarray(X[trn])
    pad_dev = np.ascontiguousarray(X[dev])
    v_tr = local_fail_count_map(pad_tr, k=5, chunk=4000).astype(np.float32)
    ref = fit_band_reference(v_tr, pad_tr, y=y[trn], n_bands=32, seed=0)
    del v_tr

    def a_sc(x):
        o = np.empty(len(x))
        for i in range(0, len(x), 4000):
            xb = x[i:i + 4000]
            u = calibrate_map(local_fail_count_map(xb, k=5, chunk=4000),
                              band_index(xb, 32), ref)
            o[i:i + len(xb)] = u.reshape(len(xb), -1).max(1)
        return o

    def comps(x):
        return {"A": a_sc(x),
                "B": line_density_max(x, length=11, n_orient=8, chunk=4000, min_dies=11),
                "C": local_fail_count_max(x, k=3, chunk=4000)}

    c_tr, c_dev = comps(pad_tr), comps(pad_dev)
    fuse3 = fisher([ecdf_percentile(c_tr[k], c_dev[k]) for k in ("A", "B", "C")])
    log("기준선 fuse3 준비 완료  blocked AUPR %.4f" % aupr_blocked(fuse3, (y_dev != 0).astype(int)))

    scores, meta = {"fuse3": fuse3}, {}
    for seed in seeds:
        for arm in ("random", "trained"):
            tag = "O2_%s_s%d" % (arm, seed)
            t1 = time.time()
            torch.manual_seed(seed)
            enc = M.PatchEncoder().to(a.device)
            if arm == "trained":
                log("%s 학습 시작" % tag)
                hist = train_encoder(enc, X, trn, seed, a.epochs, a.device)
            else:
                hist = None
            enc.eval()
            bank, cnt, tot = gather_bank(enc, X, trn, a.bank, seed, a.device)
            s_tr, fb_tr = score_partition(enc, X, trn, bank, a.device)
            s_dev, fb_dev = score_partition(enc, X, dev, bank, a.device)
            scores[tag] = s_dev
            scores[tag + "_fuse"] = fisher([
                ecdf_percentile(s_tr, s_dev),
                ecdf_percentile(c_tr["A"], c_dev["A"]),
                ecdf_percentile(c_tr["B"], c_dev["B"]),
                ecdf_percentile(c_tr["C"], c_dev["C"])])
            meta[tag] = {"history": hist, "bank": int(len(bank)),
                         "valid_total_train": tot,
                         "wafers_without_valid_window_train": int((cnt == 0).sum()),
                         "fallback_train": fb_tr, "fallback_dev": fb_dev,
                         "seconds": round(time.time() - t1, 1)}
            log("%s 완료 (%.0fs)  blocked AUPR %.4f"
                % (tag, time.time() - t1, aupr_blocked(s_dev, (y_dev != 0).astype(int))))
            np.savez_compressed(OUT / "o2_scores.npz", y_dev=y_dev, size_dev=size_dev,
                                **{k: v.astype(np.float64) for k, v in scores.items()})

    rep = {"meta": meta, "metrics": {}}
    for si, (k, s) in enumerate(scores.items()):
        rep["metrics"][k] = full_metrics(s, y_dev, size_dev, si)
        m = rep["metrics"][k]
        log("%-20s AUROC %.4f  AUPRblk %.4f  FPR@95 %.4f  고유값 %d"
            % (k, m["auroc"], m["aupr_blocked"], m["fpr_at_95tpr"], m["n_unique"]))

    isd = (y_dev != 0).astype(np.int64)
    rep["paired"] = {}
    for seed in seeds:
        for pair in ((f"O2_trained_s{seed}", "fuse3"),
                     (f"O2_trained_s{seed}", f"O2_random_s{seed}"),
                     (f"O2_trained_s{seed}_fuse", "fuse3")):
            if pair[0] not in scores:
                continue
            lo, hi, pv = paired_aupr_blocked_diff_ci(scores[pair[0]], scores[pair[1]],
                                                     isd, 300, 31 + seed)
            rep["paired"]["%s vs %s" % pair] = {"lo": lo, "hi": hi, "p": pv}
            log("  %-40s [%+.4f, %+.4f] p=%.4f  %s"
                % ("%s vs %s" % pair, lo, hi, pv,
                   "이김" if lo > 0 else ("짐" if hi < 0 else "무승부")))

    (OUT / "o2_metrics.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log("저장 완료 → %s" % OUT)
