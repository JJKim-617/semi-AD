"""train/val/test 분포 이동 실측.

1) 클래스 분포 per split (splits_v1.npz 기준)
2) 원본 웨이퍼맵 크기(H,W)와 dieSize 분포 per split, per class (LSWMD.pkl 직접 로딩)
3) 캐시 64x64 기준 클래스별 시각 통계(불량 die 비율, 공간 분산, 엣지 집중도)의
   train vs test 차이
"""
import sys
from pathlib import Path

import numpy as np

CLASSES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Loc", "Random", "Scratch", "Near-full"]


def img_stats(X_chunk):
    """X: (n,64,64) uint8 {0,1,2} -> per-image 통계."""
    W = X_chunk > 0
    D = X_chunk == 2
    n = X_chunk.shape[0]
    wafer_px = W.reshape(n, -1).sum(1).astype(np.float64)
    def_px = D.reshape(n, -1).sum(1).astype(np.float64)
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float64)
    cy, cx = 31.5, 31.5
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)  # 중심 거리
    Df = D.reshape(n, -1).astype(np.float64)
    rf = r.reshape(-1)
    # 결함 픽셀의 평균 반경 (이미지 중심 기준, /32 정규화)
    sum_r = Df @ rf
    mean_r = np.where(def_px > 0, sum_r / np.maximum(def_px, 1), np.nan) / 32.0
    # 결함 픽셀 자체 centroid 로부터의 RMS 산포 (/32)
    sx = Df @ xx.reshape(-1)
    sy = Df @ yy.reshape(-1)
    sx2 = Df @ (xx.reshape(-1) ** 2)
    sy2 = Df @ (yy.reshape(-1) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = sx / def_px, sy / def_px
        var = (sx2 + sy2) / def_px - (mx ** 2 + my ** 2)
    spread = np.sqrt(np.maximum(var, 0)) / 32.0
    spread[def_px == 0] = np.nan
    # 엣지 집중도: 결함 픽셀 중 r > 0.8 * (해당 이미지 웨이퍼 최대반경) 비율
    Wf = W.reshape(n, -1)
    wafer_rmax = np.where(Wf, rf[None, :], 0).max(1)
    edge_mask = Df * (rf[None, :] > 0.8 * wafer_rmax[:, None])
    edge_frac = np.where(def_px > 0, edge_mask.sum(1) / np.maximum(def_px, 1), np.nan)
    def_frac = np.where(wafer_px > 0, def_px / np.maximum(wafer_px, 1), 0.0)
    return def_px, def_frac, mean_r, spread, edge_frac


def summarize(name, arrs, idx, y, cls_i):
    sel = idx[y[idx] == cls_i]
    out = []
    for a in arrs:
        v = a[sel]
        v = v[~np.isnan(v)]
        out.append((float(np.mean(v)) if len(v) else float("nan"),
                    float(np.median(v)) if len(v) else float("nan")))
    return len(sel), out


def main():
    d = np.load("data/wm811k/cache/wm811k_64.npz", allow_pickle=False)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    X, y = d["X"], d["y"].astype(np.int64)
    split = d["split"]
    tr, va, te = sp["train"], sp["val"], sp["test"]

    print("== 1) 클래스 분포 per split ==")
    print(f"{'class':10s} {'train':>8s} {'%':>6s} {'val':>8s} {'%':>6s} {'test':>8s} {'%':>6s} {'test/train 비':>10s}")
    for i, c in enumerate(CLASSES):
        nt = int((y[tr] == i).sum()); nv = int((y[va] == i).sum()); ns = int((y[te] == i).sum())
        pt, pv, ps = nt / len(tr) * 100, nv / len(va) * 100, ns / len(te) * 100
        ratio = (ps / pt) if pt > 0 else float("inf")
        print(f"{c:10s} {nt:8d} {pt:6.2f} {nv:8d} {pv:6.2f} {ns:8d} {ps:6.2f} {ratio:10.2f}")
    print(f"{'total':10s} {len(tr):8d} {'':6s} {len(va):8d} {'':6s} {len(te):8d}")

    print("\n== 2) 원본 크기 / dieSize (LSWMD.pkl) ==", flush=True)
    import pandas as pd
    df = pd.read_pickle("data/wm811k/LSWMD.pkl")
    sys.path.insert(0, "src")
    from a1_preprocess_wm811k import unwrap
    ft = df["failureType"].map(unwrap)
    labeled = ft.isin(CLASSES).to_numpy()
    df_l = df[labeled].reset_index(drop=True)
    assert len(df_l) == len(y), (len(df_l), len(y))
    y_pkl = df_l["failureType"].map(unwrap).map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    assert (y_pkl == y).all(), "캐시와 pkl 라벨 순서 불일치"
    dims = np.array([wm.shape for wm in df_l["waferMap"]])
    H, Wd = dims[:, 0], dims[:, 1]
    die = df_l["dieSize"].to_numpy().astype(np.float64)
    minhw = np.minimum(H, Wd)
    up64 = (minhw < 64)

    for nm, idx in (("train", tr), ("val", va), ("test", te)):
        print(f"\n[{nm}] n={len(idx)}  min(H,W) 분위수 "
              f"p10={np.percentile(minhw[idx],10):.0f} p50={np.percentile(minhw[idx],50):.0f} "
              f"p90={np.percentile(minhw[idx],90):.0f}  "
              f"dieSize p50={np.percentile(die[idx],50):.0f}  "
              f"64미만(업샘플) 비율={up64[idx].mean()*100:.1f}%")
        # 상위 크기 조합
        from collections import Counter
        cnt = Counter(map(tuple, dims[idx]))
        top = cnt.most_common(6)
        print("   top dims: " + ", ".join(f"{t}x{u} {v} ({v/len(idx)*100:.0f}%)" for (t, u), v in top))

    print(f"\n-- 클래스별 median dieSize, median min(H,W) (train | test) --")
    print(f"{'class':10s} {'die tr':>8s} {'die te':>8s} {'hw tr':>6s} {'hw te':>6s} {'up64 tr%':>8s} {'up64 te%':>8s}")
    for i, c in enumerate(CLASSES):
        st = tr[y[tr] == i]; se = te[y[te] == i]
        if len(st) == 0 or len(se) == 0:
            continue
        print(f"{c:10s} {np.median(die[st]):8.0f} {np.median(die[se]):8.0f} "
              f"{np.median(minhw[st]):6.0f} {np.median(minhw[se]):6.0f} "
              f"{up64[st].mean()*100:8.1f} {up64[se].mean()*100:8.1f}")

    print("\n== 3) 캐시 64x64 시각 통계 (클래스별 train vs test) ==", flush=True)
    N = len(X)
    def_px = np.empty(N); def_frac = np.empty(N)
    mean_r = np.empty(N); spread = np.empty(N); edge_frac = np.empty(N)
    for s in range(0, N, 20000):
        e = min(s + 20000, N)
        a, b, c_, dd, ee = img_stats(X[s:e])
        def_px[s:e], def_frac[s:e], mean_r[s:e], spread[s:e], edge_frac[s:e] = a, b, c_, dd, ee
        print(f"   ...{e}/{N}", flush=True)

    arrs = [def_px, def_frac, mean_r, spread, edge_frac]
    names = ["def_px", "def_frac", "mean_r", "spread", "edge_frac"]
    print(f"\n{'class':10s} {'split':6s} {'n':>7s} " + "".join(f"{nm+'_mu':>10s}{nm+'_md':>10s}" for nm in names))
    for i, c in enumerate(CLASSES):
        for nm, idx in (("train", tr), ("test", te)):
            n, out = summarize(nm, arrs, idx, y, i)
            row = f"{c:10s} {nm:6s} {n:7d} "
            for mu, md in out:
                row += f"{mu:10.3f}{md:10.3f}"
            print(row)

    # 효과 크기: train vs test 평균 차 / pooled std
    print(f"\n-- train vs test 효과 크기 (|mean diff| / pooled std) --")
    print(f"{'class':10s} " + "".join(f"{nm:>10s}" for nm in names))
    for i, c in enumerate(CLASSES):
        st = tr[y[tr] == i]; se = te[y[te] == i]
        row = f"{c:10s} "
        for a in arrs:
            v1 = a[st]; v1 = v1[~np.isnan(v1)]
            v2 = a[se]; v2 = v2[~np.isnan(v2)]
            if len(v1) < 2 or len(v2) < 2:
                row += f"{'--':>10s}"; continue
            ps = np.sqrt((v1.var() + v2.var()) / 2)
            es = abs(v1.mean() - v2.mean()) / ps if ps > 0 else 0
            row += f"{es:10.2f}"
        print(row)

    # 저장: per-split per-class 통계 npz (재분석용)
    np.savez_compressed("docs/research/e4_diagnosis/evidence/img_stats.npz",
                        def_px=def_px, def_frac=def_frac, mean_r=mean_r,
                        spread=spread, edge_frac=edge_frac,
                        minhw=minhw, die=die, y=y)
    print("\n[save] docs/research/e4_diagnosis/evidence/img_stats.npz")


if __name__ == "__main__":
    main()
