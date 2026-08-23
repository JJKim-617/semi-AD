"""새는 none 웨이퍼는 라벨이 애매한 것인가 (18차).

## 물음

상한 분석이 남긴 것: **none/결함 이진을 완벽히 풀어야 0.8746** 이다.
현재 최선(TTA 27개)에서 none 110,701장 중 **900장이 결함으로 샌다.**
17차는 그 중 상당수가 "독립적인 자 둘에서 결함처럼 보인다" 고 적었다.

18차에 두 가지를 더 쟀다.
1. **모델들이 얼마나 확신하는가** — TTA 18개 풀에서 새는 1,057장 중 **280장(26.5%)을
   전원 일치로, 평균 확신도 0.857 로** 결함이라 부른다. 진짜 결함을 맞힌 것의
   확신도가 0.853 이니 **같은 수준이다.**
2. **입력 공간에서 라벨이 모순되는가** — 64x64 pad 표현에서 **완전히 같은 웨이퍼맵**인데
   라벨이 갈리는 것은 test 에 **14장(0.01%)뿐**이다. **모순 라벨로는 설명되지 않는다.**

여기서는 셋째를 잰다. **새는 웨이퍼가 학습 데이터의 어떤 이웃 사이에 앉아 있는가.**
모델을 쓰지 않는다 — 원본 격자의 해밍 거리만 쓴다(순환을 피한다).

**새는 웨이퍼의 최근접 학습 이웃이 결함 라벨을 달고 있다면**, 그 웨이퍼가
"none 이라 불리는 것이 이상한" 자리에 있다는 뜻이다.
**대조군은 새지 않은 none 웨이퍼**다 — 그것들의 이웃 분포와 비교해야 의미가 있다.
"""

from __future__ import annotations

import numpy as np


def hamming_topk(A: np.ndarray, B: np.ndarray, k: int, device: str = "cuda",
                 chunk: int = 256):
    """A 의 각 행에 대해 B 에서 가장 가까운 k개의 (거리, 색인).

    명목형 격자라 해밍 거리를 쓴다. 값이 {0,1,2} 이므로 클래스별 one-hot 내적의
    합이 '일치한 칸 수' 이고, 거리는 전체 칸 수에서 그것을 뺀 값이다.
    """
    import torch
    n, d = A.shape
    Bt = torch.from_numpy(B).to(device).long()
    Boh = torch.stack([(Bt == c).float() for c in (0, 1, 2)], 0)   # (3, m, d)
    out_d = np.empty((n, k), np.int32)
    out_i = np.empty((n, k), np.int64)
    for i in range(0, n, chunk):
        At = torch.from_numpy(A[i:i + chunk]).to(device).long()
        match = torch.zeros(At.shape[0], B.shape[0], device=device)
        for c in (0, 1, 2):
            match += (At == c).float() @ Boh[c].T
        dist = d - match
        v, idx = torch.topk(dist, k, dim=1, largest=False)
        out_d[i:i + At.shape[0]] = v.cpu().numpy().astype(np.int32)
        out_i[i:i + At.shape[0]] = idx.cpu().numpy()
    return out_d, out_i


def main():
    import argparse
    import glob
    import json
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _runtime import setup
    setup("eval")

    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64pad.npz")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-control", type=int, default=3000)
    p.add_argument("--out", default="result/cls_baseline/e24_label_neighbors.json")
    a = p.parse_args()

    d = np.load(a.cache)
    X, y = d["X"], d["y"].astype(np.int64)
    classes = [str(c) for c in d["classes"]]
    sp = np.load(a.splits)
    tr, te = sp["train"], sp["test"]

    # 현재 최선 풀(TTA 판, e20/e21/e22 포함 전부)의 예측으로 '새는 것' 을 정한다.
    P, ty = None, None
    for f in sorted(glob.glob("result/posthoc/*_tta_logits.npz")):
        z = np.load(f)
        ty = z["test_y"] if ty is None else ty
        lg = z["test_logits"].astype(np.float64)
        e = np.exp(lg - lg.max(1, keepdims=True))
        P = e / e.sum(1, keepdims=True) if P is None else P + e / e.sum(1, keepdims=True)
    pred = P.argmax(1)
    none_m = ty == 0
    leaked = np.where(none_m & (pred != 0))[0]
    kept = np.where(none_m & (pred == 0))[0]
    rng = np.random.default_rng(0)
    kept = rng.choice(kept, min(a.n_control, len(kept)), replace=False)
    print(f"새는 none {len(leaked):,}장, 대조군(안 새는 none) {len(kept):,}장, "
          f"학습 {len(tr):,}장")

    Xtr = X[tr].reshape(len(tr), -1)
    ytr = y[tr]

    def profile(idx_test, name):
        A = X[te[idx_test]].reshape(len(idx_test), -1)
        dist, nb = hamming_topk(A, Xtr, a.k)
        lab = ytr[nb]                                   # (n, k)
        defect_frac = (lab != 0).mean(1)
        print(f"\n## {name} ({len(idx_test):,}장)")
        print(f"  최근접 이웃까지 해밍 거리  중앙값 {np.median(dist[:, 0]):.0f}  "
              f"평균 {dist[:, 0].mean():.1f}")
        print(f"  최근접 {a.k}개 중 결함 라벨 비율  평균 {defect_frac.mean():.3f}")
        for thr in (0.0, 0.3, 0.5, 0.8):
            print(f"    이웃의 {thr:.0%} 초과가 결함인 웨이퍼: "
                  f"{(defect_frac > thr).mean():.1%}")
        top1_def = (lab[:, 0] != 0).mean()
        print(f"  **최근접 1개가 결함 라벨인 비율: {top1_def:.1%}**")
        # 거리가 멀면 이웃 논증이 약하다. 거리 구간별로 나눠 본다.
        print(f"  {'최근접 거리':<14}{'장수':>7}{'최근접이 결함':>14}")
        buckets = [(0, 50), (50, 100), (100, 150), (150, 250), (250, 10 ** 9)]
        by_dist = {}
        for lo, hi in buckets:
            m = (dist[:, 0] >= lo) & (dist[:, 0] < hi)
            if m.sum() == 0:
                continue
            v = float((lab[m, 0] != 0).mean())
            by_dist[f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"] = dict(
                n=int(m.sum()), top1_defect=v)
            print(f"  {str(lo) + '~' + (str(hi) if hi < 10 ** 9 else '') :<14}"
                  f"{int(m.sum()):>7}{v:>14.1%}")
        return dict(n=len(idx_test), d1_median=float(np.median(dist[:, 0])),
                    d1_mean=float(dist[:, 0].mean()),
                    defect_frac_mean=float(defect_frac.mean()),
                    top1_defect=float(top1_def), by_distance=by_dist)

    r_leak = profile(leaked, "새는 none 웨이퍼")
    r_keep = profile(kept, "대조군 — 안 새는 none 웨이퍼")

    # **교란을 잡는다.** 거리 구간별 표에서 두 무리 모두 거리가 멀수록 결함 이웃이
    # 늘어난다(대조군도 250칸 이상에서 85%). 해밍 거리는 '다른 칸 수' 이고 그것은
    # **불량 다이 개수**에 끌려간다. 그리고 불량이 많은 웨이퍼는 결함 라벨을 단다.
    # 즉 위 비교는 "새는 웨이퍼가 불량 다이가 많다" 를 다시 말한 것일 수 있다.
    # 불량 다이 개수를 맞춰 다시 비교한다.
    print("\n## 교란 점검 — 불량 다이 개수를 맞추면 남는가")
    nf_all = (X[te] == 2).reshape(len(te), -1).sum(1)
    def top1(idx_test):
        A = X[te[idx_test]].reshape(len(idx_test), -1)
        dist, nb = hamming_topk(A, Xtr, 1)
        return (ytr[nb][:, 0] != 0).astype(float), dist[:, 0]
    lk_v, lk_d = top1(leaked)
    kp_v, kp_d = top1(kept)
    lk_n, kp_n = nf_all[leaked], nf_all[kept]
    edges = np.quantile(lk_n, [0, 0.25, 0.5, 0.75, 1.0])
    edges[-1] += 1
    print(f"  {'불량 다이 수':<16}{'새는 것 n':>10}{'결함이웃':>10}"
          f"{'대조군 n':>10}{'결함이웃':>10}{'차이':>9}")
    strat = {}
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        ml = (lk_n >= lo) & (lk_n < hi)
        mk = (kp_n >= lo) & (kp_n < hi)
        if ml.sum() < 5 or mk.sum() < 5:
            continue
        a_, b_ = float(lk_v[ml].mean()), float(kp_v[mk].mean())
        strat[f"{int(lo)}-{int(hi)}"] = dict(n_leak=int(ml.sum()), leak=a_,
                                             n_keep=int(mk.sum()), keep=b_)
        print(f"  {str(int(lo)) + '~' + str(int(hi)):<16}{int(ml.sum()):>10}"
              f"{a_:>10.1%}{int(mk.sum()):>10}{b_:>10.1%}{a_ - b_:>+9.1%}")
    if strat:
        dm = float(np.mean([v["leak"] - v["keep"] for v in strat.values()]))
        print(f"  **층 평균 차이 {dm:+.1%}p** (층을 안 나눴을 때 "
              f"{r_leak['top1_defect'] - r_keep['top1_defect']:+.1%}p)")
        print("  -> " + ("불량 다이 수를 맞춰도 차이가 남는다."
                         if dm > 0.10 else
                         "**불량 다이 수를 맞추면 차이가 크게 준다 — 교란이었다.**"))

    base = float((ytr != 0).mean())
    print("\n## 읽는 법 — **기저율과 견줘야 한다**")
    print(f"  학습 세트의 결함 기저율 {base:.1%} (none 을 덜어낸 분할이라 test 와 다르다)")
    print(f"  최근접이 결함일 확률: 새는 것 {r_leak['top1_defect']:.1%} "
          f"({r_leak['top1_defect'] / base:.2f}배)  대조군 {r_keep['top1_defect']:.1%} "
          f"({r_keep['top1_defect'] / base:.2f}배)")
    print(f"  차이 {r_leak['top1_defect'] - r_keep['top1_defect']:+.1%}p")
    print("\n  **해석의 한계를 먼저 적는다.**")
    print(f"  최근접 이웃까지 거리가 중앙값 {r_leak['d1_median']:.0f}칸이다. "
          "다이 칸이 보통 600~1,500개이므로")
    print("  '가장 가까운 이웃' 이라도 다이의 10~25% 가 다르다. "
          "**같은 웨이퍼의 이웃이 아니다.**")
    print("  그리고 이웃의 라벨이 곧 정답은 아니다. "
          "이 표는 '애매하다' 를 증명하지 못하고,")
    print("  **'새는 웨이퍼가 결함 라벨이 밀집한 자리에 앉아 있다' 까지만 말한다.**")

    Path(a.out).write_text(json.dumps(
        dict(leaked=r_leak, kept=r_keep, stratified=strat),
        ensure_ascii=False, indent=2))
    print(f"\n[저장] {a.out}")


if __name__ == "__main__":
    main()
