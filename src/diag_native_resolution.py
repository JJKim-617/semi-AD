"""입력 해상도가 정보량 문제인지 확인한다 (E21 진단, 학습 없음).

## 왜 이걸 먼저 재는가

Scratch(F1 0.710)는 가는 선이고 64x64 가 그걸 뭉갤 수 있다는 가설이 있다.
그런데 반론이 있다 — **원본이 대부분 26x26 대라 상향 리사이즈는 정보를 만들지 못한다.**
둘 다 맞을 수 있다. 갈리는 지점은 하나다.

`pad_center` 는 원본이 64 보다 크면 **nearest 로 줄인 뒤** 배치한다. 그 웨이퍼들만이
해상도로 정보를 잃는다. 나머지는 그냥 여백에 놓일 뿐이라 96 으로 키워도 새 정보가 없다.

따라서 답해야 할 것은 하나다. **test 의 몇 %가 실제로 축소를 겪고, 그때 얼마를 잃는가.**
결함 클래스별로 본다 — 병목인 Scratch, Edge-Loc 이 큰 웨이퍼에 몰려 있으면
해상도는 정보량 문제이고, 아니면 해상도 축은 애초에 닫혀 있다.

학습이 없다. 원본 pkl 의 웨이퍼맵 크기만 센다.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

OUT = Path("docs/research/input_resolution/evidence")
NAMES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
         "Random", "Scratch", "Near-full"]


def downsampled_loss(wm: np.ndarray, size: int) -> tuple[int, int]:
    """`pad_center` 와 같은 축소를 걸었을 때 (원래 다이 수, 남은 다이 수)."""
    h, w = wm.shape
    n0 = int((wm > 0).sum())
    if h <= size and w <= size:
        return n0, n0
    scale = min(size / h, size / w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    rows = (np.arange(nh) * h // nh).clip(0, h - 1)
    cols = (np.arange(nw) * w // nw).clip(0, w - 1)
    return n0, int((wm[rows[:, None], cols] > 0).sum())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pkl = "data/wm811k/LSWMD.pkl"
    df = pd.read_pickle(pkl)
    print(f"[load] {len(df):,} 행", flush=True)

    sp = np.load("data/wm811k/cache/splits_v1.npz")
    d = np.load("data/wm811k/cache/wm811k_64pad.npz")
    y_all = d["y"].astype(np.int64)
    # 캐시는 라벨이 있는 172,950 장만 담고 원본 순서를 유지한다.
    from a1_preprocess_wm811k import CLASSES, unwrap
    labels = df["failureType"].map(unwrap)
    lab = df[labels.isin(CLASSES)].reset_index(drop=True)
    assert len(lab) == len(y_all), (len(lab), len(y_all))

    te = sp["test"]
    rows = []
    for i in te:
        wm = np.asarray(lab["waferMap"].iloc[int(i)])
        h, w = wm.shape
        n0, n1 = downsampled_loss(wm, 64)
        rows.append((int(y_all[i]), h, w, n0, n1))
    a = np.array(rows, dtype=np.int64)
    cls, H, W, N0, N1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    big = (H > 64) | (W > 64)

    print("\n== test 원본 크기 ==")
    print(f"  n = {len(a):,}")
    print(f"  H 중앙값 {np.median(H):.0f}  W 중앙값 {np.median(W):.0f}")
    print(f"  최빈 (H,W): " + ", ".join(
        "%dx%d %d장" % (k[0], k[1], v)
        for k, v in Counter(map(tuple, a[:, 1:3].tolist())).most_common(5)))
    print(f"  **64 를 넘는 웨이퍼 {big.sum():,}장 ({big.mean() * 100:.1f}%)** "
          f"— 이것들만 pad64 에서 축소를 겪는다")
    print(f"  96 을 넘는 웨이퍼 {((H > 96) | (W > 96)).sum():,}장 "
          f"({((H > 96) | (W > 96)).mean() * 100:.1f}%)")

    print("\n== 축소되는 웨이퍼에서 실제로 잃는 다이 ==")
    if big.any():
        keep = N1[big] / np.maximum(N0[big], 1)
        print(f"  남는 다이 비율 중앙값 {np.median(keep):.3f}  "
              f"(하위 10% {np.quantile(keep, 0.1):.3f})")
        print(f"  잃는 다이 총합 {int((N0[big] - N1[big]).sum()):,} / {int(N0[big].sum()):,}")
    else:
        print("  없음")

    print("\n== 클래스별 ==")
    print("%-11s %8s %10s %10s %10s %10s" % (
        "클래스", "support", ">64 비율", "H 중앙값", "W 중앙값", "남는다이"))
    per_class = {}
    for c, name in enumerate(NAMES):
        m = cls == c
        if not m.any():
            continue
        b = big & m
        keep = float(np.median(N1[b] / np.maximum(N0[b], 1))) if b.any() else 1.0
        per_class[name] = {"support": int(m.sum()), "frac_over_64": float(big[m].mean()),
                           "median_h": int(np.median(H[m])), "median_w": int(np.median(W[m])),
                           "median_keep_when_downsampled": keep}
        print("%-11s %8d %9.1f%% %10d %10d %10.3f" % (
            name, m.sum(), big[m].mean() * 100, np.median(H[m]), np.median(W[m]), keep))

    (OUT / "native_resolution.json").write_text(json.dumps({
        "n_test": int(len(a)),
        "frac_over_64": float(big.mean()),
        "frac_over_96": float(((H > 96) | (W > 96)).mean()),
        "median_h": int(np.median(H)), "median_w": int(np.median(W)),
        "dies_lost_total": int((N0[big] - N1[big]).sum()) if big.any() else 0,
        "dies_total_in_downsampled": int(N0[big].sum()) if big.any() else 0,
        "per_class": per_class,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 -> {OUT / 'native_resolution.json'}")


if __name__ == "__main__":
    main()
