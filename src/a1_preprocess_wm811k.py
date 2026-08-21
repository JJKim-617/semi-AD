"""WM-811K 전처리: 원본 pickle -> 고정 크기 uint8 텐서 캐시(.npz).

원본 LSWMD.pkl 은 811,457개의 가변 크기 웨이퍼맵(uint8, 값 {0,1,2})을 담은
DataFrame 이라 매번 읽기 무겁다(로딩 수십 초, RAM 수 GB). 한 번 전처리해
고정 크기로 stack 한 뒤 npz 로 저장한다.

셀 값은 명목형(0=die 없음, 1=정상, 2=불량)이므로 리사이즈는 반드시
nearest-neighbor 로 한다. bilinear 등 보간은 존재하지 않는 중간값을 만든다.

사용:
    python src/a1_preprocess_wm811k.py --config configs/wm811k_64.yaml
    python src/a1_preprocess_wm811k.py --size 32 --tag wm811k_32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = [
    "none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Random", "Scratch", "Near-full",
]


def unwrap(cell) -> str:
    """[['none']] 같은 중첩 배열을 문자열로 편다. 비어있으면 ''."""
    while isinstance(cell, (np.ndarray, list)):
        if len(cell) == 0:
            return ""
        cell = cell[0]
    return str(cell) if cell is not None else ""


def resize_nn(wm: np.ndarray, size: int) -> np.ndarray:
    """nearest-neighbor 리사이즈. 인덱스 매핑만 쓰므로 의존성이 없다."""
    h, w = wm.shape
    rows = (np.arange(size) * h // size).clip(0, h - 1)
    cols = (np.arange(size) * w // size).clip(0, w - 1)
    return wm[rows[:, None], cols]


def pad_center(wm: np.ndarray, size: int) -> np.ndarray:
    """중앙 배치 + 0 패딩. size 보다 크면 nearest 로 줄인 뒤 배치한다."""
    h, w = wm.shape
    if h > size or w > size:
        scale = min(size / h, size / w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        rows = (np.arange(nh) * h // nh).clip(0, h - 1)
        cols = (np.arange(nw) * w // nw).clip(0, w - 1)
        wm = wm[rows[:, None], cols]
        h, w = wm.shape
    out = np.zeros((size, size), dtype=np.uint8)
    top, left = (size - h) // 2, (size - w) // 2
    out[top:top + h, left:left + w] = wm
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WM-811K 전처리")
    p.add_argument("--config", type=Path, help="YAML 설정. CLI 인자가 우선한다.")
    p.add_argument("--src", type=Path, default=Path("data/wm811k/LSWMD.pkl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/wm811k/cache"))
    p.add_argument("--size", type=int, default=64, help="정사각 리사이즈 크기 (기본 64)")
    p.add_argument("--mode", choices=["resize", "pad"], default="resize",
                   help="resize=nearest 보간, pad=중앙 배치 후 0 패딩")
    p.add_argument("--labeled-only", action="store_true", default=True,
                   help="라벨된 172,950장만 저장 (기본)")
    p.add_argument("--include-unlabeled", dest="labeled_only", action="store_false",
                   help="미라벨 638,507장도 함께 저장")
    p.add_argument("--tag", default=None, help="출력 파일명. 기본 wm811k_<size>")
    p.add_argument("--limit", type=int, default=None, help="앞 N개만 (스모크 테스트용)")
    args = p.parse_args()

    if args.config:
        import yaml
        cfg = yaml.safe_load(args.config.read_text()) or {}
        defaults = p.parse_args([])
        for k, v in cfg.items():
            k = k.replace("-", "_")
            if not hasattr(args, k):
                continue
            # CLI 로 명시한 값이 YAML 보다 우선한다(기본값과 다르면 건드리지 않음).
            if getattr(args, k) != getattr(defaults, k):
                continue
            # 기본값의 타입을 유지한다. Path 필드가 str 로 덮이지 않게.
            ref = getattr(defaults, k)
            if isinstance(ref, Path):
                v = Path(v)
            elif isinstance(ref, bool):
                v = bool(v)
            elif isinstance(ref, int) and not isinstance(ref, bool):
                v = int(v)
            setattr(args, k, v)
    if args.tag is None:
        args.tag = f"wm811k_{args.size}"
    return args


def main() -> None:
    args = parse_args()
    t0 = time.time()

    print(f"[load] {args.src}")
    df = pd.read_pickle(args.src)
    if args.limit:
        df = df.iloc[: args.limit]
    print(f"       {len(df):,} rows, {time.time() - t0:.1f}s")

    labels = df["failureType"].map(unwrap)
    split = df["trianTestLabel"].map(unwrap)  # 원본 컬럼명 오타 그대로
    keep = labels.isin(CLASSES) if args.labeled_only else pd.Series(True, index=df.index)
    idx = np.flatnonzero(keep.to_numpy())
    print(f"[select] {len(idx):,} / {len(df):,} (labeled_only={args.labeled_only})")

    fn = resize_nn if args.mode == "resize" else pad_center
    X = np.zeros((len(idx), args.size, args.size), dtype=np.uint8)
    for i, j in enumerate(idx):
        X[i] = fn(df["waferMap"].iat[j], args.size)
        if (i + 1) % 20000 == 0:
            print(f"       {i + 1:,}/{len(idx):,}")

    lab = labels.to_numpy()[idx]
    y = np.array([CLASSES.index(v) if v in CLASSES else -1 for v in lab], dtype=np.int8)
    meta = dict(
        size=args.size, mode=args.mode, labeled_only=bool(args.labeled_only),
        n=len(idx), classes=CLASSES,
        counts={c: int((y == k).sum()) for k, c in enumerate(CLASSES)},
        n_unlabeled=int((y == -1).sum()),
        src=str(args.src), seconds=round(time.time() - t0, 1),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz = args.out_dir / f"{args.tag}.npz"
    np.savez_compressed(
        npz, X=X, y=y,
        split=split.to_numpy()[idx].astype("U8"),
        die_size=df["dieSize"].to_numpy()[idx].astype(np.float32),
        lot_name=df["lotName"].to_numpy()[idx].astype("U16"),
        classes=np.array(CLASSES, dtype="U12"),
    )
    (args.out_dir / f"{args.tag}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[save] {npz}  ({npz.stat().st_size / 1e6:.1f} MB)")
    print(f"[done] X={X.shape} {X.dtype}, {meta['seconds']}s")
    for c, n in meta["counts"].items():
        if n:
            print(f"       {c:<10} {n:>7,}")
    if meta["n_unlabeled"]:
        print(f"       {'unlabeled':<10} {meta['n_unlabeled']:>7,}")


if __name__ == "__main__":
    main()
