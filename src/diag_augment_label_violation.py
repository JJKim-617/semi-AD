"""증강이 none 웨이퍼를 결함처럼 만드는가 (18차).

`diag_leakage_mechanism` 의 결과가 세운 설명을 정면으로 잰다.

| 추가 증강 | none -> 결함 누출 (군 평균) |
|---|---:|
| 없음 (건강한 대조군 2 seed) | 1,925 |
| **translate** | **1,190** |
| scale | 2,500 |
| dropout | 4,014 |
| noise | 7,496 |

**설명**: `die_noise` 는 정상 다이를 불량으로 뒤집고 `die_dropout` 은 다이를 지운다.
none 웨이퍼에 그것을 걸면 **산발적 불량이 구조를 이룬 것처럼 보인다** — 즉 라벨이 깨진다.
모델은 "그런 것도 none 이다" 가 아니라 "그런 것이 결함이다" 쪽으로 밀린다.
`random_scale` 은 다이 격자를 nearest 로 다시 뜨므로 가는 구조를 지우거나 굵힌다.
**`random_translate` 만 강체 이동이라 웨이퍼 자체를 건드리지 않는다.**

자 둘로 잰다.
1. **구조 변화** — 모델 없이. 불량 다이가 몇 개 생겼나/사라졌나.
2. **뒤집힘 비율** — 참조 모델(증강 없이 학습된 것)이 none 이라 맞힌 웨이퍼가
   증강 뒤 결함으로 불리는 비율. **증강이 라벨을 깨뜨리는 정도의 대리 측정**이다.
   참조 모델의 판단을 진리로 두는 것이 아니라, 이 데이터셋에서 "결함처럼 보인다" 를
   정의하는 데 쓴다. **대리 측정이라는 점을 결론에서 반드시 밝힌다.**
"""

from __future__ import annotations

import numpy as np


def structural_change(x: np.ndarray, xa: np.ndarray) -> dict:
    """원본과 증강본 사이에서 다이 값이 어떻게 바뀌었나. 전부 원본 다이 수로 나눈다.

    강체 이동(translate)은 웨이퍼가 캔버스 안에서 옮겨갈 뿐이라 세 값이 전부 0 이다.
    **다이가 하나도 없으면 비율이 정의되지 않으므로 NaN 이다** — 0 은
    "아무 일도 없었다" 와 구분되지 않는다.
    """
    x = np.asarray(x); xa = np.asarray(xa)
    if x.shape != xa.shape:
        raise ValueError(f"모양이 다르다: {x.shape} vs {xa.shape}")
    ndie = float((x > 0).sum())
    if ndie == 0:
        return dict(fail_added_frac=np.nan, fail_removed_frac=np.nan,
                    die_removed_frac=np.nan)
    # 이동은 위치가 바뀌므로 칸 대 칸으로 비교하면 전부 '바뀐 것' 이 된다.
    # 우리가 묻는 것은 위치가 아니라 **구성**이다 — 값별 개수로 센다.
    n_fail_before = float((x == 2).sum()); n_fail_after = float((xa == 2).sum())
    n_die_before = ndie; n_die_after = float((xa > 0).sum())
    return dict(
        fail_added_frac=max(n_fail_after - n_fail_before, 0.0) / ndie,
        fail_removed_frac=max(n_fail_before - n_fail_after, 0.0) / ndie,
        die_removed_frac=max(n_die_before - n_die_after, 0.0) / ndie,
    )


def flip_rate(logits_before: np.ndarray, logits_after: np.ndarray,
              none_index: int = 0) -> float:
    """참조 모델이 none 이라 맞힌 것 중, 증강 뒤 결함이 된 비율.

    **원래 틀렸던 웨이퍼는 분모에서 뺀다** — 안 그러면 참조 모델의 오류율이
    증강의 성질로 오인된다. 방향이 있는 자다(결함 -> none 회복은 세지 않는다).
    맞힌 것이 없으면 NaN 이다.
    """
    b = np.asarray(logits_before).argmax(1)
    a = np.asarray(logits_after).argmax(1)
    right = b == none_index
    if not right.any():
        return float("nan")
    return float((a[right] != none_index).mean())


def main():
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _runtime import setup
    setup("eval")
    import torch  # noqa: E402  (setup 이후여야 한다)

    from a3_train_wm811k_cls import build_model, encode_device  # noqa: E402
    from a18_augment import RECIPES, apply_recipe  # noqa: E402

    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64pad.npz")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--refs", nargs="*",
                   default=["result/cls_baseline/e8_pad_best.pt",
                            "result/cls_baseline/e10_pad_s2_best.pt"],
                   help="증강 없이 학습된 참조 모델. 여러 개면 평균한다.")
    p.add_argument("--n", type=int, default=20000, help="표본으로 쓸 none 웨이퍼 수")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--out", default="result/cls_baseline/e22_augment_violation.json")
    a = p.parse_args()

    d = np.load(a.cache); sp = np.load(a.splits)
    X, y = d["X"], d["y"].astype(np.int64)
    te = sp["test"]
    rng = np.random.default_rng(a.seed)
    idx = te[y[te] == 0]
    if len(idx) > a.n:
        idx = rng.choice(idx, a.n, replace=False)
    x0 = X[np.sort(idx)]
    print(f"none 표본 {len(x0):,}장 (test)")

    variants = {"원본": x0}
    for name in ("translate", "rotate", "scale", "dropout", "noise"):
        fn, kw = RECIPES[name]
        variants[name] = fn(x0, rng=np.random.default_rng(a.seed + 1), **kw)
    variants["all(4종)"] = apply_recipe(
        x0, ["scale", "translate", "noise", "dropout"],
        np.random.default_rng(a.seed + 1))

    def run(model, xb):
        outs = []
        with torch.no_grad():
            for i in range(0, len(xb), a.batch):
                t = encode_device(xb[i:i + a.batch], device="cuda")
                outs.append(model(t).float().cpu().numpy())
        return np.concatenate(outs)

    # 반대 방향도 잰다. 첫 판이 내 설명을 반증했기 때문이다 —
    # noise 는 뒤집힘이 가장 낮은데 누출은 가장 높았다. 그렇다면 증강이 none 을 결함처럼
    # 만드는 것이 아니라 **결함을 none 처럼 만들어** 모델을 예민하게 미는 것일 수 있다.
    didx = te[y[te] != 0]
    if len(didx) > a.n:
        didx = rng.choice(didx, a.n, replace=False)
    xd = X[np.sort(didx)]
    print(f"결함 표본 {len(xd):,}장 (test)")
    dvar = {"원본": xd}
    for name in ("translate", "rotate", "scale", "dropout", "noise"):
        fn, kw = RECIPES[name]
        dvar[name] = fn(xd, rng=np.random.default_rng(a.seed + 2), **kw)
    dvar["all(4종)"] = apply_recipe(xd, ["scale", "translate", "noise", "dropout"],
                                    np.random.default_rng(a.seed + 2))

    per_ref, per_ref_d = {}, {}
    for ref in a.refs:
        m = build_model(num_classes=9).cuda().eval()
        m.load_state_dict(torch.load(ref, map_location="cuda", weights_only=True))
        lg = {k: run(m, v) for k, v in variants.items()}
        per_ref[Path(ref).stem] = {k: flip_rate(lg["원본"], v)
                                   for k, v in lg.items() if k != "원본"}
        # 결함 -> none 파괴율. 같은 `flip_rate` 를 방향만 바꿔 쓴다:
        # 로짓의 none 칸과 나머지를 맞바꾼 이진 문제로 보면 된다.
        dl = {k: run(m, v) for k, v in dvar.items()}
        bin_ = lambda z: np.stack(  # noqa: E731
            [z[:, 1:].max(1), z[:, 0]], 1)   # [결함, none] 순서로 뒤집는다
        per_ref_d[Path(ref).stem] = {k: flip_rate(bin_(dl["원본"]), bin_(v))
                                     for k, v in dl.items() if k != "원본"}
        del m
        torch.cuda.empty_cache()

    struct = {}
    for k, v in variants.items():
        if k == "원본":
            continue
        c = structural_change(x0, v)
        struct[k] = c

    print("\n## 증강이 none 웨이퍼의 구성을 얼마나 바꾸나 (모델 없이)")
    print(f"{'증강':<12}{'불량 추가':>12}{'불량 제거':>12}{'다이 제거':>12}")
    for k, c in struct.items():
        print(f"{k:<12}{c['fail_added_frac']:>12.4%}"
              f"{c['fail_removed_frac']:>12.4%}{c['die_removed_frac']:>12.4%}")

    print("\n## 참조 모델이 none 이라 맞힌 것이 증강 뒤 결함으로 뒤집히는 비율")
    names = list(struct)
    print(f"{'참조 모델':<22}" + "".join(f"{n:>12}" for n in names))
    for r, v in per_ref.items():
        print(f"{r:<22}" + "".join(f"{v[n]:>12.2%}" for n in names))
    mean = {n: float(np.mean([v[n] for v in per_ref.values()])) for n in names}
    print(f"{'평균':<22}" + "".join(f"{mean[n]:>12.2%}" for n in names))

    print("\n## 반대 방향 — 결함이라 맞힌 것이 증강 뒤 none 으로 무너지는 비율")
    print(f"{'참조 모델':<22}" + "".join(f"{n:>12}" for n in names))
    for r, v in per_ref_d.items():
        print(f"{r:<22}" + "".join(f"{v[n]:>12.2%}" for n in names))
    meand = {n: float(np.mean([v[n] for v in per_ref_d.values()])) for n in names}
    print(f"{'평균':<22}" + "".join(f"{meand[n]:>12.2%}" for n in names))

    print("\n## 두 방향을 나란히 (증강이 라벨을 어느 쪽으로 깨뜨리나)")
    print(f"{'증강':<12}{'none->결함':>12}{'결함->none':>12}{'합':>10}"
          f"{'실제 누출':>12}{'실제 recall':>13}")
    obs = {"translate": (1190, 0.8180), "scale": (2500, 0.8273),
           "rotate": (float("nan"), float("nan")),
           "dropout": (4014, 0.8223), "noise": (7496, 0.8359),
           "all(4종)": (3672, 0.8684)}
    for n in names:
        lk, rc = obs.get(n, (float("nan"), float("nan")))
        print(f"{n:<12}{mean[n]:>12.2%}{meand[n]:>12.2%}"
              f"{mean[n] + meand[n]:>10.2%}{lk:>12,.0f}{rc:>13.4f}")

    Path(a.out).write_text(json.dumps(
        dict(structural=struct, flip_rate=per_ref, flip_rate_mean=mean,
             destroy_rate=per_ref_d, destroy_rate_mean=meand,
             n=len(x0), n_defect=len(xd)), ensure_ascii=False, indent=2))
    print(f"\n[저장] {a.out}")


if __name__ == "__main__":
    main()
