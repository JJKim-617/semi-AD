"""위치 민감도가 none -> 결함 누출을 예측하는가 (18차).

## 물음

`docs/research/leakage_mechanism/README.md` 가 남긴 좁혀진 물음은
**"왜 translate 만 판별력 자체를 올리는가"** 다. 그럴듯한 이야기 하나는

> translate 는 모델에게서 **절대 위치**를 빼앗는다. 국소 3종(Edge-Loc, Loc, Scratch)의
> 누출은 반지름에 의존하는데(대조군 Q5/Q1 = 2.35), 그 의존이 곧 절대 위치를 쓰고
> 있다는 증거다. 위치를 못 쓰게 하면 그 실패가 사라진다.

**그런데 translate 로 학습한 모델이 이동에 둔감한 것은 동어반복이다.**
그래서 **translate 를 안 쓴 모델들 안에서** 관계를 본다. E18 백본 7개는 전부
`extra_augment=[]` 인데 누출이 1,246~3,939 로 3배 넘게 벌어진다.
거기서 위치 민감도가 누출을 예측하면 **순환이 아니다.**

## 자

같은 웨이퍼를 조금 옮겨 넣고 **예측 분포가 얼마나 움직이는지**를 총변동거리로 잰다.
argmax 뒤집힘보다 섬세하다 — 확신도만 흔드는 변환도 잡힌다
(가설 D 를 재던 자가 그것을 못 잡아 실패했다).

**대조군 변환을 같이 잰다.** `die_noise` 는 위치와 무관한 교란이므로,
어떤 모델이 그냥 전반적으로 예민한 것인지 **위치에만** 예민한 것인지를 가른다.
비율(이동 민감도 / 잡음 민감도)이 "위치 특유의 예민함" 이다.
"""

from __future__ import annotations

import numpy as np


def mean_total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """행별 총변동거리의 평균. 확률 분포 두 벌을 받는다.

    총변동거리는 L1 의 **절반**이라 [0,1] 에 산다. 로짓을 그대로 넘기는 실수를
    조용히 통과시키지 않는다 — 행 합이 1 이 아니면 거절한다.
    """
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"모양이 다르다: {p.shape} vs {q.shape}")
    for name, z in (("p", p), ("q", q)):
        if z.min() < -1e-9 or not np.allclose(z.sum(1), 1.0, atol=1e-4):
            raise ValueError(f"{name} 의 행이 확률 분포가 아니다 (로짓을 넘겼나?)")
    return float(0.5 * np.abs(p - q).sum(1).mean())


def main():
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _runtime import setup
    setup("eval")
    import torch  # noqa: E402

    from a3_train_wm811k_cls import build_model, encode_device  # noqa: E402
    from a18_augment import die_noise, random_rotate, random_translate  # noqa: E402

    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="docs/experiments/ensemble_members.json")
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64pad.npz")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="result/cls_baseline/e22_shift_sensitivity.json")
    a = p.parse_args()

    entries = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    # pad64 표현에 3채널인 것만. 극좌표와 채널 추가 모델은 자가 달라진다.
    entries = [e for e in entries
               if e["representation"] == "pad64"
               and not e.get("density_ks") and not e.get("line_ls")
               and Path(e["ckpt"]).exists()]

    d, sp = np.load(a.cache), np.load(a.splits)
    X, y = d["X"], d["y"].astype(np.int64)
    te = sp["test"]
    rng = np.random.default_rng(a.seed)
    idx = te[y[te] == 0]
    idx = np.sort(rng.choice(idx, min(a.n, len(idx)), replace=False))
    x0 = X[idx]
    xt = random_translate(x0, rng=np.random.default_rng(a.seed + 1), max_shift=4)
    xn = die_noise(x0, rng=np.random.default_rng(a.seed + 1), rate=0.01)
    # E23 용. 회전 학습이 회전 민감도만 줄이는지 이동 민감도까지 줄이는지 가른다.
    xr = random_rotate(x0, rng=np.random.default_rng(a.seed + 1), max_deg=180.0)
    print(f"none 표본 {len(x0):,}장, 모델 {len(entries)}개")

    def probs(model, xb):
        outs = []
        with torch.no_grad():
            for i in range(0, len(xb), a.batch):
                z = model(encode_device(xb[i:i + a.batch], device="cuda"))
                outs.append(torch.softmax(z.float(), 1).cpu().numpy())
        return np.concatenate(outs).astype(np.float64)

    rows = {}
    for e in entries:
        try:
            m = build_model(num_classes=9, backbone=e.get("backbone", "resnet18"))
            m.load_state_dict(torch.load(e["ckpt"], map_location="cpu",
                                         weights_only=True))
            m = m.cuda().eval()
        except Exception as ex:                       # 못 읽는 것은 건너뛰되 알린다
            print(f"  [건너뜀] {e['tag']}: {ex}")
            continue
        p0 = probs(m, x0)
        rows[e["tag"]] = dict(
            shift=mean_total_variation(p0, probs(m, xt)),
            noise=mean_total_variation(p0, probs(m, xn)),
            rot=mean_total_variation(p0, probs(m, xr)),
            augment=e.get("augment", ""), backbone=e.get("backbone", "resnet18"))
        del m
        torch.cuda.empty_cache()
        print(f"  {e['tag']:<30} 이동 {rows[e['tag']]['shift']:.4f} "
              f"회전 {rows[e['tag']]['rot']:.4f} "
              f"잡음 {rows[e['tag']]['noise']:.4f}", flush=True)

    # 누출은 캐시된 로짓에서 읽는다(같은 모델, 전체 test).
    leak = {}
    for t in rows:
        f = Path("result/posthoc") / f"{t}_logits.npz"
        if not f.exists():
            continue
        z = np.load(f)
        leak[t] = int(((z["test_y"] == 0) & (z["test_logits"].argmax(1) != 0)).sum())

    def corr(tags, key):
        v = np.array([rows[t][key] for t in tags]); L = np.array([leak[t] for t in tags])
        if len(tags) < 3:
            return None
        return float(np.corrcoef(v, L)[0, 1])

    have = [t for t in rows if t in leak]
    notr = [t for t in have if "translate" not in rows[t]["augment"]]
    withtr = [t for t in have if "translate" in rows[t]["augment"]]
    bb = [t for t in notr if rows[t]["backbone"] != "resnet18"]

    print("\n## 이동 민감도와 누출의 상관 (양수면 '위치에 예민할수록 많이 샌다')")
    for name, tags in [("전체", have), ("**translate 안 쓴 것만**", notr),
                       ("그 중 E18 백본만", bb), ("translate 쓴 것만", withtr)]:
        c = corr(tags, "shift"); cn = corr(tags, "noise")
        if c is None:
            print(f"  {name:<24} (n<3)")
            continue
        print(f"  {name:<24} n={len(tags):2d}  이동 r={c:+.3f}   잡음 r={cn:+.3f}")

    print("\n## 군별 평균 (이동/잡음 = 위치 특유의 예민함)")
    groups = {}
    for t in have:
        g = rows[t]["augment"] or ("백본" if rows[t]["backbone"] != "resnet18"
                                   else "무증강 resnet18")
        groups.setdefault(g, []).append(t)
    print(f"  {'군':<22}{'n':>3}{'이동':>9}{'회전':>9}{'잡음':>9}"
          f"{'이동/잡음':>11}{'누출':>9}")
    for g, tags in sorted(groups.items(), key=lambda kv: -np.mean(
            [rows[t]["shift"] for t in kv[1]])):
        s = np.mean([rows[t]["shift"] for t in tags])
        r = np.mean([rows[t].get("rot", np.nan) for t in tags])
        n = np.mean([rows[t]["noise"] for t in tags])
        L = np.mean([leak[t] for t in tags])
        print(f"  {g:<22}{len(tags):>3}{s:>9.4f}{r:>9.4f}{n:>9.4f}"
              f"{s / n:>11.2f}{L:>9,.0f}")

    Path(a.out).write_text(json.dumps(dict(rows=rows, leak=leak),
                                      ensure_ascii=False, indent=2))
    print(f"\n[저장] {a.out}")


if __name__ == "__main__":
    main()
