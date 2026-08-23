"""왜 translate 만 none -> 결함 누출의 **총량**을 줄이는가 (18차).

17차가 "설명 없는 관측" 으로 남긴 것이다. 여러 기법이 누출의 **퍼짐**(모델 간 범위)은
줄였는데 **수준**(총량)은 translate 만 줄였다. 설명이 있으면 같은 성질의 기법을 더 찾을 수 있다.

**경쟁 가설 넷을 실행 전에 적는다.**

- **A (증강이면 아무거나)**: 추가 증강이면 무엇이든 준다. -> E17 형제(scale, noise,
  dropout, all)로 가른다. 이들도 줄이면 translate 는 특별하지 않다.
- **R (재현율과 맞바꿈)**: 그냥 결함을 덜 부르는 것이다. -> 결함 recall 이 같이 떨어지는가.
- **C (확신도 하락)**: 정칙화라 확신이 낮아졌을 뿐이다. -> 대조군의 none 로짓에 편향을
  더해 **같은 누출 수준으로 운영점을 맞춘 뒤** 프로필과 macro-F1 을 비교한다.
  맞춘 대조군이 translate 와 같아지면 C 다.
- **P (위치 지름길)**: WM-811K 의 결함 이름 넷은 **위치의 이름**이다
  (Center, Donut, Edge-Loc, Edge-Ring). none 웨이퍼의 불량 다이가 우연히 그 자리에 있으면
  위치만으로 그 이름이 붙는다. translate 는 위치->라벨 지름길을 끊는다.
  -> 누출 감소가 **위치로 정의된 넷에 몰리고** 모양으로 정의된 넷(Loc, Random, Scratch,
  Near-full)에서는 작아야 한다. 그리고 새는 웨이퍼의 **불량 다이 반지름**이 달라야 한다.

전부 캐시된 로짓(`result/posthoc/*_logits.npz`)으로 계산한다. GPU 를 쓰지 않는다.
"""

from __future__ import annotations

import numpy as np

NONE = 0
CLASSES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Loc", "Random", "Scratch", "Near-full"]
# 이름 자체가 위치인 클래스. 실행 전에 정해 둔다 — 결과를 보고 나누면 안 된다.
POSITIONAL = ["Center", "Donut", "Edge-Loc", "Edge-Ring"]
SHAPE = ["Loc", "Random", "Scratch", "Near-full"]


def leakage_by_class(logits: np.ndarray, y: np.ndarray, none_index: int = NONE
                     ) -> np.ndarray:
    """라벨이 none 인 행이 어느 클래스로 갔는지 센다.

    `none_index` 칸은 **지킨 것**이지 누출이 아니다. 총 누출은 나머지의 합이다.
    none 이 한 장도 없으면 전부 0 을 돌려준다 — 없는 숫자를 만들지 않는다.
    """
    logits = np.asarray(logits)
    y = np.asarray(y)
    sel = y == none_index
    out = np.zeros(logits.shape[1], dtype=np.int64)
    if not sel.any():
        return out
    pred = logits[sel].argmax(1)
    idx, cnt = np.unique(pred, return_counts=True)
    out[idx] = cnt
    return out


def _leak_total(logits, y, none_index, bias):
    shifted = logits.copy()
    shifted[:, none_index] += bias
    c = leakage_by_class(shifted, y, none_index)
    return int(c.sum() - c[none_index])


def none_bias_for_target_leak(logits: np.ndarray, y: np.ndarray,
                              none_index: int = NONE, target: int = 0,
                              tol: float = 1e-3) -> float:
    """none 로짓에 더할 **가장 작은 음이 아닌 편향** — 누출이 `target` 이하가 되게.

    가설 C 를 가르는 도구다. 누출을 줄이는 방향만 본다(편향 >= 0). 이미 목표 이하면 0 이다.
    누출은 편향에 대해 비증가 계단함수라 이분탐색이 성립한다.
    """
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y)
    if _leak_total(logits, y, none_index, 0.0) <= target:
        return 0.0
    lo, hi = 0.0, 1.0
    while _leak_total(logits, y, none_index, hi) > target:
        hi *= 2.0
        if hi > 1e6:                      # 도달 불가 — 조용히 큰 값을 주지 않는다
            raise ValueError("편향으로 목표 누출에 도달하지 못했다")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if _leak_total(logits, y, none_index, mid) > target:
            lo = mid
        else:
            hi = mid
    return float(hi)


def fail_centroid_radius(x: np.ndarray) -> np.ndarray:
    """불량 다이 무게중심이 **다이 영역의 중심**에서 얼마나 떨어져 있나 (정규화).

    (N,H,W) uint8, 0=다이 없음 / 1=정상 / 2=불량. 다이 영역의 RMS 반지름으로 나누므로
    **웨이퍼 크기에 불변**이다 — 크기 교란이 위치 효과로 오인되지 않게 하려는 것이다.
    불량이 하나도 없으면 NaN 이다. 0 으로 채우면 '중심' 과 구분이 안 된다.
    """
    x = np.asarray(x)
    n, h, w = x.shape
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rr = rr.astype(np.float64); cc = cc.astype(np.float64)
    out = np.full(n, np.nan)
    for i in range(n):
        die = x[i] > 0
        fail = x[i] == 2
        if not die.any() or not fail.any():
            continue
        cy, cx = rr[die].mean(), cc[die].mean()
        rms = np.sqrt(((rr[die] - cy) ** 2 + (cc[die] - cx) ** 2).mean())
        if rms == 0:
            continue
        fy, fx = rr[fail].mean(), cc[fail].mean()
        out[i] = np.hypot(fy - cy, fx - cx) / rms
    return out


# --- 아래는 보고용. 위 셋만 시험이 있다 -------------------------------------

def macro_f1(y, pred, n_classes=9):
    f = []
    for c in range(n_classes):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def defect_recall(y, pred, none_index=NONE):
    """결함 웨이퍼를 (클래스는 틀려도) 결함이라고 부른 비율."""
    sel = y != none_index
    return float((pred[sel] != none_index).mean())


def main():
    import argparse
    import json
    from collections import OrderedDict
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--logit-dir", default="result/posthoc")
    p.add_argument("--cache", default="data/wm811k/cache/wm811k_64pad.npz")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--out", default="result/cls_baseline/e22_leakage_mechanism.json")
    p.add_argument("--skip-radius", action="store_true")
    a = p.parse_args()

    groups = OrderedDict([
        ("대조군 resnet18 (증강 없음)", ["e8_pad", "e10_pad_s1", "e10_pad_s2"]),
        ("translate", [f"e17_translate_s{i}" for i in range(4)]),
        ("scale", ["e17_scale_s0", "e17_scale_s1"]),
        ("noise", ["e17_noise_s0", "e17_noise_s1"]),
        ("dropout", ["e17_dropout_s0", "e17_dropout_s1"]),
        ("all(4종 동시)", ["e17_all_s0", "e17_all_s1"]),
        ("백본(증강 없음)", ["e18_shufflenet_v2_s0", "e18_shufflenet_v2_s1",
                             "e18_mobilenet_v3_s0", "e18_mobilenet_v3_s1",
                             "e18_efficientnet_b0_s0", "e18_efficientnet_b0_s1",
                             "e18_convnext_tiny_s0"]),
        ("밀도채널 E20", ["e20_dens_s0", "e20_dens_s1", "e20_dens_s2"]),
        ("선필터 E21", ["e21_line_s0", "e21_line_s1", "e21_line_s2"]),
    ])
    # E22 가 끝나면 자동으로 붙는다
    e22 = sorted(q.stem[:-7] for q in Path(a.logit_dir).glob("e22_*_logits.npz"))
    if e22:
        groups["**E22 백본 x translate**"] = e22

    ld = Path(a.logit_dir)
    per_model, y_test = {}, None
    for g, tags in groups.items():
        for t in tags:
            f = ld / f"{t}_logits.npz"
            if not f.exists():
                print(f"  [없음] {t}")
                continue
            d = np.load(f)
            lg, y = d["test_logits"].astype(np.float64), d["test_y"]
            y_test = y if y_test is None else y_test
            pred = lg.argmax(1)
            per_model[t] = dict(
                group=g, logits=lg, pred=pred,
                leak=leakage_by_class(lg, y), f1=macro_f1(y, pred),
                rec=defect_recall(y, pred))

    def gmean(g, key, fn=np.mean):
        v = [m[key] for m in per_model.values() if m["group"] == g]
        return fn(np.array(v), axis=0) if v else None

    print(f"\ntest {len(y_test):,}장, none {int((y_test == NONE).sum()):,}장\n")
    print("## 군별 — 누출 총량, 결함 recall, macro-F1")
    print(f"{'군':<28}{'n':>3}{'누출 총량':>12}{'범위':>16}"
          f"{'결함 recall':>13}{'macro-F1':>10}")
    rows = {}
    for g in groups:
        ms = [m for m in per_model.values() if m["group"] == g]
        if not ms:
            continue
        tot = np.array([int(m["leak"].sum() - m["leak"][NONE]) for m in ms])
        rec = np.array([m["rec"] for m in ms]); f1 = np.array([m["f1"] for m in ms])
        rows[g] = dict(n=len(ms), leak_mean=float(tot.mean()),
                       leak_min=int(tot.min()), leak_max=int(tot.max()),
                       recall=float(rec.mean()), f1=float(f1.mean()),
                       per_class=gmean(g, "leak").tolist())
        print(f"{g:<28}{len(ms):>3}{tot.mean():>12,.0f}"
              f"{f'{tot.min():,}~{tot.max():,}':>16}"
              f"{rec.mean():>13.4f}{f1.mean():>10.4f}")

    print("\n## 가설 P — 누출이 어느 클래스로 가나 (군 평균)")
    hdr = "".join(f"{c:>11}" for c in CLASSES[1:])
    print(f"{'군':<28}{hdr}")
    for g, r in rows.items():
        pc = r["per_class"]
        print(f"{g:<28}" + "".join(f"{pc[i]:>11,.0f}" for i in range(1, 9)))

    print("\n## 모델별 — 군 평균이 한 모델에 끌려가지 않는지 확인한다")
    print(f"{'모델':<24}{'누출':>8}" + "".join(f"{c[:9]:>10}" for c in CLASSES[1:])
          + f"{'recall':>9}{'macro-F1':>10}")
    for t, m in per_model.items():
        tot = int(m["leak"].sum() - m["leak"][NONE])
        print(f"{t:<24}{tot:>8,}"
              + "".join(f"{m['leak'][i]:>10,}" for i in range(1, 9))
              + f"{m['rec']:>9.4f}{m['f1']:>10.4f}")
    out_pm = {t: dict(group=m["group"], leak=int(m["leak"].sum() - m["leak"][NONE]),
                      per_class=m["leak"].tolist(), recall=m["rec"], f1=m["f1"])
              for t, m in per_model.items()}

    base = rows.get("대조군 resnet18 (증강 없음)")
    print("\n## 가설 P — 위치로 정의된 넷 대 모양으로 정의된 넷 (대조군 대비 감소율)")
    print(f"{'군':<28}{'위치 4종':>12}{'감소율':>9}{'모양 4종':>12}{'감소율':>9}")
    ip = [CLASSES.index(c) for c in POSITIONAL]
    ish = [CLASSES.index(c) for c in SHAPE]
    bp, bs = sum(base["per_class"][i] for i in ip), sum(base["per_class"][i] for i in ish)
    for g, r in rows.items():
        pp = sum(r["per_class"][i] for i in ip); ss = sum(r["per_class"][i] for i in ish)
        r["pos_leak"], r["shape_leak"] = pp, ss
        print(f"{g:<28}{pp:>12,.0f}{1 - pp / bp:>9.1%}{ss:>12,.0f}{1 - ss / bs:>9.1%}")

    print("\n## 가설 C — 대조군의 운영점을 translate 의 누출 수준으로 맞춘다")
    tgt = int(round(rows["translate"]["leak_mean"]))
    print(f"목표 누출 = translate 군 평균 {tgt:,}")
    print(f"{'모델':<22}{'편향':>7}{'누출':>9}{'위치4':>8}{'모양4':>8}"
          f"{'결함recall':>12}{'macro-F1':>10}")
    matched = []
    for t, m in per_model.items():
        if m["group"] != "대조군 resnet18 (증강 없음)":
            continue
        b = none_bias_for_target_leak(m["logits"], y_test, NONE, tgt)
        lg = m["logits"].copy(); lg[:, NONE] += b
        pr = lg.argmax(1); lk = leakage_by_class(lg, y_test)
        row = dict(tag=t, bias=float(b), leak=int(lk.sum() - lk[NONE]),
                   pos=int(sum(lk[i] for i in ip)), shape=int(sum(lk[i] for i in ish)),
                   recall=defect_recall(y_test, pr), f1=macro_f1(y_test, pr))
        matched.append(row)
        print(f"{t:<22}{b:>7.2f}{row['leak']:>9,}{row['pos']:>8,}{row['shape']:>8,}"
              f"{row['recall']:>12.4f}{row['f1']:>10.4f}")
    tr = rows["translate"]
    print(f"{'translate (실제)':<22}{'-':>7}{tr['leak_mean']:>9,.0f}"
          f"{tr['pos_leak']:>8,.0f}{tr['shape_leak']:>8,.0f}"
          f"{tr['recall']:>12.4f}{tr['f1']:>10.4f}")

    # e10_pad_s1 은 Scratch 가 붕괴한 모델이다. 군 평균의 Scratch 칸을 혼자 끌고 갈 수
    # 있으므로 **그것을 뺀 대조군**으로 같은 표를 다시 낸다. 교란인지 아닌지가 갈린다.
    print("\n## 가설 P (재검) — 망가진 e10_pad_s1 을 뺀 건강한 대조군 기준")
    healthy = [per_model[t]["leak"] for t in ("e8_pad", "e10_pad_s2") if t in per_model]
    if healthy:
        hb = np.mean(healthy, axis=0)
        hp, hs = sum(hb[i] for i in ip), sum(hb[i] for i in ish)
        print(f"{'군':<28}{'위치 4종':>12}{'감소율':>9}{'모양 4종':>12}{'감소율':>9}")
        print(f"{'건강한 대조군 (2 seed)':<28}{hp:>12,.0f}{0:>9.1%}{hs:>12,.0f}{0:>9.1%}")
        for g in ("translate", "밀도채널 E20", "선필터 E21", "백본(증강 없음)", "scale"):
            if g not in rows:
                continue
            pp = rows[g]["pos_leak"]; ss = rows[g]["shape_leak"]
            print(f"{g:<28}{pp:>12,.0f}{1 - pp / hp:>9.1%}{ss:>12,.0f}{1 - ss / hs:>9.1%}")
        out_healthy = dict(pos=float(hp), shape=float(hs), per_class=hb.tolist())
    else:
        out_healthy = None

    out = dict(groups=rows, per_model=out_pm, matched_control=matched,
               target_leak=tgt, healthy_control=out_healthy)

    if not a.skip_radius:
        print("\n## 가설 P — 새는 웨이퍼의 불량 다이 반지름 (다이 영역 중심 기준, 크기 불변)")
        X = np.load(a.cache)["X"]
        te = np.load(a.splits)["test"]
        noneish = y_test == NONE
        rad = fail_centroid_radius(X[te][noneish])
        ok = ~np.isnan(rad)
        qs = np.nanquantile(rad, [0.2, 0.4, 0.6, 0.8])
        qi = np.digitize(rad, qs)
        print(f"none 웨이퍼 {noneish.sum():,}장 중 불량 다이가 있는 것 {ok.sum():,}장 "
              f"({ok.mean():.1%}). 5분위 경계 {np.round(qs, 3).tolist()}")
        print(f"{'군':<28}{'Q1(중심)':>10}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Q5(가장자리)':>13}"
              f"{'새는 것의 평균 r':>16}")
        radrows = {}
        for g in groups:
            ms = [m for m in per_model.values() if m["group"] == g]
            if not ms:
                continue
            rate = np.zeros(5); mr = []
            for m in ms:
                leaked = (m["pred"][noneish] != NONE) & ok
                for q in range(5):
                    inq = ok & (qi == q)
                    rate[q] += (leaked & inq).sum() / max(inq.sum(), 1)
                mr.append(float(np.nanmean(rad[leaked])) if leaked.any() else np.nan)
            rate /= len(ms)
            radrows[g] = dict(rate_by_quintile=rate.tolist(),
                              mean_r_of_leaked=float(np.nanmean(mr)))
            print(f"{g:<28}" + "".join(f"{rate[q]:>10.2%}" if q == 0 else
                                       (f"{rate[q]:>8.2%}" if q < 4 else f"{rate[q]:>13.2%}")
                                       for q in range(5))
                  + f"{np.nanmean(mr):>16.3f}")
        out["radius"] = radrows

        # 국소 3종(Edge-Loc, Loc, Scratch) 대 전역 3종(Center, Donut, Edge-Ring).
        # 건강한 대조군으로 재보니 감소가 국소 3종에 몰려 있어서, 반지름 의존이
        # 그 셋에서만 평평해지는지를 확인한다.
        LOCAL = [CLASSES.index(c) for c in ("Edge-Loc", "Loc", "Scratch")]
        GLOBAL = [CLASSES.index(c) for c in ("Center", "Donut", "Edge-Ring")]
        print("\n## 국소 3종(Edge-Loc/Loc/Scratch) 으로 새는 비율의 반지름 의존")
        print(f"{'군':<28}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Q5':>8}{'Q5/Q1':>8}")
        loc_rows = {}
        for g in groups:
            ms = [m for m in per_model.values() if m["group"] == g]
            if not ms:
                continue
            for name, idxs in (("국소", LOCAL), ("전역", GLOBAL)):
                rate = np.zeros(5)
                for m in ms:
                    hit = np.isin(m["pred"][noneish], idxs) & ok
                    for q in range(5):
                        inq = ok & (qi == q)
                        rate[q] += (hit & inq).sum() / max(inq.sum(), 1)
                rate /= len(ms)
                loc_rows[f"{g}|{name}"] = rate.tolist()
                print(f"{g + ' ' + name:<28}"
                      + "".join(f"{rate[q]:>8.2%}" for q in range(5))
                      + f"{rate[4] / max(rate[0], 1e-9):>8.2f}")
        out["radius_local_global"] = loc_rows

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[저장] {a.out}")


if __name__ == "__main__":
    main()
