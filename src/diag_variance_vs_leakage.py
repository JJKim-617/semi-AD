"""설명 없는 관측 하나를 파본다 — 왜 어떤 기법은 분산만 줄이고 평균은 못 올리는가.

이 프로젝트에서 세 기법이 seed 분산을 크게 줄이고도 평균을 못 올려 기각됐다
(자기지도 20배, 추가 증강 3~7배, EMA 2배). 유일한 예외가 translate 로,
분산도 줄이고(0.0495 -> 0.0067) 평균도 올렸다(+0.038).

**가설:** 공통 화폐는 `none -> 결함` 누출이다. 이 문서가 이미 보인 것은
(1) macro 분산의 3/4 이 Scratch 하나에서 나오고, (2) Scratch 붕괴의 정체는 recall 이 아니라
none 110,701장 중 일부가 Scratch 로 흘러드는 것이라는 두 가지다.
그렇다면 분산을 줄이는 기법은 **누출이 어디로 가는지를 고르게 만들 뿐**이고,
평균을 올리려면 **누출 총량 자체를 줄여야** 한다.

**반증 가능한 예측:** 기각된 세 기법은 누출의 *범위*를 줄이면서 *평균*은 유지하거나
늘린다. translate 만 평균도 줄인다. 예측이 깨지면 이 설명은 틀린 것이고
"설명 없는 관측" 으로 그대로 남긴다.

학습이 없다. 이미 캐시된 로짓만 읽는다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate  # noqa: E402

POSTHOC = "result/posthoc"
SCRATCH = 7

GROUPS = [
    ("기준선 pad (증강 dihedral 만)", ["e8_pad", "e10_pad_s1", "e10_pad_s2"]),
    ("translate  (채택, 평균 +0.038)", ["e17_translate_s%d" % s for s in range(4)]),
    ("추가 증강 scale  (기각)", ["e17_scale_s0", "e17_scale_s1"]),
    ("추가 증강 dropout (기각)", ["e17_dropout_s0", "e17_dropout_s1"]),
    ("추가 증강 all     (기각)", ["e17_all_s0", "e17_all_s1"]),
    ("추가 증강 noise   (기각)", ["e17_noise_s0", "e17_noise_s1"]),
    ("자기지도 라벨만   (판정불가)", ["e12_ssl_lab_s0", "e12_ssl_lab_s1"]),
    ("자기지도 811K     (기각)", ["e12_ssl_s0", "e12_ssl_s1"]),
    ("EMA 원본 가중치", ["e19_ema_s%d" % s for s in range(3)]),
    ("EMA 가중치       (기각)", ["e19_ema_s%d_emaw" % s for s in range(3)]),
    ("밀도 채널 E20", ["e20_dens_s%d" % s for s in range(3)]),
]


def load(tag):
    p = f"{POSTHOC}/{tag}_logits.npz"
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d["test_logits"].argmax(1), d["test_y"]


rows = []
print("%-32s %3s %8s %8s %10s %10s %8s" % (
    "군", "n", "F1평균", "F1범위", "누출평균", "누출범위", "Scr범위"))
out = {}
for name, tags in GROUPS:
    f1s, leaks, scr = [], [], []
    for t in tags:
        r = load(t)
        if r is None:
            continue
        pred, y = r
        m = evaluate(y, pred, 9)
        f1s.append(m["macro_f1"])
        leaks.append(int(((y == 0) & (pred != 0)).sum()))
        scr.append(m["per_class_f1"][SCRATCH])
    if len(f1s) < 2:
        if f1s:
            print("%-32s %3d  (n<2, 범위 없음)  F1 %.4f  누출 %d"
                  % (name, len(f1s), f1s[0], leaks[0]))
        continue
    row = dict(n=len(f1s), f1_mean=float(np.mean(f1s)), f1_range=float(max(f1s) - min(f1s)),
               leak_mean=float(np.mean(leaks)), leak_range=float(max(leaks) - min(leaks)),
               scratch_range=float(max(scr) - min(scr)))
    out[name] = row
    rows.append((name, row))
    print("%-32s %3d %8.4f %8.4f %10.0f %10.0f %8.3f" % (
        name, row["n"], row["f1_mean"], row["f1_range"],
        row["leak_mean"], row["leak_range"], row["scratch_range"]))

base = out.get("기준선 pad (증강 dihedral 만)")
if base:
    print("\n== 기준선 대비 (기준선 F1 %.4f, 범위 %.4f, 누출 %.0f, 누출범위 %.0f) ==" % (
        base["f1_mean"], base["f1_range"], base["leak_mean"], base["leak_range"]))
    print("%-32s %10s %10s %12s %12s" % ("군", "F1 차이", "범위 배율", "누출 차이", "누출범위배율"))
    for name, r in rows:
        if name == "기준선 pad (증강 dihedral 만)":
            continue
        print("%-32s %+10.4f %9.2fx %+12.0f %11.2fx" % (
            name, r["f1_mean"] - base["f1_mean"],
            r["f1_range"] / max(base["f1_range"], 1e-9),
            r["leak_mean"] - base["leak_mean"],
            r["leak_range"] / max(base["leak_range"], 1e-9)))

    print("\n== 예측 확인 ==")
    print("  예측: 기각된 기법은 누출 *범위*는 줄이되 *평균*은 유지하거나 늘린다.")
    print("        translate 만 평균도 줄인다.")
    for name, r in rows:
        if name == "기준선 pad (증강 dihedral 만)":
            continue
        narrower = r["f1_range"] < base["f1_range"]
        less_leak = r["leak_mean"] < base["leak_mean"]
        better = r["f1_mean"] > base["f1_mean"]
        print("  %-32s 분산↓ %-3s 누출↓ %-3s 평균↑ %-3s  %s" % (
            name, "예" if narrower else "아니오", "예" if less_leak else "아니오",
            "예" if better else "아니오",
            "예측대로" if (narrower and better == less_leak) else "**예측과 다름**"))

os.makedirs("docs/research/variance_leakage/evidence", exist_ok=True)
json.dump(out, open("docs/research/variance_leakage/evidence/variance_vs_leakage.json", "w"),
          ensure_ascii=False, indent=2)
print("\n저장 -> docs/research/variance_leakage/evidence/variance_vs_leakage.json")
