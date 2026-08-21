"""자기지도 사전학습이 무엇을 바꿨는가. 기준선은 같은 설정의 무작위 초기화다."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")

NAMES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full"]

GROUPS = {
    "무작위 초기화 (기준선)": ["e8_pad", "e10_pad_s1", "e10_pad_s2"],
    "SSL 811K (미라벨 포함)": ["e12_ssl_s0", "e12_ssl_s1"],
    "SSL 172K (라벨만, 대조군)": ["e12_ssl_lab_s0", "e12_ssl_lab_s1"],
}


def load(tag):
    p = f"result/cls_baseline/{tag}_best_test.json"
    return json.load(open(p)) if os.path.exists(p) else None


print("%-28s %8s %9s %9s %9s" % ("조건", "n", "평균 mF1", "범위", "평균 acc"))
stats = {}
for name, tags in GROUPS.items():
    ds = [d for d in (load(t) for t in tags) if d]
    if not ds:
        print("%-28s %8s" % (name, "(아직 없음)")); continue
    f = [d["macro_f1"] for d in ds]
    a = [d["accuracy"] for d in ds]
    stats[name] = ds
    print("%-28s %8d %9.4f %9.4f %9.4f   %s" % (
        name, len(f), np.mean(f), max(f) - min(f), np.mean(a),
        " / ".join("%.4f" % v for v in f)))

if len(stats) >= 2:
    print("\n클래스별 F1 (조건별 평균)")
    hdr = "%-11s %8s" % ("클래스", "support")
    for name in stats:
        hdr += " %14s" % name[:14]
    print(hdr)
    for i, n in enumerate(NAMES):
        row = "%-11s %8d" % (n, list(stats.values())[0][0]["support"][str(i)])
        for name, ds in stats.items():
            row += " %14.3f" % np.mean([d["per_class_f1"][str(i)] for d in ds])
        print(row)

    print("\nnone -> 결함 누출 (110,701장 중, 조건별 평균)")
    for name, ds in stats.items():
        tot = []
        for d in ds:
            C = d["confusion"]
            row0 = C[0] if isinstance(C, list) else C["0"]
            tot.append(sum(row0[1:]))
        print("  %-28s %8.0f" % (name, np.mean(tot)))
