"""구성원끼리 얼마나 다른가 — E20 이 왜 단독은 오르고 앙상블은 안 오르는지 (학습 없음).

## 물음

E20(국소 밀도 채널)은 **단독 성능이 대조군 4 seed 를 전부 넘는데 앙상블은 같은 자리**다
(LOO 군 평균 +0.0002). 실행 전에 적어 둔 두 가설이 모두 반증됐고
(A: 경계를 고친다 -> 누출 안 줄음, B: 표현 다양성 -> LOO 0), 셋째가 남았다.

> **C: 밀도 채널은 입력의 결정론적 함수다. seed 가 달라도 모든 모델이 똑같은
> 추가 특징을 계산하므로 개별 바닥은 올라가지만 서로 상관이 줄지 않는다.**

C 는 직접 잴 수 있다. **구성원끼리의 예측 불일치와 오류 상관**을 군별로 비교하면 된다.
C 가 맞다면 **밀도끼리가 translate 끼리보다 더 닮아야** 한다.

학습이 없다. 캐시된 로짓만 읽는다.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")

POST = "result/posthoc"
OUT = "docs/research/local_density/evidence"


def main() -> None:
    tags = [f"e17_translate_s{i}" for i in range(4)] \
        + [f"e20_dens_s{i}" for i in range(3)] \
        + ["e20_shuf_s0"] + [f"e21_line_s{i}" for i in range(3)]
    P, ty = {}, None
    for t in tags:
        f = f"{POST}/{t}_logits.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f)
        ty = d["test_y"] if ty is None else ty
        z = d["test_logits"].astype(np.float64)
        e = np.exp(z - z.max(1, keepdims=True))
        P[t] = e / e.sum(1, keepdims=True)
    if len(P) < 2:
        raise SystemExit("로짓이 2개 미만이다")
    pred = {t: P[t].argmax(1) for t in P}
    err = {t: (pred[t] != ty) for t in P}

    def dis(a, b):
        return float((pred[a] != pred[b]).mean())

    def errcorr(a, b):
        return float(np.corrcoef(err[a].astype(float), err[b].astype(float))[0, 1])

    groups = {
        "translate": [t for t in P if t.startswith("e17_translate")],
        "밀도": [t for t in P if t.startswith("e20_dens")],
        "셔플": [t for t in P if t.startswith("e20_shuf")],
        "선 필터": [t for t in P if t.startswith("e21_line")],
    }
    groups = {k: v for k, v in groups.items() if v}

    res = {"n_errors": {t: int(err[t].sum()) for t in P}, "pairs": {}}
    print("구성원 사이의 다양성 — 불일치가 클수록, 오류 상관이 낮을수록 다양하다\n")
    print("  %-24s %3s %14s %12s" % ("짝", "n", "예측 불일치", "오류 상관"))

    def report(pairs, name):
        if not pairs:
            return
        d = [dis(a, b) for a, b in pairs]
        c = [errcorr(a, b) for a, b in pairs]
        res["pairs"][name] = {"n": len(pairs), "disagree": float(np.mean(d)),
                              "err_corr": float(np.mean(c))}
        print("  %-24s %3d %14.4f %12.4f" % (name, len(pairs), np.mean(d), np.mean(c)))

    for g, ts in groups.items():
        report(list(itertools.combinations(ts, 2)), f"{g} 끼리")
    for a, b in itertools.combinations(groups, 2):
        report([(x, y) for x in groups[a] for y in groups[b]], f"{a} 대 {b}")

    print("\n  각 모델의 오류 개수:")
    for t, n in sorted(res["n_errors"].items()):
        print("    %-24s %d" % (t, n))

    p_tr, p_de = res["pairs"].get("translate 끼리"), res["pairs"].get("밀도 끼리")
    if p_tr and p_de:
        print("\n== 가설 C 판정 ==")
        print("  밀도끼리 오류 상관 %.4f  대  translate 끼리 %.4f"
              % (p_de["err_corr"], p_tr["err_corr"]))
        print("  -> %s" % ("**C 지지: 밀도 모델끼리 더 닮았다.** 개별은 좋아지는데 "
                           "평균낼 것이 안 생긴다."
                           if p_de["err_corr"] > p_tr["err_corr"]
                           else "**C 반증: 밀도 모델이 오히려 더 다양하다.** 다른 설명이 필요하다."))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "member_diversity.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\n저장 -> %s/member_diversity.json" % OUT)


if __name__ == "__main__":
    main()
