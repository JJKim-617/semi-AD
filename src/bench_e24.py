"""E24 판정 — 제로 패딩이 절대 위치의 출처인가 (**개입 시험**).

반증 조건은 실행 전에 `docs/experiments/candidate/padding_position_leak.md` 에 박았다.

- **P1**: 팔 A 의 이동 민감도 < 0.030 (대조군 0.0480). 0.040 초과면 출처가 아니다.
- **P2 (핵심 갈래)**: 팔 A 의 누출 < 1,600 (대조군 1,925).
  **P1 은 통과하는데 P2 가 실패하면 기전이 인과가 아니다 — 내 결론을 내가 반증한 것이다.**
- **P3 (판정 기준)**: 품질 맞춘 풀에서 LOO 군 평균 >= +0.003. E20~E22 와 같은 선.
- **P4**: 팔 B 가 0.7712 를 넘으면 translate 와 다른 것을 지우는 것이다.

이동 민감도는 `src/diag_shift_sensitivity.py` 가 매니페스트를 읽어 따로 낸다.
여기서는 단독 성능, 누출, LOO 를 낸다.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate  # noqa: E402
from a6_perclass_offset import cache_logits  # noqa: E402
from bench_ensemble import leave_one_out_delta  # noqa: E402

POSTHOC = "result/posthoc"
CACHE = "data/wm811k/cache/wm811k_64pad.npz"
SPLITS = "data/wm811k/cache/splits_v1.npz"
SEEDS = [0, 1, 2]

# 사전 등록된 기준선. 문서의 값을 박아 두어 판정 때 옮기지 못하게 한다.
CTRL_PLAIN_F1, CTRL_PLAIN_LEAK, CTRL_PLAIN_SHIFT = 0.7396, 1925, 0.0480
CTRL_TR_F1, CTRL_TR_LEAK = 0.7645, 1190
P1_LINE, P1_REFUTE = 0.030, 0.040
P2_LINE = 1600
P3_LINE = 0.003
P4_LINE = CTRL_TR_F1 + 0.0067      # 0.7712


def load(tags):
    P, ty = {}, None
    for t in tags:
        f = f"{POSTHOC}/{t}_logits.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f)
        ty = d["test_y"] if ty is None else ty
        z = d["test_logits"].astype(np.float32)
        e = np.exp(z - z.max(1, keepdims=True))
        P[t] = e / e.sum(1, keepdims=True)
    return P, ty


def main():
    arms = {"A (reflect 단독)": [], "B (reflect+translate)": []}
    for arm, key in (("A (reflect 단독)", "pad"), ("B (reflect+translate)", "padtr")):
        for s in SEEDS:
            tag = f"e24_{key}_s{s}"
            if not os.path.exists(f"result/cls_baseline/{tag}_best_test.json"):
                print(f"  [아직] {tag}")
                continue
            if not os.path.exists(f"{POSTHOC}/{tag}_logits.npz"):
                print(f"  [로짓 생성] {tag}", flush=True)
                # **padding_mode 를 반드시 넘긴다.** 빼면 zeros 모델로 실려 조용히 틀린다.
                cache_logits(f"result/cls_baseline/{tag}_best.pt", CACHE, SPLITS,
                             POSTHOC, tag, padding_mode="reflect")
            arms[arm].append(tag)

    CTRL_PLAIN = ["e8_pad", "e10_pad_s2"]
    CTRL_TR = [f"e17_translate_s{i}" for i in range(4)]
    allt = CTRL_PLAIN + CTRL_TR + arms["A (reflect 단독)"] + arms["B (reflect+translate)"]
    P, ty = load(allt)
    if not (arms["A (reflect 단독)"] or arms["B (reflect+translate)"]):
        raise SystemExit("E24 로짓이 하나도 없다")

    single = {t: float(evaluate(ty, P[t].argmax(1), 9)["macro_f1"]) for t in P}
    leak = {t: int(((ty == 0) & (P[t].argmax(1) != 0)).sum()) for t in P}

    def show(name, tags):
        tags = [t for t in tags if t in P]
        if not tags:
            return None, None
        v = np.array([single[t] for t in tags])
        lk = np.array([leak[t] for t in tags])
        print("  %-26s n=%d  단독 %.4f  범위 %.4f  누출 %6.0f  [%s]" % (
            name, len(tags), v.mean(), v.max() - v.min(), lk.mean(),
            " ".join("%.4f" % x for x in sorted(v, reverse=True))))
        return float(v.mean()), float(lk.mean())

    print("\n== 단독 성능과 누출 ==")
    show("대조군 무증강 resnet18", CTRL_PLAIN)
    show("대조군 translate", CTRL_TR)
    fa, la = show("**팔 A (reflect 단독)**", arms["A (reflect 단독)"])
    fb, lb = show("**팔 B (reflect+translate)**", arms["B (reflect+translate)"])
    for t in arms["A (reflect 단독)"] + arms["B (reflect+translate)"]:
        print("    %-24s 단독 %.4f  누출 %5d" % (t, single[t], leak[t]))

    print("\n== P2 — **핵심 갈래**. 민감도가 내려가면 누출도 내려가는가 ==")
    if la is not None:
        n = len(arms["A (reflect 단독)"])
        verdict = "예측 맞음" if la < P2_LINE else "**예측 반증**"
        print("  팔 A 누출 %.0f (대조군 %d, 기준선 %d 미만)  ->  %s  [n=%d]" % (
            la, CTRL_PLAIN_LEAK, P2_LINE, verdict, n))
        if n < 3:
            print("  **n<3 이므로 판정하지 않는다.** 이 프로젝트는 n=2 로 세 번 뒤집혔다.")
        print("  P1(이동 민감도)은 src/diag_shift_sensitivity.py 로 따로 낸다.")
        print("  **P1 통과 + P2 반증이면 기전이 인과가 아니다 — 정정 절을 연다.**")

    print("\n== P3 — 판정 기준. 품질 맞춘 풀에서 leave-one-out (k=2~4) ==")
    for arm, ctrl, cname in (("A (reflect 단독)", CTRL_PLAIN, "무증강 대조군"),
                             ("B (reflect+translate)", CTRL_TR, "translate 대조군")):
        sub = arms[arm]
        if len(sub) < 2:
            print("  %-24s (구성원 %d개 — 풀을 못 만든다)" % (arm, len(sub)))
            continue
        pool = [t for t in ctrl + sub if t in P]
        scores = {}
        for k in (2, 3, 4):
            if k > len(pool):
                break
            for c in itertools.combinations(pool, k):
                scores[c] = float(evaluate(
                    ty, np.mean([P[t] for t in c], 0).argmax(1), 9)["macro_f1"])
        loo = {t: float(np.mean([v["delta"] for v in
                                 leave_one_out_delta(scores, t).values()]))
               for t in pool}
        g = float(np.mean([loo[t] for t in sub]))
        c = float(np.mean([loo[t] for t in ctrl if t in loo]))
        print("  %-24s 풀 %d개 (%s)" % (arm, len(pool), cname))
        for t in sorted(pool, key=lambda x: -loo[x]):
            print("      %-24s 단독 %.4f  LOO %+.4f" % (t, single[t], loo[t]))
        print("      [군 평균] E24 %+.4f   대조군 %+.4f  ->  **%s**" % (
            g, c, "채택" if g >= P3_LINE else "기각"))

    print("\n== P4 — translate 와 겹치는가 ==")
    if fb is not None:
        print("  팔 B 단독 %.4f  (translate 대조군 %.4f, 기준선 %.4f 초과여야 '다른 것')" % (
            fb, CTRL_TR_F1, P4_LINE))
        print("  ->  **%s**" % ("겹치지 않는다 (예측 반증)" if fb > P4_LINE
                                else "겹친다 (예측대로)"))

    print("\n== 앙상블 ==")
    for name, tags in [("대조군 무증강 2", CTRL_PLAIN), ("팔 A", arms["A (reflect 단독)"]),
                       ("대조군 translate 4", CTRL_TR), ("팔 B", arms["B (reflect+translate)"]),
                       ("팔 A + 팔 B", arms["A (reflect 단독)"] + arms["B (reflect+translate)"]),
                       ("translate 4 + 팔 B", CTRL_TR + arms["B (reflect+translate)"])]:
        tags = [t for t in tags if t in P]
        if not tags:
            continue
        pm = np.mean([P[t] for t in tags], 0)
        m = evaluate(ty, pm.argmax(1), 9)
        print("  %-24s n=%d  macro-F1 %.4f  acc %.4f  누출 %5d" % (
            name, len(tags), m["macro_f1"], m["accuracy"],
            int(((ty == 0) & (pm.argmax(1) != 0)).sum())))

    json.dump(dict(single=single, leak=leak, arms=arms),
              open("result/posthoc/e24_padding.json", "w"),
              ensure_ascii=False, indent=2)
    print("\n저장 -> result/posthoc/e24_padding.json")


if __name__ == "__main__":
    main()
