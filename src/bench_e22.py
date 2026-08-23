"""E22 판정 — 백본 계열 x translate 증강.

반증 조건은 실행 전에 `docs/experiments/candidate/backbone_diversity.md` 에 박았다.
여기서는 그 조건에 필요한 숫자만 뽑는다.

- **H1 (전이)**: 백본 3종 x 3 seed 단독 평균이 0.7241 -> +0.019 이상. 판정 기준 아님.
- **H2 (다양성 유지)**: `e17_translate_*` 와의 오류 상관이 0.75 미만
  (백본 0.6238 과 seed 0.8285 의 중간).
- **H3 (판정 기준)**: 앙상블 기여(LOO) 군 평균 >= +0.003.
  E20/E21 을 기각한 것과 **같은 선**이고 **같은 방법**이다 — 품질을 맞춘 좁은 풀에서
  k=2~4 조합 전수의 leave-one-out. E20 의 풀이 (translate 대조군 + 밀도) 였으므로
  여기서는 (translate 대조군 + E22) 다. **정확히 평행한 설계다.**
- **누출 예측**: E22 군 평균 누출 < 1,600 (translate 대역 1,190 대 백본 대역 2,844).
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
BACKBONES = ["shufflenet_v2", "mobilenet_v3", "efficientnet_b0"]
SEEDS = [0, 1, 2]

# 사전 등록된 기준선. 문서에 적힌 값을 그대로 박아 두어 판정 때 옮기지 못하게 한다.
BASE_BACKBONE_SINGLE = 0.7241     # e18 백본 3종 x 2 seed
BASE_TRANSLATE_GAIN = 0.0384      # resnet18 에서 translate 의 이득
H1_LINE = BASE_BACKBONE_SINGLE + BASE_TRANSLATE_GAIN / 2   # 0.7433
H2_LINE = 0.75
H3_LINE = 0.003
LEAK_LINE = 1600


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
    e22 = []
    for bb in BACKBONES:
        for s in SEEDS:
            tag = f"e22_{bb}_tr_s{s}"
            ck = f"result/cls_baseline/{tag}_best.pt"
            # 학습이 끝났다는 증거는 a4 가 남기는 *_best_test.json 이다.
            # best.pt 는 val 이 좋아질 때마다 덮어써지므로 존재만으로는 부족하다.
            if not os.path.exists(f"result/cls_baseline/{tag}_best_test.json"):
                print(f"  [아직] {tag}")
                continue
            if not os.path.exists(f"{POSTHOC}/{tag}_logits.npz"):
                print(f"  [로짓 생성] {tag}", flush=True)
                cache_logits(ck, CACHE, SPLITS, POSTHOC, tag, backbone=bb)
            e22.append(tag)

    CTRL = [f"e17_translate_s{i}" for i in range(4)]
    BB18 = [f"e18_{b}_s{s}" for b in BACKBONES for s in (0, 1)]
    RES = ["e8_pad", "e10_pad_s2"]          # 건강한 대조군 (e10_pad_s1 은 붕괴)
    P, ty = load(CTRL + BB18 + RES + e22)
    e22 = [t for t in e22 if t in P]
    if not e22:
        raise SystemExit("E22 로짓이 하나도 없다")

    single = {t: float(evaluate(ty, P[t].argmax(1), 9)["macro_f1"]) for t in P}
    leak = {t: int(((ty == 0) & (P[t].argmax(1) != 0)).sum()) for t in P}
    pred = {t: P[t].argmax(1) for t in P}
    err = {t: (pred[t] != ty) for t in P}

    def show(name, tags):
        tags = [t for t in tags if t in P]
        if not tags:
            return None
        v = np.array([single[t] for t in tags])
        lk = np.array([leak[t] for t in tags])
        print("  %-30s n=%d  단독 %.4f  범위 %.4f  [%s]  누출 %6.0f" % (
            name, len(tags), v.mean(), v.max() - v.min(),
            " ".join("%.4f" % x for x in sorted(v, reverse=True)), lk.mean()))
        return v.mean()

    print("\n== H1 — 단독 성능 (판정 기준 아님) ==")
    show("건강한 대조군 resnet18 (증강X)", RES)
    show("resnet18 + translate", CTRL)
    show("백본 3종, translate 없음 (E18)", BB18)
    m22 = show("**E22 백본 3종 + translate**", e22)
    for bb in BACKBONES:
        show(f"    {bb} + translate", [t for t in e22 if bb in t])
        show(f"    {bb} (E18, translate 없음)", [t for t in BB18 if bb in t])
    gain = m22 - BASE_BACKBONE_SINGLE
    print("  사전 등록: 이득 >= %+.4f 여야 H1 통과 (단독 평균 %.4f)" % (
        BASE_TRANSLATE_GAIN / 2, H1_LINE))
    print("  실제 이득 %+.4f  ->  **H1 %s**" % (gain, "통과" if gain >= BASE_TRANSLATE_GAIN / 2 else "기각"))

    print("\n== 사전 등록한 누출 예측 (translate 대역 1,190 대 백본 대역 2,844) ==")
    lk = np.array([leak[t] for t in e22])
    print("  E22 누출 군 평균 %.0f  범위 %d~%d  (기준선 %d 미만)  ->  **%s**" % (
        lk.mean(), lk.min(), lk.max(), LEAK_LINE,
        "예측 맞음" if lk.mean() < LEAK_LINE else "예측 반증"))
    for t in e22:
        print("    %-30s 단독 %.4f  누출 %5d" % (t, single[t], leak[t]))

    print("\n== H2 — 다양성. 오류 상관 (낮을수록 다양) ==")
    print("  17차 기준: 백본 계열 0.6238 / seed 만 0.8285 / 채널 추가 0.8287")

    def errcorr(a, b):
        return float(np.corrcoef(err[a].astype(np.float32),
                                 err[b].astype(np.float32))[0, 1])

    def pairs(A, B, same=False, matched=True):
        out = []
        it = itertools.combinations(A, 2) if same else itertools.product(A, B)
        for a, b in it:
            if a == b:
                continue
            # 17차와 같은 자: 오류 개수가 400 이내인 짝만. 상관이 오류율에 교란된다.
            if matched and abs(int(err[a].sum()) - int(err[b].sum())) > 400:
                continue
            out.append(errcorr(a, b))
        return np.array(out)

    rows = [
        ("E22 x e17_translate (백본이 다름)", pairs(e22, CTRL)),
        ("E22 끼리, 백본이 다름", np.array(
            [errcorr(a, b) for a, b in itertools.combinations(e22, 2)
             if a.split("_tr_")[0] != b.split("_tr_")[0]
             and abs(int(err[a].sum()) - int(err[b].sum())) <= 400])),
        ("E22 끼리, 같은 백본 seed 만 다름", np.array(
            [errcorr(a, b) for a, b in itertools.combinations(e22, 2)
             if a.split("_tr_")[0] == b.split("_tr_")[0]
             and abs(int(err[a].sum()) - int(err[b].sum())) <= 400])),
        ("e17_translate 끼리 (seed 만)", pairs(CTRL, CTRL, same=True)),
        ("E18 백본 x e17_translate (참고)", pairs(BB18, CTRL)),
    ]
    for name, v in rows:
        if len(v) == 0:
            print("  %-38s (짝 없음)" % name)
            continue
        print("  %-38s n=%2d  평균 %.4f  범위 %.4f~%.4f" % (
            name, len(v), v.mean(), v.min(), v.max()))
    cross = rows[0][1]
    if len(cross):
        print("  사전 등록: E22 x translate 상관 < %.2f 여야 H2 통과" % H2_LINE)
        print("  실제 %.4f  ->  **H2 %s**" % (
            cross.mean(), "통과" if cross.mean() < H2_LINE else "기각"))

    # 17차의 머릿수(백본 0.6238 대 seed 0.8285)는 **오류 개수 ±400 필터**로 품질을
    # 맞췄다고 했다. 그런데 그 필터는 넓다 — 오류 3,300개 근처에서 400 은 12% 다.
    # E22 는 백본을 바꾸면서 품질까지 translate 수준으로 올린 첫 표본이므로,
    # **품질이 같아졌을 때도 백본이 다양성을 만드는지**를 여기서 처음 가를 수 있다.
    print("\n== 교란 점검 — 오류 상관이 품질에 얼마나 끌려가나 ==")
    allt = [t for t in P if single[t] > 0.70]
    pr = []
    for a_, b_ in itertools.combinations(allt, 2):
        pr.append((0.5 * (single[a_] + single[b_]), errcorr(a_, b_),
                   abs(int(err[a_].sum()) - int(err[b_].sum()))))
    if len(pr) > 10:
        q = np.array(pr)
        print("  짝 %d개. corr(짝 평균 macro-F1, 오류 상관) = %+.3f" % (
            len(q), float(np.corrcoef(q[:, 0], q[:, 1])[0, 1])))
        print("  (±400 필터를 통과한 짝만 보면 %+.3f)" % (
            float(np.corrcoef(q[q[:, 2] <= 400][:, 0], q[q[:, 2] <= 400][:, 1])[0, 1])
            if (q[:, 2] <= 400).sum() > 5 else float("nan")))
        edges = np.quantile(q[:, 0], [0, 1 / 3, 2 / 3, 1.0])
        print("  %-22s %5s %10s" % ("짝 평균 품질", "n", "오류 상관"))
        for i in range(3):
            m = (q[:, 0] >= edges[i]) & (q[:, 0] <= edges[i + 1])
            print("  %.4f ~ %.4f      %5d %10.4f" % (
                edges[i], edges[i + 1], m.sum(), q[m, 1].mean()))
        print("  **품질이 오르면 오류 상관도 오른다면, 17차의 0.6238 은 백본이 아니라")
        print("    백본 모델이 약했던 것을 잰 값이다.** E22 가 그것을 가른다.")

        # 옳은 처리는 필터가 아니라 **회귀**다. 오류 상관을 짝 평균 품질로 회귀한 뒤
        # 잔차를 짝 종류별로 비교한다. 잔차가 음수면 "품질이 설명하는 것보다 더 다양하다".
        def arch_of(t):
            if t.startswith("e22_"):
                return t.split("_tr_")[0][len("e22_"):]
            if t.startswith("e18_"):
                return t[len("e18_"):].rsplit("_s", 1)[0]
            return "resnet18"

        def aug_of(t):
            return "translate" if (t.startswith("e17_translate")
                                   or t.startswith("e22_")) else "none"

        def kind(a_, b_):
            # 17차의 구분을 그대로 재현한다: seed 만 다른 짝 / 백본이 다른 짝.
            if arch_of(a_) != arch_of(b_):
                return "백본이 다름"
            return "seed 만 다름" if aug_of(a_) == aug_of(b_) else "증강만 다름"

        names, qual, ec = [], [], []
        for a_, b_ in itertools.combinations(allt, 2):
            names.append(kind(a_, b_))
            qual.append(0.5 * (single[a_] + single[b_]))
            ec.append(errcorr(a_, b_))
        qual, ec = np.array(qual), np.array(ec)
        fit = np.polyfit(qual, ec, 1)
        resid = ec - np.polyval(fit, qual)
        print("\n  품질로 회귀한 뒤의 잔차 (음수 = 품질이 설명하는 것보다 더 다양하다)")
        print("  %-14s %5s %12s %12s" % ("짝 종류", "n", "오류 상관", "잔차"))
        for k in ("seed 만 다름", "증강만 다름", "백본이 다름"):
            m = np.array([n == k for n in names])
            if m.sum():
                print("  %-14s %5d %12.4f %+12.4f" % (
                    k, m.sum(), ec[m].mean(), resid[m].mean()))
        gap = (np.array([n == "백본이 다름" for n in names]),
               np.array([n == "seed 만 다름" for n in names]))
        if gap[0].sum() and gap[1].sum():
            print("  품질 통제 후 백본 효과 = %+.4f  (17차가 보고한 날것 차이는 -0.2047)" % (
                resid[gap[0]].mean() - resid[gap[1]].mean()))

    print("\n== H3 — **판정 기준**. 품질을 맞춘 풀에서 leave-one-out (k=2~4) ==")
    POOL = CTRL + e22
    print("  풀 %d개 = translate 대조군 %d + E22 %d" % (len(POOL), len(CTRL), len(e22)))
    scores = {}
    for k in (2, 3, 4):
        for c in itertools.combinations(POOL, k):
            scores[c] = float(evaluate(
                ty, np.mean([P[t] for t in c], 0).argmax(1), 9)["macro_f1"])
    loo = {}
    for t in POOL:
        d = leave_one_out_delta(scores, t)
        loo[t] = float(np.mean([v["delta"] for v in d.values()]))
    for t in sorted(POOL, key=lambda x: -loo[x]):
        print("    %-30s 단독 %.4f  LOO %+.4f" % (t, single[t], loo[t]))
    g22 = float(np.mean([loo[t] for t in e22]))
    gct = float(np.mean([loo[t] for t in CTRL]))
    print("  [군 평균] E22 %+.4f   translate 대조군 %+.4f" % (g22, gct))
    verdict = "채택" if g22 >= H3_LINE else "기각"
    print("  사전 등록: 군 평균 LOO >= %+.4f  ->  **H3 %s**" % (H3_LINE, verdict))

    # 사전 등록한 한계 3: 9개를 한꺼번에 넣으면 서로가 서로의 LOO 를 깎는다.
    # 백본당 1 seed 만 넣은 풀에서도 재고, seed 집합 셋을 전부 돌려 사후 선택을 막는다.
    print("\n== H3 (b) — 백본당 1 seed 만 넣은 풀 (사전 등록한 한계 3) ==")
    subs = {}
    for s in SEEDS:
        sel = [t for t in e22 if t.endswith(f"_s{s}")]
        if len(sel) < 2:
            continue
        pool2 = CTRL + sel
        sc2 = {}
        for k in (2, 3, 4):
            for c in itertools.combinations(pool2, k):
                sc2[c] = float(evaluate(
                    ty, np.mean([P[t] for t in c], 0).argmax(1), 9)["macro_f1"])
        l2 = {t: float(np.mean([v["delta"] for v in
                                leave_one_out_delta(sc2, t).values()])) for t in pool2}
        g = float(np.mean([l2[t] for t in sel]))
        c = float(np.mean([l2[t] for t in CTRL]))
        subs[s] = dict(e22=g, ctrl=c, members={t: l2[t] for t in sel})
        print("  seed %d 집합 (%d개): E22 군 LOO %+.4f   대조군 %+.4f" % (
            s, len(sel), g, c))
    if subs:
        mb = float(np.mean([v["e22"] for v in subs.values()]))
        print("  [seed 집합 평균] E22 %+.4f  ->  **H3(b) %s**" % (
            mb, "채택" if mb >= H3_LINE else "기각"))

    print("\n== 참고 — 백본별 LOO ==")
    for bb in BACKBONES:
        sub = [t for t in e22 if bb in t]
        if sub:
            print("    %-20s %+.4f" % (bb, float(np.mean([loo[t] for t in sub]))))

    print("\n== 앙상블 (원본 로짓만. TTA 판은 학습 큐가 끝난 뒤) ==")
    for name, tags in [("translate 대조군 4", CTRL), ("E22 9", e22),
                       ("대조군 + E22", CTRL + e22),
                       ("E18 백본 6", BB18), ("대조군 + E18 백본", CTRL + BB18),
                       ("대조군 + E22 + E18", CTRL + e22 + BB18)]:
        tags = [t for t in tags if t in P]
        if not tags:
            continue
        pm = np.mean([P[t] for t in tags], 0)
        m = evaluate(ty, pm.argmax(1), 9)
        print("  %-24s n=%2d  macro-F1 %.4f  acc %.4f  누출 %5d" % (
            name, len(tags), m["macro_f1"], m["accuracy"],
            int(((ty == 0) & (pm.argmax(1) != 0)).sum())))

    # --- H4. 풀 재구성 --------------------------------------------------------
    # 사전 등록: 0.7931 + 0.002(풀 구성 잡음) = 0.7951 을 넘어야 "기록이 올랐다" 고 쓴다.
    import glob as _glob
    TTA, tty = {}, None
    for _p in sorted(_glob.glob(f"{POSTHOC}/*_tta_logits.npz")):
        _t = os.path.basename(_p).replace("_logits.npz", "")
        _d = np.load(_p)
        tty = _d["test_y"] if tty is None else tty
        _z = _d["test_logits"].astype(np.float32)
        _e = np.exp(_z - _z.max(1, keepdims=True))
        TTA[_t] = _e / _e.sum(1, keepdims=True)
    e22t = [t + "_tta" for t in e22 if t + "_tta" in TTA]
    if not e22t:
        print("\n== H4 — E22 의 TTA 로짓이 아직 없다 (a17_tta 를 먼저 돌려라) ==")
    else:
        print("\n== H4 — 풀 재구성 (TTA 판). 사전 등록선 0.7951 ==")
        # 17차의 '현재 기준선' 정의 그대로: e20/e21 을 뺀 TTA 판 전부 = 18개
        BASE = [t for t in TTA if not t.startswith(("e20_", "e21_", "e22_", "e23_"))]
        BB_T = [t for t in BASE if t.startswith("e18_")]
        BASE_NOBB = [t for t in BASE if not t.startswith("e18_")]

        def ens_t(tags, name):
            tags = [t for t in tags if t in TTA]
            if not tags:
                return
            pm = np.mean([TTA[t] for t in tags], 0)
            m = evaluate(tty, pm.argmax(1), 9)
            mark = " <- 사전 등록선 넘음" if m["macro_f1"] >= 0.7951 else ""
            print("  %-40s n=%2d  macro-F1 %.4f  acc %.4f  누출 %5d%s" % (
                name, len(tags), m["macro_f1"], m["accuracy"],
                int(((tty == 0) & (pm.argmax(1) != 0)).sum()), mark))
            return float(m["macro_f1"])

        ens_t(BASE, "현재 기준선 (17차의 TTA 18개)")
        ens_t(BASE + e22t, "기준선 + E22 전부")
        ens_t(BASE_NOBB + e22t, "**E18 백본을 E22 로 교체**")
        ens_t(e22t, "E22 만")
        ens_t([t for t in BASE if t.startswith("e17_translate")] + e22t,
              "translate 4 + E22")

    json.dump(dict(single=single, leak=leak, loo=loo, group_loo_e22=g22,
                   group_loo_ctrl=gct, e22=e22, loo_by_seedset=subs,
                   errcorr={n: (float(v.mean()) if len(v) else None)
                            for n, v in rows}),
              open("result/posthoc/e22_backbone_translate.json", "w"),
              ensure_ascii=False, indent=2)
    print("\n저장 -> result/posthoc/e22_backbone_translate.json")


if __name__ == "__main__":
    main()
