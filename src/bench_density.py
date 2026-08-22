"""E20 판정. 국소 밀도 입력 채널이 앙상블에 기여하는가.

반증 조건은 실행 전에 `docs/experiments/candidate/local_density_channels.md` 에 박았다.
여기서는 그 조건에 필요한 숫자만 뽑는다 — 판정 기준은 **앙상블 기여(leave-one-out)** 이고
단독 성능이 아니다(작업 원칙 6).

비교 풀은 `e17_translate_s0~s3` 4개 + 밀도 3개(+ 셔플 대조군)로 **고정**한다.
16차에서 풀에 강한 모델이 몰려 품질을 다양성으로 착각한 교란이 있었으므로,
같은 config 에서 나온 모델끼리만 넣어 품질을 맞춘다.
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

NAMES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
         "Random", "Scratch", "Near-full"]
POSTHOC = "result/posthoc"
SPLITS = "data/wm811k/cache/splits_v1.npz"

entries = json.loads(open("docs/experiments/ensemble_members.json", encoding="utf-8").read())
WANT = [e for e in entries
        if e["tag"].startswith("e17_translate_s") or e["tag"].startswith("e20_")
        or e["tag"].startswith("e21_")]

for e in WANT:
    p = f"{POSTHOC}/{e['tag']}_logits.npz"
    if os.path.exists(p):
        continue
    if not os.path.exists(e["ckpt"]):
        print(f"  [건너뜀] 체크포인트 없음 {e['tag']}")
        continue
    # **학습이 끝나기 전에 로짓을 굳히면 안 된다.** `{tag}_best.pt` 는 val 이 좋아질
    # 때마다 덮어써지므로 학습 중에도 파일이 존재한다. a4 평가가 남기는
    # `{tag}_best_test.json` 이 있어야 그 실행이 끝난 것이다.
    done = os.path.exists(f"result/cls_baseline/{e['tag']}_best_test.json")
    if not done and e["tag"].startswith(("e20_", "e21_")):
        print(f"  [건너뜀] 아직 학습 중 {e['tag']}")
        continue
    print(f"  [로짓 생성] {e['tag']}", flush=True)
    cache_logits(e["ckpt"], e["cache"], SPLITS, POSTHOC, e["tag"],
                 backbone=e.get("backbone", "resnet18"),
                 density_ks=tuple(e.get("density_ks", ())),
                 density_shuffle_seed=e.get("density_shuffle"),
                 line_ls=tuple(e.get("line_ls", ())),
                 batch=int(os.environ.get("LOGIT_BATCH", "512")))

P, ty = {}, None
for e in WANT:
    p = f"{POSTHOC}/{e['tag']}_logits.npz"
    if not os.path.exists(p):
        continue
    d = np.load(p)
    ty = d["test_y"] if ty is None else ty
    z = d["test_logits"].astype(np.float64)
    ex = np.exp(z - z.max(1, keepdims=True))
    P[e["tag"]] = ex / ex.sum(1, keepdims=True)

if ty is None:
    raise SystemExit("로짓이 하나도 없다")

CTRL = sorted(t for t in P if t.startswith("e17_translate_s"))
DENS = sorted(t for t in P if t.startswith("e20_dens_s"))
SHUF = sorted(t for t in P if t.startswith("e20_shuf_s"))
LINE = sorted(t for t in P if t.startswith("e21_"))
single = {t: float(evaluate(ty, P[t].argmax(1), 9)["macro_f1"]) for t in P}
leak = {t: int(((ty == 0) & (P[t].argmax(1) != 0)).sum()) for t in P}


def ens(tags):
    tags = [t for t in tags if t in P]
    return evaluate(ty, np.mean([P[t] for t in tags], 0).argmax(1), 9), len(tags)


def summarize(name, tags):
    if not tags:
        return None
    v = [single[t] for t in tags]
    print("  %-16s n=%d  평균 %.4f  범위 %.4f  [%s]  누출 평균 %.0f" % (
        name, len(v), np.mean(v), max(v) - min(v),
        " ".join("%.4f" % x for x in v), np.mean([leak[t] for t in tags])))
    return float(np.mean(v))


print("\n== 단독 성능 (판정 기준 아님, 기록용) ==")
print("  기준선: pad 3 seed 평균 0.7261, 범위 0.0495 (잡음 폭)")
summarize("대조군 translate", CTRL)
summarize("밀도 5,7", DENS)
summarize("셔플 대조군", SHUF)
summarize("선 필터 E21", LINE)

print("\n== 앙상블 ==")
for name, tags in [("대조군만", CTRL), ("밀도만", DENS),
                   ("대조군 + 밀도", CTRL + DENS),
                   ("대조군 + 셔플", CTRL + SHUF),
                   ("전부", CTRL + DENS + SHUF)]:
    if not [t for t in tags if t in P]:
        continue
    m, n = ens(tags)
    print("  %-16s n=%d  macro-F1 %.4f  acc %.4f  누출 %d" % (
        name, n, m["macro_f1"], m["accuracy"],
        int(((ty == 0) & (np.mean([P[t] for t in tags], 0).argmax(1) != 0)).sum())))

POOL = CTRL + DENS + SHUF + LINE
scores = {}
for k in (2, 3, 4):
    for c in itertools.combinations(POOL, k):
        scores[c] = float(evaluate(ty, np.mean([P[t] for t in c], 0).argmax(1), 9)["macro_f1"])

print("\n== leave-one-out (풀 %d개, k=2~4) — **이것이 판정 기준이다** ==" % len(POOL))
rows = []
for t in POOL:
    d = leave_one_out_delta(scores, t)
    rows.append((t, float(np.mean([v["delta"] for v in d.values()])), single[t]))
for t, loo, s in sorted(rows, key=lambda r: -r[1]):
    print("  %-22s 단독 %.4f  LOO %+.4f" % (t, s, loo))

loo_of = {t: v for t, v, _ in rows}
grp = {}
for name, tags in [("대조군 translate", CTRL), ("밀도 5,7", DENS), ("셔플 대조군", SHUF), ("선 필터 E21", LINE)]:
    if tags:
        grp[name] = float(np.mean([loo_of[t] for t in tags]))
        print("  [군 평균] %-18s LOO %+.4f" % (name, grp[name]))

print("\n== 사전 등록한 판정 ==")
if "밀도 5,7" in grp:
    d = grp["밀도 5,7"]
    ok_shuf = ("셔플 대조군" not in grp) or (d > grp["셔플 대조군"])
    if d >= 0.005 and ok_shuf:
        verdict = "채택"
    elif d < 0.003:
        verdict = "기각"
    else:
        verdict = "판정 보류"
    if d >= 0.005 and not ok_shuf:
        verdict = "기각 (셔플 대조군을 못 넘음 — 정보가 아니라 용량)"
    print("  밀도 군 LOO 평균 %+.4f  ->  **%s**" % (d, verdict))

print("\n== 클래스별 (군별 앙상블) ==")
cols = [("대조군", CTRL), ("밀도", DENS)] + ([("셔플", SHUF)] if SHUF else []) + ([("선필터", LINE)] if LINE else [])
mats = {n: ens(t)[0] for n, t in cols if t}
print("  %-11s %8s " % ("클래스", "support")
      + " ".join("%9s" % n for n in mats))
base = list(mats.values())[0]
for i, n in enumerate(NAMES):
    print("  %-11s %8d " % (n, base["support"][i])
          + " ".join("%9.3f" % m["per_class_f1"][i] for m in mats.values()))

print("\n== 크기 5분위별 none 오검출률 (극좌표 실패 모드 확인) ==")
meta = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
te = np.load(SPLITS)["test"]
size = meta["die_size"].astype(np.float64)[te]
none_mask = ty == 0
edges = np.percentile(size[none_mask], [0, 20, 40, 60, 80, 100])
edges[-1] += 1
print("  %-16s %s" % ("군", "  ".join("Q%d" % (i + 1) for i in range(5))))
for name, tags in cols:
    if not tags:
        continue
    pred = np.mean([P[t] for t in tags], 0).argmax(1)
    row = [float((pred[none_mask & (size >= edges[i]) & (size < edges[i + 1])] != 0).mean())
           for i in range(5)]
    print("  %-16s %s   |  Q1/Q5 = %.1f배" % (
        name, "  ".join("%.3f" % v for v in row), row[0] / max(row[-1], 1e-9)))
print("  (Q1 = 가장 작은 웨이퍼. 극좌표는 여기서 0.4~0.5 로 무너졌다)")

json.dump({"single": single, "leak": leak, "loo": loo_of, "group_loo": grp},
          open("result/posthoc/e20_density.json", "w"), ensure_ascii=False, indent=2)
print("\n저장 -> result/posthoc/e20_density.json")

# --- 전체 최고 기록이 바뀌는가 ---------------------------------------------------
#
# 위 비교는 품질을 맞춘 좁은 풀에서의 판정이다. 별개로 "지금 최선(TTA 18개, 0.7931)에
# 새 구성원을 넣으면 오르는가" 도 봐야 한다. 16차에서 구성원을 더 넣는다고 오르지
# 않는다는 것이 확인됐으므로(38개 < 18개) 기대는 낮다.

import glob as _glob  # noqa: E402

TTA = {}
for _p in sorted(_glob.glob(f"{POSTHOC}/*_tta_logits.npz")):
    _t = os.path.basename(_p).replace("_logits.npz", "")
    _d = np.load(_p)
    _z = _d["test_logits"].astype(np.float64)
    _e = np.exp(_z - _z.max(1, keepdims=True))
    TTA[_t] = _e / _e.sum(1, keepdims=True)

if TTA:
    BASE = [t for t in TTA if not t.startswith("e20_") and not t.startswith("e21_")]
    NEW = [t for t in TTA if t.startswith("e20_") or t.startswith("e21_")]
    print("\n== 전체 최고 기록 (TTA 판만) ==")
    for name, tags in [("현재 기준선", BASE), ("기준선 + 새 구성원", BASE + NEW),
                       ("새 구성원만", NEW)]:
        tags = [t for t in tags if t in TTA]
        if not tags:
            continue
        pm = np.mean([TTA[t] for t in tags], 0)
        m = evaluate(ty, pm.argmax(1), 9)
        print("  %-20s n=%2d  macro-F1 %.4f  acc %.4f  누출 %d" % (
            name, len(tags), m["macro_f1"], m["accuracy"],
            int(((ty == 0) & (pm.argmax(1) != 0)).sum())))
    print("  (새 구성원의 TTA 로짓이 없으면 위 줄이 안 나온다 — a17_tta 를 먼저 돌려라)")

    print("\n== 밀도 모델에 TTA 를 걸면 ==")
    for t in sorted(DENS + SHUF + LINE):
        tt = t + "_tta"
        if tt in TTA:
            f0 = single[t]
            f1 = float(evaluate(ty, TTA[tt].argmax(1), 9)["macro_f1"])
            print("  %-16s 기본 %.4f -> TTA %.4f  (%+.4f)" % (t, f0, f1, f1 - f0))
