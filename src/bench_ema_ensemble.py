"""EMA 가 앙상블 구성원으로 값어치가 있는가. 그리고 현재 최선 갱신 여부."""
from __future__ import annotations
import itertools, json, os, sys
import numpy as np
sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate
from a6_perclass_offset import cache_logits
from bench_ensemble import leave_one_out_delta

NAMES=["none","Center","Donut","Edge-Loc","Edge-Ring","Loc","Random","Scratch","Near-full"]
POSTHOC="result/posthoc"
entries=json.loads(open("docs/experiments/ensemble_members.json",encoding="utf-8").read())
for e in entries:
    p=f"{POSTHOC}/{e['tag']}_logits.npz"
    if not os.path.exists(p) and os.path.exists(e["ckpt"]):
        print(f"  [로짓] {e['tag']}", flush=True)
        cache_logits(e["ckpt"], e["cache"], "data/wm811k/cache/splits_v1.npz",
                     POSTHOC, e["tag"], backbone=e.get("backbone","resnet18"))
P,ty={},None
for e in entries:
    p=f"{POSTHOC}/{e['tag']}_logits.npz"
    if not os.path.exists(p): continue
    d=np.load(p); ty=d["test_y"] if ty is None else ty
    z=d["test_logits"].astype(np.float64); ex=np.exp(z-z.max(1,keepdims=True))
    P[e["tag"]]=ex/ex.sum(1,keepdims=True)
single={t: float(evaluate(ty,P[t].argmax(1),9)["macro_f1"]) for t in P}
def ens(tags):
    tags=[t for t in tags if t in P]
    return evaluate(ty,np.mean([P[t] for t in tags],0).argmax(1),9), len(tags)

TRANS=sorted(t for t in P if t.startswith("e17_translate"))
BB=sorted(t for t in P if t.startswith("e18_"))
EMAW=sorted(t for t in P if t.endswith("_emaw"))
EMARAW=sorted(t for t in P if t.startswith("e19_") and not t.endswith("_emaw"))
PREV11=["e8_pad","e10_pad_s1","e10_pad_s2","e7_ce_long_best","e6_focal_long_best",
        "e6_focal_long_s1_best","e7_ce_long80_best","e8_res96","e11_size","e11_shuffle",
        "e15b_polar_s0"]

print("\nE19 단독")
for s in (0,1,2):
    r,e_=f"e19_ema_s{s}",f"e19_ema_s{s}_emaw"
    if r in single and e_ in single:
        print("  seed %d  원본 %.4f  EMA %.4f  차이 %+.4f" % (s,single[r],single[e_],single[e_]-single[r]))
if EMARAW and EMAW:
    rw=[single[t] for t in EMARAW]; ew=[single[t] for t in EMAW]
    print("  평균    원본 %.4f  EMA %.4f  (%+.4f)" % (np.mean(rw),np.mean(ew),np.mean(ew)-np.mean(rw)))
    print("  범위    원본 %.4f  EMA %.4f" % (max(rw)-min(rw),max(ew)-min(ew)))

print("\n앙상블")
for name,tags in [
    ("translate 4 + 백본", TRANS+BB),
    ("11 + translate 4", PREV11+TRANS),
    ("translate + 백본 + EMA가중치", TRANS+BB+EMAW),
    ("translate + 백본 + EMA원본", TRANS+BB+EMARAW),
    ("translate + 백본 + EMA 둘다", TRANS+BB+EMAW+EMARAW),
    ("11 + translate + 백본 + EMA", PREV11+TRANS+BB+EMAW+EMARAW),
]:
    m,n=ens(tags)
    print("  %-30s n=%2d  macro-F1 %.4f  acc %.4f" % (name,n,m["macro_f1"],m["accuracy"]))

POOL=TRANS+BB+EMAW+EMARAW
scores={}
for k in (2,3):
    for c in itertools.combinations(POOL,k):
        scores[c]=float(evaluate(ty,np.mean([P[t] for t in c],0).argmax(1),9)["macro_f1"])
print("\nleave-one-out (풀 %d개)" % len(POOL))
rows=[(t,float(np.mean([v["delta"] for v in leave_one_out_delta(scores,t).values()])),single[t]) for t in POOL]
for t,loo,s in sorted(rows,key=lambda r:-r[1]):
    print("  %-24s 단독 %.4f  LOO %+.4f" % (t,s,loo))
