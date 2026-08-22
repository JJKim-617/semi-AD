"""최종 집계. translate, 백본, TTA 를 어떻게 조합하는 것이 최선인가."""
from __future__ import annotations
import glob, itertools, json, os, sys
import numpy as np
sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate

NAMES=["none","Center","Donut","Edge-Loc","Edge-Ring","Loc","Random","Scratch","Near-full"]
POSTHOC="result/posthoc"
P,ty={},None
for p in glob.glob(f"{POSTHOC}/*_logits.npz"):
    t=os.path.basename(p).replace("_logits.npz","")
    d=np.load(p)
    if ty is None: ty=d["test_y"]
    z=d["test_logits"].astype(np.float64); ex=np.exp(z-z.max(1,keepdims=True))
    P[t]=ex/ex.sum(1,keepdims=True)
single={t: float(evaluate(ty,P[t].argmax(1),9)["macro_f1"]) for t in P}

def ens(tags):
    tags=[t for t in tags if t in P]
    return evaluate(ty,np.mean([P[t] for t in tags],0).argmax(1),9), len(tags)

TR=sorted(t for t in P if t.startswith("e17_translate_s") and not t.endswith("_tta"))
TRT=sorted(t for t in P if t.startswith("e17_translate_s") and t.endswith("_tta"))
BB=sorted(t for t in P if t.startswith("e18_"))
BBT=sorted(t for t in P if t.startswith("e18_") and t.endswith("_tta"))
BB=[t for t in BB if not t.endswith("_tta")]
OLD=["e8_pad","e10_pad_s1","e10_pad_s2","e7_ce_long_best","e6_focal_long_best",
     "e6_focal_long_s1_best","e7_ce_long80_best","e8_res96","e11_size","e11_shuffle"]
OLDT=[t+"_tta" for t in ["e8_pad","e10_pad_s1","e10_pad_s2"]]
OLDT+= [t for t in ["e7_ce_long_best_tta","e6_focal_long_best_tta",
                    "e6_focal_long_s1_best_tta","e7_ce_long80_best_tta","e8_res96_tta"] if t in P]

print("단독 최고 10개")
for t in sorted(single, key=lambda x:-single[x])[:10]:
    print("  %-28s %.4f" % (t, single[t]))

print("\n앙상블")
cands = [
    ("translate 4", TR),
    ("translate 4 (TTA)", TRT),
    ("translate 4 + 백본 6", TR+BB),
    ("translate 4 TTA + 백본 6", TRT+BB),
    ("translate 원본4+TTA4", TR+TRT),
    ("translate 8(원본+TTA) + 백본 6", TR+TRT+BB),
    ("옛10 + translate 4", OLD+TR),
    ("옛10 + translate 8 + 백본 6", OLD+TR+TRT+BB),
    ("옛10 + 옛TTA + translate 8 + 백본", OLD+OLDT+TR+TRT+BB),
]
best=None
for name,tags in cands:
    m,n=ens(tags)
    star=""
    if best is None or m["macro_f1"]>best[1]:
        best=(name,m["macro_f1"],tags); star=" <-"
    print("  %-34s n=%2d  macro-F1 %.4f  acc %.4f%s" % (name,n,m["macro_f1"],m["accuracy"],star))

name,f1,tags=best
m,_=ens(tags)
print("\n최선: %s  macro-F1 %.4f  accuracy %.4f" % (name,m["macro_f1"],m["accuracy"]))
print("  %-11s %9s %8s" % ("클래스","support","F1"))
for i,n in enumerate(NAMES):
    print("  %-11s %9d %8.3f" % (n,m["support"][i],m["per_class_f1"][i]))
pred=np.mean([P[t] for t in tags if t in P],0).argmax(1)
print("  none -> 결함 누출 %d / 110,701" % int(((ty==0)&(pred!=0)).sum()))
