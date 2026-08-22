"""OOD one-class E0 — 자명한 스칼라 기준선. 메모리 가볍게, 청크 처리."""
import numpy as np

sp = np.load("data/wm811k/cache/splits_v1.npz")
d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
y = d["y"].astype(np.int64)
NAMES = ["none","Center","Donut","Edge-Loc","Edge-Ring","Loc","Random","Scratch","Near-full"]
tr, va, te = sp["train"], sp["val"], sp["test"]
print("one-class 학습 자료 (train none): %d" % int((y[tr]==0).sum()), flush=True)
print("test none %d / 결함 %d (결함비율 %.4f)" % (
    int((y[te]==0).sum()), int((y[te]!=0).sum()), float((y[te]!=0).mean())), flush=True)

X = d["X"]                      # 한 번만 읽는다
yte = y[te]
is_def = (yte != 0).astype(np.int64)

n_die = np.empty(len(te), np.float64)
n_fail = np.empty(len(te), np.float64)
disp = np.empty(len(te), np.float64)
elong = np.empty(len(te), np.float64)

H = W = X.shape[1]
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
CH = 4000
for a in range(0, len(te), CH):
    idx = te[a:a+CH]
    xb = X[idx]
    die = xb > 0
    fl = (xb == 2)
    n_die[a:a+len(idx)] = die.sum((1,2))
    n_fail[a:a+len(idx)] = fl.sum((1,2))
    f = fl.astype(np.float32)
    n = np.maximum(f.sum((1,2)), 1.0)
    sx = (f*xx).sum((1,2))/n; sy = (f*yy).sum((1,2))/n
    sxx = (f*xx*xx).sum((1,2))/n - sx*sx
    syy = (f*yy*yy).sum((1,2))/n - sy*sy
    sxy = (f*xx*yy).sum((1,2))/n - sx*sy
    t = sxx+syy; det = sxx*syy - sxy*sxy
    disc = np.sqrt(np.maximum(t*t/4.0 - det, 0.0))
    lmax = t/2.0 + disc; lmin = t/2.0 - disc
    disp[a:a+len(idx)] = np.sqrt(np.maximum(lmax,0.0))
    elong[a:a+len(idx)] = np.sqrt(np.maximum(lmax,1e-9)/np.maximum(lmin,1e-9))
    print("  %d/%d" % (a+len(idx), len(te)), flush=True)

def auroc(s, l):
    o = np.argsort(s, kind="mergesort"); s2, l2 = s[o], l[o]
    r = np.empty(len(s2))
    st = np.flatnonzero(np.concatenate([[True], s2[1:]!=s2[:-1], [True]]))
    for a_,b_ in zip(st[:-1], st[1:]): r[a_:b_] = (a_+b_-1)/2.0+1.0
    n1 = l2.sum(); n0 = len(l2)-n1
    return float((r[l2==1].sum() - n1*(n1+1)/2.0)/(n1*n0))
def aupr(s, l):
    o = np.argsort(-s, kind="mergesort"); l2 = l[o]
    tp = np.cumsum(l2); prec = tp/np.arange(1,len(l2)+1); rec = tp/l2.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0],rec]))*prec))
def fpr95(s, l):
    o = np.argsort(-s, kind="mergesort"); l2 = l[o]
    tpr = np.cumsum(l2)/l2.sum(); fpr = np.cumsum(1-l2)/(len(l2)-l2.sum())
    return float(fpr[min(int(np.searchsorted(tpr,0.95)), len(fpr)-1)])

scores = {
    "불량 다이 비율": n_fail/np.maximum(n_die,1),
    "불량 다이 개수": n_fail,
    "웨이퍼 크기(die수)": n_die,
    "불량 공간 퍼짐": disp,
    "불량 이심률": elong,
}
print("\n%-22s %8s %8s %10s" % ("자명한 스칼라","AUROC","AUPR","FPR@95TPR"), flush=True)
best=None
for k,s in scores.items():
    a,p,f = auroc(s,is_def), aupr(s,is_def), fpr95(s,is_def)
    print("%-22s %8.4f %8.4f %10.4f" % (k,a,p,f), flush=True)
    if best is None or a>best[1]: best=(k,a,s)
print("%-22s %8.4f %8.4f %10s" % ("무작위",0.5,float(is_def.mean()),"0.95"))
k,a,s = best
print("\n최고 = %s (AUROC %.4f)\n클래스별 (none 대비)" % (k,a))
ns = s[yte==0]; z = np.zeros(len(ns), np.int64)
for c in range(1,9):
    m = yte==c
    if m.sum()==0: continue
    print("  %-11s n=%5d AUROC %.4f" % (NAMES[c], int(m.sum()),
        auroc(np.concatenate([ns,s[m]]), np.concatenate([z,np.ones(int(m.sum()),np.int64)]))))
