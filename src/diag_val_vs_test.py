"""해로운 구성원을 val 로 알아볼 수 있었는가."""
import json, os, sys
import numpy as np
sys.path.insert(0, "src")
from a4_eval_wm811k_cls import evaluate

entries = json.loads(open("docs/experiments/ensemble_members.json", encoding="utf-8").read())
rows = []
for e in entries:
    p = f"result/posthoc/{e['tag']}_logits.npz"
    if not os.path.exists(p):
        continue
    d = np.load(p)
    v = float(evaluate(d["val_y"], d["val_logits"].argmax(1), 9)["macro_f1"])
    t = float(evaluate(d["test_y"], d["test_logits"].argmax(1), 9)["macro_f1"])
    rows.append((e["tag"], v, t))

rows.sort(key=lambda r: -r[1])
print("%-22s %9s %9s %9s" % ("구성원", "val", "test", "격차"))
for tag, v, t in rows:
    print("%-22s %9.4f %9.4f %9.4f" % (tag, v, t, v - t))

v = [r[1] for r in rows]; t = [r[2] for r in rows]
def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))
print("\nval 대 test 상관 r = %+.3f (n=%d)" % (pearson(v, t), len(rows)))
print("val 최저 3개: %s" % ", ".join(r[0] for r in rows[-3:]))
print("test 최저 3개: %s" % ", ".join(r[0] for r in sorted(rows, key=lambda r: r[2])[:3]))

print("\n이 상관이 이상치 하나에 의존하는가")
def corr(sub):
    a = [r[1] for r in sub]; b = [r[2] for r in sub]
    return pearson(a, b), len(sub)
print("  전체            r = %+.3f (n=%d)" % corr(rows))
print("  극좌표 제외      r = %+.3f (n=%d)" % corr([r for r in rows if "polar" not in r[0]]))
print("  극좌표+SSL 제외  r = %+.3f (n=%d)" % corr(
    [r for r in rows if "polar" not in r[0] and "ssl" not in r[0]]))
