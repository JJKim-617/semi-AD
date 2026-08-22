"""E20 의 전제를 학습 없이 직접 검사한다 — CNN 은 이미 국소 밀도를 배웠는가.

## 왜 이걸 따로 재는가

E20 은 "국소적으로 몰려 있는가 를 conv 가 스스로 배워야 한다" 는 전제 위에 서 있다.
**그 전제가 틀릴 수 있다.** 이미 잘 배웠다면 채널로 다시 줘도 보탤 것이 없다.

가르는 방법이 있고 학습이 필요 없다. 지금 최고 앙상블의 `1 - p(none)` 과
**학습이 전혀 없는 국소 밀도 최대값**을 같은 이진 과제(none vs 결함)에서 재고,
**앙상블이 틀린 웨이퍼들 위에서** 밀도가 방향을 맞추는지 본다.

- 밀도가 앙상블 오류 위에서 무작위면 -> 이미 배운 것이고 E20 은 오르지 않아야 한다.
- 밀도가 앙상블 오류를 가르면 -> 안 배운 신호가 남아 있다.

**이건 E20 의 판정이 아니다.** 판정은 앙상블 기여이고 사전 등록돼 있다.
이건 결과가 어느 쪽으로 나오든 **왜 그런지**를 말할 수 있게 하는 사전 진단이다.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "src")
from a21_density import local_fail_density_map  # noqa: E402

POSTHOC = "result/posthoc"
KS = (3, 5, 7)


def dens_max(x, k):
    m = local_fail_density_map(x, k=k)
    die = x > 0
    return np.where(die, m, -np.inf).reshape(len(x), -1).max(1)


sp = np.load("data/wm811k/cache/splits_v1.npz")
d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
te = sp["test"]
X = np.ascontiguousarray(d["X"][te])
y = d["y"].astype(np.int64)[te]
is_def = (y != 0).astype(np.int64)
print("test %d장  결함 %d장 (%.1f%%)" % (len(y), is_def.sum(), 100 * is_def.mean()), flush=True)

scores = {"E0 전역 불량률": (X == 2).sum((1, 2)) / np.maximum((X > 0).sum((1, 2)), 1)}
for k in KS:
    scores["국소 밀도 max k=%d" % k] = dens_max(X, k)
    print("  밀도 k=%d 완료" % k, flush=True)

tta = sorted(glob.glob(f"{POSTHOC}/*_tta_logits.npz"))
if not tta:
    raise SystemExit("TTA 로짓이 없다")
probs, ty = [], None
for p in tta:
    z = np.load(p)
    ty = z["test_y"] if ty is None else ty
    a = z["test_logits"].astype(np.float64)
    ex = np.exp(a - a.max(1, keepdims=True))
    probs.append(ex / ex.sum(1, keepdims=True))
Pm = np.mean(probs, 0)
assert np.array_equal(ty, y)
print("앙상블: TTA 판 %d개" % len(tta))
scores["앙상블 1-p(none)"] = 1.0 - Pm[:, 0]

print("\n== 이진 과제 (none vs 결함) ==")
print("%-24s %8s %8s" % ("점수", "AUROC", "AUPR"))
res = {}
for n, s in scores.items():
    res[n] = (float(roc_auc_score(is_def, s)), float(average_precision_score(is_def, s)))
    print("%-24s %8.4f %8.4f" % (n, *res[n]))

# 순위로 바꿔 더한다 — 척도가 전혀 다른 두 점수를 합치는 가장 단순한 방법이고
# 튜닝 손잡이가 없다.
def rank(v):
    r = np.empty(len(v), np.float64)
    r[np.argsort(v, kind="stable")] = np.arange(len(v))
    return r / (len(v) - 1)


ens_s = scores["앙상블 1-p(none)"]
print("\n== 순위 합 (가중치 없음, 손잡이 없음) ==")
for k in KS:
    c = rank(ens_s) + rank(scores["국소 밀도 max k=%d" % k])
    print("  앙상블 + 밀도k%d      AUROC %.4f  AUPR %.4f" % (
        k, roc_auc_score(is_def, c), average_precision_score(is_def, c)))

print("\n== 앙상블이 틀린 웨이퍼 위에서 밀도가 방향을 맞추는가 ==")
pred = Pm.argmax(1)
fp = (y == 0) & (pred != 0)          # none 을 결함이라 부른 것
fn = (y != 0) & (pred == 0)          # 결함을 none 이라 부른 것
tn = (y == 0) & (pred == 0)
tp = (y != 0) & (pred != 0)
print("  FP %d  FN %d  TN %d  TP %d" % (fp.sum(), fn.sum(), tn.sum(), tp.sum()))
print("%-24s %10s %10s %10s" % ("점수", "FP 대 TN", "FN 대 TP", "FP 대 FN"))
for k in KS:
    s = scores["국소 밀도 max k=%d" % k]
    # FP 는 사실 none 이므로 밀도가 TN 보다 **낮아야** 잡을 수 있다(AUROC < 0.5 가 유용).
    a1 = roc_auc_score(np.r_[np.ones(fp.sum()), np.zeros(tn.sum())], np.r_[s[fp], s[tn]])
    a2 = roc_auc_score(np.r_[np.ones(fn.sum()), np.zeros(tp.sum())], np.r_[s[fn], s[tp]])
    a3 = roc_auc_score(np.r_[np.ones(fn.sum()), np.zeros(fp.sum())], np.r_[s[fn], s[fp]])
    print("%-24s %10.4f %10.4f %10.4f" % ("국소 밀도 max k=%d" % k, a1, a2, a3))
print("  읽는 법: FP 대 TN 이 0.5 에서 멀수록 밀도가 그 오류를 구분한다.")
print("           FP 대 FN 이 0.5 이면 두 오류 방향을 밀도로는 못 가른다.")

print("\n== 앙상블 점수와 밀도의 상관 (이미 배웠는가) ==")
from scipy.stats import spearmanr  # noqa: E402
for k in KS:
    r = spearmanr(ens_s, scores["국소 밀도 max k=%d" % k]).statistic
    rn = spearmanr(ens_s[y == 0], scores["국소 밀도 max k=%d" % k][y == 0]).statistic
    print("  k=%d  전체 rho %+.3f   none 안에서만 rho %+.3f" % (k, r, rn))

os.makedirs("docs/research/local_density/evidence", exist_ok=True)
json.dump({"binary": res, "n_tta": len(tta)},
          open("docs/research/local_density/evidence/complementarity.json", "w"),
          ensure_ascii=False, indent=2)
print("\n저장 -> docs/research/local_density/evidence/complementarity.json")
