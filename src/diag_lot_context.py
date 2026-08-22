"""웨이퍼맵 **밖**의 정보로 경계 오류를 고칠 수 있는가 — lot 문맥 (학습 없음).

## 왜 이걸 보나

같은 사이클의 상한 재계산이 이렇게 말한다. **none/결함 이진을 완벽하게 풀어야
0.8746 이고 그래야 목표 0.87 을 겨우 넘는다.** 그런데 손으로 만든 신호
(국소 밀도, 선 필터)는 전부 앙상블의 경계 오류 위에서 **같은 편**이었다 —
오검출은 그 신호로도 결함처럼 보이고 놓친 결함은 정상처럼 보인다.

그러면 **웨이퍼맵 안에 없는 정보**를 봐야 한다. 캐시에 `lot_name` 이 있다.
같은 lot 은 공정 조건을 공유하므로 라벨이 뭉칠 수 있다.

`splits.md` 대로 **train/val 과 test 는 lot 을 하나도 공유하지 않는다.**
그러니 train 라벨을 끌어올 수는 없다. 대신 **test lot 안에서**
형제 웨이퍼들에 대한 **모델 자신의 예측**을 쓴다 — 라벨을 안 보므로 누수가 아니고,
팹에서 웨이퍼가 lot 단위로 들어온다는 점에서 실제로 쓸 수 있는 정보다.

두 가지를 잰다.
- `lot 참 결함률` — 자기 자신을 뺀 lot 의 실제 결함 비율. **상한**이다(라벨을 본다).
- `lot 예측 점수 평균` — 자기 자신을 뺀 lot 의 모델 결함점수 평균. **실제로 쓸 수 있다.**

## 반증 가능한 판정

경계 오류를 고치려면 lot 문맥이 모델과 **반대 편**이어야 한다 —
`FP 대 TN < 0.5` 이고 `FN 대 TP > 0.5`. 같은 편이면 고칠 수 없다.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")

OUT = "docs/research/lot_context/evidence"


def main() -> None:
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    te = sp["test"]
    lot, y = d["lot_name"][te], d["y"].astype(int)[te]

    probs, ty = [], None
    for p in sorted(glob.glob("result/posthoc/*_tta_logits.npz")):
        z = np.load(p)
        ty = z["test_y"] if ty is None else ty
        a = z["test_logits"].astype(np.float64)
        ex = np.exp(a - a.max(1, keepdims=True))
        probs.append(ex / ex.sum(1, keepdims=True))
    P = np.mean(probs, 0)
    pred = P.argmax(1)
    assert np.array_equal(ty, y)

    uniq, inv = np.unique(lot, return_inverse=True)
    n_lot = len(uniq)
    cnt = np.bincount(inv, minlength=n_lot).astype(float)
    s = 1.0 - P[:, 0]
    isd = (y != 0).astype(float)
    # 자기 자신을 뺀 lot 평균 (leave-one-out). 자기 정보가 새면 아무 의미가 없다.
    loo_s = (np.bincount(inv, weights=s, minlength=n_lot)[inv] - s) / np.maximum(cnt[inv] - 1, 1)
    loo_y = (np.bincount(inv, weights=isd, minlength=n_lot)[inv] - isd) / np.maximum(cnt[inv] - 1, 1)
    ok = cnt[inv] >= 3

    res = {"n_lot": int(n_lot), "n_usable": int(ok.sum()),
           "median_lot_size": float(np.median(cnt))}
    print("lot %d개, test lot 당 웨이퍼 중앙값 %.0f, 3장 이상인 lot 의 웨이퍼 %d/%d"
          % (n_lot, np.median(cnt), ok.sum(), len(y)))

    print("\n== lot 안에서 라벨이 뭉치는가 (이진 none vs 결함) ==")
    for name, v in [("lot 참 결함률 (상한, 라벨을 봄)", loo_y),
                    ("lot 예측 점수 평균 (실사용 가능)", loo_s),
                    ("(참고) 자기 웨이퍼의 모델 점수", s)]:
        a = float(roc_auc_score(isd[ok].astype(int), v[ok]))
        res[name] = a
        print("  %-34s AUROC %.4f" % (name, a))

    FP = (y == 0) & (pred != 0) & ok
    TN = (y == 0) & (pred == 0) & ok
    FN = (y != 0) & (pred == 0) & ok
    TP = (y != 0) & (pred != 0) & ok
    print("\n== 앙상블 오류 위에서 (FP %d TN %d FN %d TP %d) =="
          % (FP.sum(), TN.sum(), FN.sum(), TP.sum()))
    for name, v in [("lot 참 결함률 (상한)", loo_y), ("lot 예측 점수 평균", loo_s)]:
        a1 = float(roc_auc_score(np.r_[np.ones(FP.sum()), np.zeros(TN.sum())],
                                 np.r_[v[FP], v[TN]]))
        a2 = float(roc_auc_score(np.r_[np.ones(FN.sum()), np.zeros(TP.sum())],
                                 np.r_[v[FN], v[TP]]))
        res[name + " | FP vs TN"], res[name + " | FN vs TP"] = a1, a2
        verdict = "반대 편 (고칠 수 있다)" if (a1 < 0.5 < a2) else "**같은 편 (못 고친다)**"
        print("  %-22s FP 대 TN %.4f   FN 대 TP %.4f   -> %s" % (name, a1, a2, verdict))

    print("\n  비교: 국소 밀도 k=5 는 FP 대 TN 0.8806 / FN 대 TP 0.2126 (같은 편)")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "lot_context.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\n저장 -> %s/lot_context.json" % OUT)


if __name__ == "__main__":
    main()
