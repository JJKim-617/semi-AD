"""앙상블 구성원의 기여를 조합 전수로 잰다.

E14 에서 밝혀진 것: **단독 성능은 앙상블 기여를 예측하지 못한다.** resize96 은 단독
+0.005 로 잡음 안이라 "판정 보류" 였는데 상위 조합에 전부 들어간다. 반대로 단독 최고인
e8_pad(0.7485)를 혼자 쓰면 all-8 앙상블(0.7785)에 0.030 뒤진다.

따라서 새 구성원의 평가 기준은 단독 점수가 아니라 **leave-one-out** 이다.
그 구성원을 포함한 조합의 평균과, 같은 크기의 포함하지 않은 조합의 평균을 비교한다.
크기를 맞추지 않으면 앙상블 크기 효과가 기여로 오인된다.

구성원 목록은 docs/experiments/ensemble_members.json 에 둔다.
로짓이 없으면 만들고, 있으면 재사용한다.

사용:
    python src/bench_ensemble.py                      # 전체 집계
    python src/bench_ensemble.py --focus e15_polar_s0 # 특정 구성원의 기여
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np


def group_by_representation_count(combos, representation: dict) -> dict:
    """조합을 '구성원이 쓰는 서로 다른 표현의 가짓수' 로 묶는다.

    구성원 수가 아니라 표현 가짓수다. pad seed 3개는 1종이다.
    """
    out = defaultdict(list)
    for combo in combos:
        reps = {representation[m] for m in combo}   # 없는 구성원이면 KeyError
        out[len(reps)].append(tuple(combo))
    return dict(out)


def leave_one_out_delta(scores: dict, member: str) -> dict:
    """구성원 하나를 포함한 조합과 포함하지 않은 조합을 **같은 크기끼리** 비교한다.

    scores: {조합(튜플): macro-F1}. 조합 안의 순서는 무시한다.
    반환: {크기: {with_mean, without_mean, delta, n_with, n_without}}.
    한쪽이 비어 비교가 불가능한 크기는 아예 넣지 않는다 — 없는 숫자를 만들지 않는다.
    """
    by_size = defaultdict(lambda: ([], []))
    for combo, score in scores.items():
        with_, without = by_size[len(combo)]
        (with_ if member in set(combo) else without).append(float(score))

    out = {}
    for size, (with_, without) in by_size.items():
        if not with_ or not without:
            continue
        out[size] = dict(
            with_mean=float(np.mean(with_)), without_mean=float(np.mean(without)),
            delta=float(np.mean(with_) - np.mean(without)),
            n_with=len(with_), n_without=len(without),
        )
    return dict(sorted(out.items()))


def _load_members(manifest: str, out_dir: str, splits: str):
    """매니페스트를 읽고, 없는 로짓은 만들어서 확률로 돌려준다."""
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from a6_perclass_offset import cache_logits

    entries = json.loads(Path(manifest).read_text(encoding="utf-8"))
    probs, rep, y = {}, {}, None
    for e in entries:
        path = Path(out_dir) / f"{e['tag']}_logits.npz"
        if not path.exists():
            print(f"  [로짓 생성] {e['tag']}")
            # 백본과 밀도 채널을 넘기지 않으면 3채널 resnet18 로 만들어
            # e18/e20 체크포인트를 못 읽는다.
            cache_logits(e["ckpt"], e["cache"], splits, out_dir, e["tag"],
                         backbone=e.get("backbone", "resnet18"),
                         density_ks=tuple(e.get("density_ks", ())),
                         density_shuffle_seed=e.get("density_shuffle"))
        d = np.load(path)
        if y is None:
            y = d["test_y"]
        elif not np.array_equal(y, d["test_y"]):
            raise ValueError(f"{e['tag']} 의 test 정렬이 다르다")
        z = d["test_logits"].astype(np.float64)
        ex = np.exp(z - z.max(1, keepdims=True))
        probs[e["tag"]] = ex / ex.sum(1, keepdims=True)
        rep[e["tag"]] = e["representation"]
    return probs, rep, y


def main() -> None:
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from a4_eval_wm811k_cls import evaluate

    p = argparse.ArgumentParser(description="앙상블 구성원 기여 집계")
    p.add_argument("--manifest", default="docs/experiments/ensemble_members.json")
    p.add_argument("--out-dir", default="result/posthoc")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--max-k", type=int, default=None, help="이 크기까지만 전수 계산")
    p.add_argument("--focus", nargs="*", default=None, help="기여를 볼 구성원")
    a = p.parse_args()

    probs, rep, y = _load_members(a.manifest, a.out_dir, a.splits)
    tags = list(probs)
    nc = probs[tags[0]].shape[1]
    print(f"구성원 {len(tags)}개, 표현 {len(set(rep.values()))}종\n")

    print("단독 성능")
    singles = {}
    for t in tags:
        singles[t] = float(evaluate(y, probs[t].argmax(1), nc)["macro_f1"])
        print("  %-20s %-10s %.4f" % (t, rep[t], singles[t]))

    max_k = a.max_k or len(tags)
    scores = {}
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(tags, k):
            pred = np.mean([probs[t] for t in combo], 0).argmax(1)
            scores[combo] = float(evaluate(y, pred, nc)["macro_f1"])

    print("\n앙상블 크기 곡선 (모든 조합)")
    print("%3s %6s %10s %10s %10s" % ("k", "조합수", "평균", "최저", "최고"))
    for k in range(1, max_k + 1):
        v = [s for c, s in scores.items() if len(c) == k]
        print("%3d %6d %10.4f %10.4f %10.4f" % (k, len(v), np.mean(v), min(v), max(v)))

    print("\n표현 가짓수별 (크기 3 조합)")
    g = group_by_representation_count([c for c in scores if len(c) == 3], rep)
    for n_rep in sorted(g):
        v = [scores[c] for c in g[n_rep]]
        print("  표현 %d종  n=%3d  평균 %.4f  최고 %.4f" % (n_rep, len(v), np.mean(v), max(v)))

    focus = a.focus if a.focus is not None else tags
    print("\nleave-one-out 기여 (같은 크기끼리 비교)")
    for m in focus:
        d = leave_one_out_delta(scores, m)
        if not d:
            print("  %-20s 비교 가능한 크기 없음" % m)
            continue
        parts = ["k=%d %+.4f (%d대%d)" % (k, v["delta"], v["n_with"], v["n_without"])
                 for k, v in d.items() if 2 <= k <= min(5, max_k)]
        avg = np.mean([v["delta"] for k, v in d.items() if 2 <= k <= min(5, max_k)])
        print("  %-20s 평균 %+.4f   %s" % (m, avg, "  ".join(parts)))

    best = max(scores, key=scores.get)
    full = tuple(tags)
    print("\n최고 조합 %s" % (", ".join(best)))
    print("  macro-F1 %.4f" % scores[best])
    if full in scores:
        m = evaluate(y, np.mean([probs[t] for t in full], 0).argmax(1), nc)
        print("전체 %d개 앙상블  macro-F1 %.4f  accuracy %.4f" % (len(full), m["macro_f1"], m["accuracy"]))


if __name__ == "__main__":
    main()
