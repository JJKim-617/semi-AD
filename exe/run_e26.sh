#!/usr/bin/env bash
# E26 — 무증강 대조군을 seed 3,4,5 로 넓힌다. 붕괴가 얼마나 흔한지 재는 것이다.
# 새 축이 아니라 대조군의 분포 측정이다. 사전 등록은
# docs/experiments/candidate/control_variance.md 에 실행 전에 박았다.
#
# 기존 대조군(e8_pad s0, e10_pad_s1 s1, e10_pad_s2 s2)과 완전히 같은 설정을 쓴다.
source "$(dirname "$0")/_common.sh"
CACHE=data/wm811k/cache/wm811k_64pad.npz

for seed in "$@"; do
  tag="e26_pad_s${seed}"
  if [ -f "result/cls_baseline/${tag}_best_test.json" ]; then
    echo "이미 있음: $tag"; continue
  fi
  echo "=========== $tag  ($(date +%H:%M:%S)) ==========="
  $PY src/a3_train_wm811k_cls.py --config configs/cls_e8_pad.yaml \
      --tag "$tag" --seed "$seed"
  $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" \
      --split test --cache "$CACHE"
done
echo "=========== E26 DONE: $*  ($(date +%H:%M:%S)) ==========="
