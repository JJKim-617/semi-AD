#!/usr/bin/env bash
# E27 — batch 128 대조군. E25 설계를 완성하는 것이지 새 축이 아니다.
# e17_translate 와 **batch 하나만** 다르게 둔다. 그것이 대조군의 뜻이다.
# 사전 등록은 실행 전에 docs/experiments/candidate/batch_control.md 에 박았다.
source "$(dirname "$0")/_common.sh"
CACHE=data/wm811k/cache/wm811k_64pad.npz

for seed in "$@"; do
  tag="e27_b128_s${seed}"
  if [ -f "result/cls_baseline/${tag}_best_test.json" ]; then
    echo "이미 있음: $tag"; continue
  fi
  echo "=========== $tag  ($(date +%H:%M:%S)) ==========="
  $PY src/a3_train_wm811k_cls.py --config configs/cls_e17.yaml \
      --tag "$tag" --seed "$seed" --batch-size 128 --extra-augment translate
  $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" \
      --split test --cache "$CACHE"
done
echo "=========== E27 DONE: $*  ($(date +%H:%M:%S)) ==========="
