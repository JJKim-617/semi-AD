#!/usr/bin/env bash
# E21 — 방향성 선 필터 입력 채널. 반증 조건은 실행 전에
# docs/experiments/candidate/line_filter_channel.md 에 박았다.
# 사용: ./exe/run_e21.sh 0 [1 2]
source "$(dirname "$0")/_common.sh"
CACHE=data/wm811k/cache/wm811k_64pad.npz

for seed in "$@"; do
  tag="e21_line_s${seed}"
  if [ -f "result/cls_baseline/${tag}_best_test.json" ]; then
    echo "이미 있음: $tag"; continue
  fi
  echo "=========== $tag ==========="
  $PY src/a3_train_wm811k_cls.py --config configs/cls_e17.yaml \
      --tag "$tag" --seed "$seed" --extra-augment translate --line-ls 11
  $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" \
      --split test --cache "$CACHE" --line-ls 11
done
echo "=========== E21 DONE: $* ==========="
