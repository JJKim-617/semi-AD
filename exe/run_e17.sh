#!/usr/bin/env bash
# E17 — 추가 증강 선별. 변형 5종 x 2 seed 를 먼저 돌려 넓게 훑고,
# 유망한 것에만 seed 를 더한다. 반증 조건은
# docs/experiments/candidate/augmentation_expansion.md 에 미리 적었다.
source "$(dirname "$0")/_common.sh"
CACHE=data/wm811k/cache/wm811k_64pad.npz

run() {   # $1=tag suffix, 나머지=--extra-augment 인자
  local name="$1"; shift
  for seed in 0 1; do
    local tag="e17_${name}_s${seed}"
    [ -f "result/cls_baseline/${tag}_best_test.json" ] && { echo "skip $tag"; continue; }
    echo "=========== $tag  (extra: $*) ==========="
    $PY src/a3_train_wm811k_cls.py --config configs/cls_e17.yaml \
        --tag "$tag" --seed "$seed" --extra-augment "$@"
    $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" \
        --split test --cache "$CACHE"
  done
}

run scale     scale
run all       scale translate noise dropout
run noise     noise
run dropout   dropout
run translate translate
echo "=========== E17 DONE ==========="
