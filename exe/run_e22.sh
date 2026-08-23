#!/usr/bin/env bash
# E22 — 백본 계열 x translate 증강. 17차가 남긴 빈칸이다.
# 다양성을 만드는 축(백본, 오류 상관 0.6238)과 개별 품질을 가장 크게 올리는 것
# (translate, +0.038)을 같이 건 적이 한 번도 없었다.
# 반증 조건은 실행 전에 docs/experiments/candidate/backbone_diversity.md 에 박았다.
#
# seed 를 바깥 루프로 둔다 — 도중에 멈춰도 백본 3종이 고르게 남게 하려는 것이다.
# 사용: ./exe/run_e22.sh          (seed 0 1 2 전부)
#       ./exe/run_e22.sh 0 1      (지정한 seed 만)
source "$(dirname "$0")/_common.sh"
CACHE=data/wm811k/cache/wm811k_64pad.npz
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)

for seed in "${SEEDS[@]}"; do
  for bb in shufflenet_v2 mobilenet_v3 efficientnet_b0; do
    tag="e22_${bb}_tr_s${seed}"
    if [ -f "result/cls_baseline/${tag}_best_test.json" ]; then
      echo "이미 있음: $tag"; continue
    fi
    echo "=========== $tag  ($(date +%H:%M:%S)) ==========="
    $PY src/a3_train_wm811k_cls.py --config configs/cls_e17.yaml \
        --tag "$tag" --seed "$seed" --backbone "$bb" --extra-augment translate
    $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" \
        --backbone "$bb" --split test --cache "$CACHE"
  done
done
echo "=========== E22 DONE: ${SEEDS[*]}  ($(date +%H:%M:%S)) ==========="
