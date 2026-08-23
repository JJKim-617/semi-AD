#!/usr/bin/env bash
# E25 — 준지도(FixMatch 식 일관성). 미라벨 638,507장을 쓴다.
# 반증 조건은 실행 전에 docs/experiments/candidate/semi_supervised.md 에 박았다.
#
# **18차는 이것을 돌리지 않았다.** 사전 등록, 구현, 시험, 비용 실측, 1에폭 스모크까지만.
# 다음 세션이 그대로 이어받으면 된다.
#
# 비용 실측(18차): batch 128, mu=3 -> 에폭 156초, 40에폭 1.74시간, 3 seed 5.21시간.
#   **mu=7, batch 256 은 OOM 이다**(스텝당 4,096장). 하드웨어 상한이지 튜닝이 아니다.
#
# 사용: ./exe/run_e25.sh 0 1 2      (seed 지정)
#       ./exe/run_e25.sh --smoke    (1에폭만, 배선 확인)
source "$(dirname "$0")/_common.sh"
CACHE=data/wm811k/cache/wm811k_64pad.npz
ALL=data/wm811k/cache/wm811k_64pad_all.npz

if [ "${1:-}" = "--smoke" ]; then
  echo "=========== E25 스모크 (1 에폭) ==========="
  $PY src/a3_train_wm811k_cls.py --config configs/cls_e17.yaml \
      --tag smoke_e25 --seed 0 --epochs 1 \
      --batch-size 128 --extra-augment translate \
      --unlabeled-cache "$ALL" --mu 3 --tau 0.95
  rm -f result/cls_baseline/smoke_e25*
  exit 0
fi

SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
for seed in "${SEEDS[@]}"; do
  tag="e25_semi_s${seed}"
  if [ -f "result/cls_baseline/${tag}_best_test.json" ]; then
    echo "이미 있음: $tag"; continue
  fi
  echo "=========== $tag  ($(date +%H:%M:%S)) ==========="
  # batch 128 은 메모리 상한이라 고른 것이다. 대조군(e17_translate)은 256 이므로
  # **batch 가 다르다는 것을 판정 때 교란으로 반드시 밝혀야 한다.**
  $PY src/a3_train_wm811k_cls.py --config configs/cls_e17.yaml \
      --tag "$tag" --seed "$seed" --batch-size 128 \
      --extra-augment translate \
      --unlabeled-cache "$ALL" --mu 3 --tau 0.95
  $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" \
      --split test --cache "$CACHE"
done
echo "=========== E25 DONE: ${SEEDS[*]}  ($(date +%H:%M:%S)) ==========="
