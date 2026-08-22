#!/usr/bin/env bash
# E19(EMA) 먼저, 그 다음 convnext 재시도.
#
# E18 의 convnext 가 batch 256 에서 CUDA OOM 으로 죽었다 — stem stride 를 1 로 낮춰
# 64x64 해상도가 유지되는데 27.8M 파라미터라 활성값이 크다. batch 64 로 낮춰 다시 돌린다.
# **다른 백본과 batch 가 달라지는 교란을 감수한다.** E18 은 이미 기각됐고 convnext 는
# 확인용이므로, 이 구성원의 쓸모는 앙상블 기여로만 본다.
source "$(dirname "$0")/_common.sh"
CACHE=data/wm811k/cache/wm811k_64pad.npz

for seed in 0 1 2; do
  tag="e19_ema_s${seed}"
  if [ ! -f "result/cls_baseline/${tag}_best_test.json" ]; then
    echo "=========== $tag ==========="
    $PY src/a3_train_wm811k_cls.py --config configs/cls_e17.yaml \
        --tag "$tag" --seed "$seed" --ema-decay 0.999
    echo "--- 원본 가중치 ---"
    $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" \
        --split test --cache "$CACHE"
  fi
  echo "--- EMA 가중치 ---"
  $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_ema.pt" \
      --split test --cache "$CACHE"
done
echo "=========== E19 DONE ==========="

for seed in 0 1; do
  tag="e18_convnext_tiny_s${seed}"
  [ -f "result/cls_baseline/${tag}_best_test.json" ] && continue
  echo "=========== $tag (batch 64) ==========="
  $PY src/a3_train_wm811k_cls.py --config configs/cls_e17.yaml \
      --tag "$tag" --seed "$seed" --backbone convnext_tiny --batch-size 64
  $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" \
      --backbone convnext_tiny --split test --cache "$CACHE"
done
echo "=========== CONVNEXT DONE ==========="
