#!/usr/bin/env bash
# E18(백본 계열 교체) + E19(가중치 EMA). E17 이 끝난 뒤 이어서 돈다.
# 반증 조건은 docs/experiments/candidate/{backbone_diversity,ema}.md 에 미리 적었다.
source "$(dirname "$0")/_common.sh"
while pgrep -f "exe/run_e17" > /dev/null; do sleep 60; done
CACHE=data/wm811k/cache/wm811k_64pad.npz

# --- E18: 계열이 다른 백본 4종 x 2 seed ---
# resnet18 은 기존 3 seed(e8_pad, e10_pad_s1, e10_pad_s2)를 대조군으로 재사용한다.
for bb in shufflenet_v2 mobilenet_v3 efficientnet_b0 convnext_tiny; do
  for seed in 0 1; do
    tag="e18_${bb}_s${seed}"
    [ -f "result/cls_baseline/${tag}_best_test.json" ] && { echo "skip $tag"; continue; }
    echo "=========== $tag ==========="
    $PY src/a3_train_wm811k_cls.py --config configs/cls_e17.yaml \
        --tag "$tag" --seed "$seed" --backbone "$bb"
    $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" \
        --backbone "$bb" --split test --cache "$CACHE"
  done
done
echo "=========== E18 DONE ==========="

# --- E19: EMA. 원본과 EMA 를 같은 실행에서 짝지어 낸다 ---
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
