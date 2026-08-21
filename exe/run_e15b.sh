#!/usr/bin/env bash
# E15b — 극좌표를 올바른 증강(angular)으로 다시 돌리고, 이어서 E12 를 돌린다.
# 첫 E15 는 dihedral 을 써서 극좌표 구조를 파괴했다(test 0.5206). 그 결과는 폐기한다.
source "$(dirname "$0")/_common.sh"
CACHE=data/wm811k/cache/wm811k_64polar.npz

for seed in 0 1; do
  tag="e15b_polar_s${seed}"
  echo "=========== $tag (angular 증강) ==========="
  $PY src/a3_train_wm811k_cls.py --config configs/cls_e15_polar.yaml --tag "$tag" --seed "$seed"
  $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" --split test --cache "$CACHE"
done
echo "=========== E15B DONE ==========="

# --- E12 : 미라벨 638K 자기지도 ---
# 예산 적응: 사전학습 epoch 를 6 -> 3 으로 줄인다. 09:00 안에 두 조건 x 2 seed 를
# 끝내려면 필요하다. 대조군에도 똑같이 적용하므로 비교 자체는 공정하다.
ALL=data/wm811k/cache/wm811k_64pad_all.npz
PAD=data/wm811k/cache/wm811k_64pad.npz
if [ -f "$ALL" ]; then
  echo "=========== 사전학습: 전체 811K (3 epoch) ==========="
  $PY src/a11_ssl_pretrain.py --tag ssl_all --cache "$ALL" --epochs 3
  echo "=========== 사전학습: 라벨만 172K (대조군, 3 epoch) ==========="
  $PY src/a11_ssl_pretrain.py --tag ssl_lab --cache "$ALL" --labeled-only --epochs 3
  for pair in "e12_ssl:ssl_all:cls_e12_ssl" "e12_ssl_lab:ssl_lab:cls_e12_ssl_lab"; do
    base=${pair%%:*}; rest=${pair#*:}; enc=${rest%%:*}; cfgname=${rest##*:}
    for seed in 0 1; do
      tag="${base}_s${seed}"
      echo "=========== fine-tune: $tag (인코더 $enc, seed $seed) ==========="
      $PY src/a3_train_wm811k_cls.py --config "configs/$cfgname.yaml" \
          --tag "$tag" --seed "$seed" --init-encoder "result/ssl/${enc}_encoder.pt"
      $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" \
          --split test --cache "$PAD"
    done
  done
  echo "=========== E12 DONE ==========="
else
  echo "미라벨 캐시 없음: $ALL"
fi
