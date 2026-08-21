#!/usr/bin/env bash
# E3, E4 실험을 순차 실행하고 각각 공식 test 로 평가한다.
source "$(dirname "$0")/_common.sh"
for cfg in cls_e3_aug cls_e4_cw cls_e4_aug_cw; do
  tag=$($PY -c "import yaml;print(yaml.safe_load(open('configs/$cfg.yaml'))['tag'])")
  echo "=========== $cfg ($tag) ==========="
  $PY src/a3_train_wm811k_cls.py --config "configs/$cfg.yaml"
  $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" --split test
done
echo "=========== ALL DONE ==========="
