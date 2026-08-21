#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
for cfg in cls_e6_focal_long cls_e6_focal_long_s1; do
  tag=$($PY -c "import yaml;print(yaml.safe_load(open('configs/$cfg.yaml'))['tag'])")
  echo "=========== $cfg ($tag) ==========="
  $PY src/a3_train_wm811k_cls.py --config "configs/$cfg.yaml"
  $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" --split test
done
echo "=========== ALL DONE ==========="
