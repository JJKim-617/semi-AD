#!/usr/bin/env bash
# E6 가 끝난 뒤 실행된다.
source "$(dirname "$0")/_common.sh"
while pgrep -f "run_e6.sh" > /dev/null; do sleep 20; done
for cfg in cls_e7_ce_long cls_e7_ce_long80; do
  tag=$($PY -c "import yaml;print(yaml.safe_load(open('configs/$cfg.yaml'))['tag'])")
  echo "=========== $cfg ($tag) ==========="
  $PY src/a3_train_wm811k_cls.py --config "configs/$cfg.yaml"
  $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" --split test
done
echo "=========== ALL DONE ==========="
