#!/usr/bin/env bash
# E8, E9 가 끝난 뒤 pad seed 를 늘리고 pad 앙상블까지 만든다.
source "$(dirname "$0")/_common.sh"
while pgrep -f "run_e8.sh|run_e9.sh" > /dev/null; do sleep 30; done
CACHE=data/wm811k/cache/wm811k_64pad.npz
for cfg in cls_e10_pad_s1 cls_e10_pad_s2; do
  tag=$($PY -c "import yaml;print(yaml.safe_load(open('configs/$cfg.yaml'))['tag'])")
  echo "=========== $cfg ($tag) ==========="
  $PY src/a3_train_wm811k_cls.py --config "configs/$cfg.yaml"
  $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" --split test --cache "$CACHE"
done
echo "=========== pad 앙상블 ==========="
$PY src/a8_ensemble.py --tags e8_pad e10_pad_s1 e10_pad_s2 --cache "$CACHE"
echo "=========== E10 DONE ==========="
