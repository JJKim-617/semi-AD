#!/usr/bin/env bash
# E7 과 캐시 생성이 모두 끝난 뒤 실행된다.
source "$(dirname "$0")/_common.sh"
while pgrep -f "run_e7.sh|mk_caches.sh" > /dev/null; do sleep 30; done
for cfg in cls_e8_pad cls_e8_res96; do
  tag=$($PY -c "import yaml;print(yaml.safe_load(open('configs/$cfg.yaml'))['tag'])")
  cache=$($PY -c "import yaml;print(yaml.safe_load(open('configs/$cfg.yaml'))['cache'])")
  [ -f "$cache" ] || { echo "skip $tag (캐시 $cache 없음)"; continue; }
  echo "=========== $cfg ($tag) ==========="
  $PY src/a3_train_wm811k_cls.py --config "configs/$cfg.yaml"
  $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" --split test --cache "$cache"
done
echo "=========== E8 DONE ==========="
