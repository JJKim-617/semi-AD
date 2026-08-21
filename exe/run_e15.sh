#!/usr/bin/env bash
# E15 — 극좌표 표현을 앙상블의 4번째 구성원으로 추가한다.
# 단독 성능은 기대하지 않는다. E14 에서 resize96 이 단독 +0.005(잡음 안)이면서
# 앙상블에는 기여한 것이 확인됐다. 여기서 묻는 것은 앙상블 기여다.
source "$(dirname "$0")/_common.sh"
while pgrep -f "run_e11.sh|mkpolar.sh" > /dev/null; do sleep 30; done
CACHE=data/wm811k/cache/wm811k_64polar.npz
[ -f "$CACHE" ] || { echo "극좌표 캐시 없음: $CACHE"; exit 1; }
for seed in 0 1; do
  tag="e15_polar_s${seed}"
  echo "=========== $tag ==========="
  $PY src/a3_train_wm811k_cls.py --config configs/cls_e15_polar.yaml --tag "$tag" --seed "$seed"
  $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" --split test --cache "$CACHE"
done
echo "=========== E15 DONE ==========="
