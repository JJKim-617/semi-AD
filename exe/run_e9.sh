#!/usr/bin/env bash
# E8 이 끝난 뒤 실행된다. 학습 후 모든 스냅샷을 test 로 평가해 곡선을 만든다.
source "$(dirname "$0")/_common.sh"
while pgrep -f "run_e8.sh" > /dev/null; do sleep 30; done
echo "=========== cls_e9_curve ==========="
$PY src/a3_train_wm811k_cls.py --config configs/cls_e9_curve.yaml
echo "=========== 에폭별 test 평가 ==========="
for f in result/cls_baseline/snapshots/e9_curve/ep*.pt; do
  ep=$(basename "$f" .pt)
  printf "%s  " "$ep"
  $PY src/a4_eval_wm811k_cls.py --ckpt "$f" --split test --tag "e9_$ep" 2>/dev/null \
    | grep "macro-F1" | head -1
done
echo "=========== E9 DONE ==========="
