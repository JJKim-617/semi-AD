#!/usr/bin/env bash
# E7 이 끝난 뒤, 장기 학습 체크포인트 전부에 타겟 재가중 오프셋 보정을 적용한다.
# 사후 보정은 재학습이 없어 값싸므로 모든 후보에 일괄 적용해 비교한다.
source "$(dirname "$0")/_common.sh"
while pgrep -f "run_e7.sh" > /dev/null; do sleep 30; done
for tag in e6_focal_long e6_focal_long_s1 e7_ce_long e7_ce_long80; do
  ckpt="result/cls_baseline/${tag}_best.pt"
  [ -f "$ckpt" ] || { echo "skip $tag (없음)"; continue; }
  echo "=========== reweighted: $tag ==========="
  $PY src/a7_reweighted_offset.py --ckpt "$ckpt"
done
echo "=========== FINALIZE DONE ==========="
