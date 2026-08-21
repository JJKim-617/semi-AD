#!/usr/bin/env bash
# 앞선 실험들이 끝난 뒤 크기 보조 입력과 셔플 대조군을 돌린다.
source "$(dirname "$0")/_common.sh"
while pgrep -f "run_e9.sh|run_e10.sh" > /dev/null; do sleep 30; done
echo "=========== e11_size (크기 정보 사용) ==========="
$PY src/a10_size_feature.py --tag e11_size
echo "=========== e11_shuffle (대조군) ==========="
$PY src/a10_size_feature.py --tag e11_shuffle --shuffle-size
echo "=========== E11 DONE ==========="
