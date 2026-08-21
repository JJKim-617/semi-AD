#!/usr/bin/env bash
# 대안 입력 표현 캐시를 만든다. 진단 결과 리사이즈가 웨이퍼의 물리적 크기 정보를
# 파괴하는 것이 확인됐다(Scratch 는 train median 52x52, test 31x31).
source "$(dirname "$0")/_common.sh"
$PY src/a1_preprocess_wm811k.py --size 64 --mode pad --tag wm811k_64pad
$PY src/a1_preprocess_wm811k.py --size 96 --mode resize --tag wm811k_96
