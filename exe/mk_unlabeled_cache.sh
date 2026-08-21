#!/usr/bin/env bash
# 자기지도 사전학습용 미라벨 캐시. 라벨 데이터와 같은 pad 64 표현을 쓴다.
source "$(dirname "$0")/_common.sh"
$PY src/a1_preprocess_wm811k.py --size 64 --mode pad --include-unlabeled --tag wm811k_64pad_all
