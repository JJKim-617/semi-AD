#!/usr/bin/env bash
# WM-811K 전처리. 기본 64x64, 인자로 설정 파일 교체 가능.
#   ./exe/a1_preprocess_wm811k.sh                      # configs/wm811k_64.yaml
#   ./exe/a1_preprocess_wm811k.sh configs/wm811k_32.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-configs/wm811k_64.yaml}"
python3 src/a1_preprocess_wm811k.py --config "$CFG"
