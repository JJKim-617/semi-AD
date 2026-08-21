#!/usr/bin/env bash
# 사용: ./exe/a3_train_cls.sh configs/cls_e1_baseline.yaml
source "$(dirname "$0")/_common.sh"
CFG="${1:-configs/cls_e1_baseline.yaml}"; shift || true
$PY src/a3_train_wm811k_cls.py --config "$CFG" "$@"
