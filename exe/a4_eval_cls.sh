#!/usr/bin/env bash
# 사용: ./exe/a4_eval_cls.sh result/cls_baseline/e1_scratch_best.pt [--split test]
source "$(dirname "$0")/_common.sh"
CKPT="$1"; shift || true
$PY src/a4_eval_wm811k_cls.py --ckpt "$CKPT" "$@"
