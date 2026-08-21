#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
CKPT="$1"; shift || true
$PY src/a7_reweighted_offset.py --ckpt "$CKPT" "$@"
