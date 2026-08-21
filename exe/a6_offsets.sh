#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
CKPT="$1"; shift || true
$PY src/a6_perclass_offset.py --ckpt "$CKPT" "$@"
