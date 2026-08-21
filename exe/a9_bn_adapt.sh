#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
CKPT="$1"; shift || true
$PY src/a9_bn_adapt.py --ckpt "$CKPT" "$@"
