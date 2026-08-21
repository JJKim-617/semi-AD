#!/usr/bin/env bash
source "$(dirname "$0")/_common.sh"
$PY src/a2_split_wm811k.py "$@"
