#!/usr/bin/env bash
# 모든 exe 스크립트가 source 하는 공통 헤더.
# GPU 0 고정, 프로젝트 루트로 이동, 파이썬 인터프리터 고정.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[1]}")/.."
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="src:${PYTHONPATH:-}"
PY=/mnt/sdf/ryukimlee/miniconda3/envs/partfield/bin/python
export PYTHONUNBUFFERED=1   # 백그라운드 실행 시 로그가 버퍼에 갇히지 않게 한다
