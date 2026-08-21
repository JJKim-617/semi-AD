#!/usr/bin/env bash
# 모든 exe 스크립트가 source 하는 공통 헤더.
# GPU 0 고정, 프로젝트 루트로 이동, 파이썬 인터프리터 고정.
#
# 경로는 이 파일 자신의 위치(BASH_SOURCE[0])로 정한다. 호출자 위치를 쓰면
# 프로젝트 밖에 둔 스크립트가 이 파일을 source 했을 때 엉뚱한 곳으로 cd 한다.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PY=/mnt/sdf/ryukimlee/miniconda3/envs/partfield/bin/python
