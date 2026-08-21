"""실행 환경 공통 설정. 모든 학습, 평가 스크립트가 import 첫 줄에서 호출한다.

두 가지를 강제한다.
1. GPU 0 만 사용. 다른 GPU 는 공용이므로 절대 점유하지 않는다.
2. 프로세스 표시 이름을 중립적인 이름으로 바꾼다. nvitop, ps 의 COMMAND 에
   프로젝트 경로나 이름이 그대로 노출되지 않게 한다.

사용:
    from _runtime import setup
    setup("train")          # -> nvitop COMMAND 에 "wafer-train" 으로 표시
"""

from __future__ import annotations

import os
import sys

ALLOWED_GPU = "0"
TITLE_PREFIX = "wafer"


def pin_gpu(device: str = ALLOWED_GPU) -> None:
    """CUDA_VISIBLE_DEVICES 를 고정한다. torch import 전에 호출해야 한다."""
    cur = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cur not in (None, "", device):
        raise RuntimeError(
            f"이 서버에서는 GPU {device} 만 사용한다. "
            f"CUDA_VISIBLE_DEVICES={cur} 로 설정돼 있다."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    if "torch" in sys.modules:
        print("[warn] torch 가 이미 import 됐다. GPU 고정이 적용되지 않을 수 있다.",
              file=sys.stderr)


def set_title(name: str) -> None:
    """프로세스 표시 이름 변경. setproctitle 이 없으면 조용히 넘어간다."""
    try:
        from setproctitle import setproctitle
    except ImportError:
        return
    setproctitle(f"{TITLE_PREFIX}-{name}")


def setup(name: str, device: str = ALLOWED_GPU) -> None:
    pin_gpu(device)
    set_title(name)
