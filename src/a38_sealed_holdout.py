"""봉인 홀드아웃 — 선택 편향을 잴 수 있게 자료를 **코드로** 잠근다.

## 왜 코드인가

이 워크스트림은 여덟 사이클 동안 공식 test 를 보고 다음 방향을 정했다.
9차 감사가 lot 반쪽으로 재서 순위가 유지되는 것을 확인했지만
**두 반쪽 다 이미 본 test 라 선택 편향은 안 없어진다.**
"앞으로 안 보겠다" 는 문서 규약은 지켜지지 않는다 — 그래서 **읽으면 실패하는 가드**를 둔다.

## 파티션

| 이름 | 무엇 | 봉인 |
|---|---|---|
| `test_dev` | 공식 test 중 봉인되지 않은 lot | 아니오. 개발은 여기서만 한다 |
| `test_sealed` | 공식 test lot 의 25% (lot 해시, 라벨 비의존) | **예** |
| `val_unseen` | 공식 val 전체 — OOD arm 이 한 번도 점수를 매긴 적 없다 | **예** |

크기와 선택 방식의 근거는 `docs/experiments/candidate/ood_sealed_holdout.md` §2 에
**실행 전에** 적었다. 결과를 보고 문턱을 바꾸면 봉인의 의미가 없으므로
`SALT` 와 `THRESHOLD` 는 테스트로 박혀 있다(`tests/test_a38_sealed_holdout.py`).

## 해제

`load_partition(name, unseal=True, reason="...")` 만이 봉인을 연다.
`unseal` 은 **`True` 그 자체**여야 한다 — `1` 이나 `"yes"` 는 거부된다.
`if unseal:` 로 쓰면 실수로 열리기 때문이다.
해제할 때마다 `result/ood/holdout/unseal_log.jsonl` 에 시각과 이유가 append 된다.
**몰래 열 수 없다.**
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

CACHE = Path("data/wm811k/cache")
HOLDOUT_DIR = Path("result/ood/holdout")
REGISTRY = "sealed_holdout_v1.npz"
UNSEAL_LOG = "unseal_log.jsonl"

SALT = "ood_sealed_v1"      # 이 문자열을 바꾸면 다른 봉인이 된다. 바꾸지 않는다.
THRESHOLD = 2500            # /10000. test lot 의 25% 를 봉인한다.

SEALED_PARTITIONS = ("test_sealed", "val_unseen")
OPEN_PARTITIONS = ("test_dev",)
PARTITIONS = SEALED_PARTITIONS + OPEN_PARTITIONS


class SealedHoldoutError(RuntimeError):
    """봉인된 자료를 명시적 해제 없이 만졌다."""


def lot_is_sealed(lot_name) -> bool:
    """lot 이름만 보고 정한다. **라벨도 점수도 웨이퍼 수도 보지 않는다.**"""
    h = hashlib.md5(f"{SALT}|{lot_name}".encode()).hexdigest()
    return int(h, 16) % 10000 < THRESHOLD


def registry_path() -> Path:
    return HOLDOUT_DIR / REGISTRY


def build_registry(cache: Path | None = None) -> dict:
    """레지스트리를 짓는다. 이미 있으면 **덮어쓰지 않고** 같은지 검증한다.

    덮어쓰기를 막는 이유: 봉인을 다시 지어 다른 lot 을 뽑을 수 있으면
    "결과가 나쁘면 다시 뽑는다" 가 가능해진다.
    """
    cache = Path(cache) if cache is not None else CACHE
    sp = np.load(cache / "splits_v1.npz")
    lots = np.load(cache / "wm811k_64pad.npz", allow_pickle=True)["lot_name"]
    te, va = sp["test"], sp["val"]

    uniq = np.unique(lots[te])
    sealed_lots = np.array([l for l in uniq if lot_is_sealed(l)], dtype=uniq.dtype)
    dev_lots = np.array([l for l in uniq if not lot_is_sealed(l)], dtype=uniq.dtype)
    m = np.isin(lots[te], sealed_lots)

    reg = {
        "test_sealed": te[m], "test_dev": te[~m], "val_unseen": va,
        "sealed_lots": sealed_lots, "dev_lots": dev_lots,
        "salt": np.array(SALT), "threshold": np.array(THRESHOLD),
        "built_at": np.array(time.strftime("%Y-%m-%dT%H:%M:%S%z")),
    }

    path = registry_path()
    if path.exists():
        old = _read_registry()
        for k in ("test_sealed", "test_dev", "val_unseen"):
            if not np.array_equal(old[k], reg[k]):
                raise SealedHoldoutError(
                    f"기존 레지스트리와 다른 분할이 나왔다({k}). 봉인은 다시 뽑지 않는다.")
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **reg)
    return reg


def _read_registry() -> dict:
    path = registry_path()
    if not path.exists():
        raise SealedHoldoutError(f"레지스트리가 없다: {path}. 먼저 build_registry() 를 부른다.")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _record_unseal(name: str, reason: str, n: int) -> None:
    log = HOLDOUT_DIR / UNSEAL_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "partition": name, "reason": reason, "n": int(n)}
    with log.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_partition(name: str, unseal=False, reason: str | None = None) -> np.ndarray:
    """파티션 인덱스를 돌려준다. 봉인된 것은 명시적 해제 없이는 실패한다."""
    if name not in PARTITIONS:
        raise ValueError(f"모르는 파티션: {name}. 가능한 것: {PARTITIONS}")
    reg = _read_registry()
    idx = reg[name]
    if name in OPEN_PARTITIONS:
        return idx
    if unseal is not True:
        raise SealedHoldoutError(
            f"'{name}' 은 봉인돼 있다. 읽으려면 unseal=True 와 reason 을 명시해야 한다. "
            "사전등록 없이 열지 마라 — 열면 그 순간 '본 적 없는 자료' 가 아니게 된다.")
    if not isinstance(reason, str) or not reason.strip():
        raise SealedHoldoutError(f"'{name}' 해제에는 이유가 필요하다. 감사 로그에 남는다.")
    _record_unseal(name, reason.strip(), len(idx))
    return idx


def sealed_index() -> np.ndarray:
    """봉인된 전체 인덱스. **가드 전용이다** — 평가에 쓰라고 있는 것이 아니다."""
    reg = _read_registry()
    return np.concatenate([reg[k] for k in SEALED_PARTITIONS])


def assert_not_sealed(idx, what: str = "인덱스") -> None:
    """임의의 인덱스 배열에 봉인 웨이퍼가 섞였는지 본다.

    파티션 API 를 안 거치고 `sp["test"]` 를 그대로 쓰는 스크립트를 잡기 위한 것이다.
    """
    bad = np.intersect1d(np.asarray(idx).ravel(), sealed_index())
    if len(bad):
        raise SealedHoldoutError(
            f"{what} 에 봉인된 웨이퍼가 {len(bad)}장 섞였다(예: {bad[:5].tolist()}). "
            "봉인 파티션은 사전등록된 일회 평가에서만 연다.")


if __name__ == "__main__":
    r = build_registry()
    y = np.load(CACHE / "wm811k_64pad.npz", allow_pickle=True)["y"]
    print("레지스트리 →", registry_path())
    print("%-12s %8s %8s %8s %9s" % ("파티션", "웨이퍼", "정상", "결함", "결함비율"))
    for k in ("test_dev", "test_sealed", "val_unseen"):
        yy = y[r[k]]
        n, nd = len(yy), int((yy != 0).sum())
        print("%-12s %8d %8d %8d %9.4f" % (k, n, n - nd, nd, nd / n))
    print("봉인 lot %d / dev lot %d (봉인 비율 %.4f)"
          % (len(r["sealed_lots"]), len(r["dev_lots"]),
             len(r["sealed_lots"]) / (len(r["sealed_lots"]) + len(r["dev_lots"]))))
