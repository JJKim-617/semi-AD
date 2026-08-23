"""자유 각도 회전이 **명목형 격자를 망가뜨리지 않는가** (18차).

이 프로젝트의 첫 원칙은 "재표본은 최근접 이웃만, 보간 금지" 다.
셀 값 {0=다이 없음, 1=정상, 2=불량}은 명목형이라 1과 2 사이에 중간값이 없다.
그리고 이 레포에는 **도구 기본값이 조용히 값을 만들어낸 사고**가 있다
(`uniform_filter` 의 반올림 잡음이 신호로 둔갑해 AUPR 을 0.3844 -> 0.4986 으로 부풀렸다).

`random_rotate` 는 scipy 를 쓰지 않고 정수 색인 gather 로만 구현했으므로 **보간이
구조적으로 불가능**하다. 아래는 그것을 시험으로 박고, 더 위험한 것 — **격자 재표본이
다이를 만들거나 지우는가** — 에 선을 긋는다. 사각 격자를 임의 각도로 돌리면
어떤 원본 칸은 두 번 읽히고 어떤 칸은 한 번도 안 읽힌다. 그것이 `die_dropout` 과
같아지면 이 증강은 쓸 수 없다(dropout 은 이 프로젝트에서 누출을 2배로 늘렸다).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a18_augment import random_rotate  # noqa: E402


def disc(n=64, size=64, seed=0):
    """가운데 원판 위에 불량이 섞인 합성 웨이퍼 n장."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.hypot(yy - (size - 1) / 2, xx - (size - 1) / 2)
    base = np.where(d <= size // 2 - 2, 1, 0).astype(np.uint8)
    out = np.repeat(base[None], n, 0)
    fail = (rng.random(out.shape) < 0.1) & (out > 0)
    out[fail] = 2
    return out


def test_value_set_is_exactly_the_three_nominal_categories():
    """세 값이 전부 있는 입력을 돌리면 나오는 값의 집합도 정확히 {0,1,2} 다.

    부분집합이 아니라 **정확히 같은 집합**을 요구한다. 보간이 끼면 여기서 깨진다.
    """
    x = disc(32)
    assert set(np.unique(x)) == {0, 1, 2}
    out = random_rotate(x, rng=np.random.default_rng(0), max_deg=180)
    assert set(np.unique(out).tolist()) == {0, 1, 2}


def test_no_intermediate_category_appears_at_many_angles():
    """각도를 훑어도 새 범주가 안 생긴다. 한 각도만 보면 운으로 통과할 수 있다."""
    x = disc(1, size=65)
    for deg in np.arange(0, 360, 7.0):
        out = random_rotate(x, rng=None, angles=np.array([float(deg)]))
        assert set(np.unique(out).tolist()) <= {0, 1, 2}, f"{deg}도에서 새 값"


def test_output_dtype_is_still_the_nominal_uint8():
    out = random_rotate(disc(4), rng=np.random.default_rng(0), max_deg=180)
    assert out.dtype == np.uint8


def test_per_wafer_die_count_barely_moves():
    """**장마다** 다이 수가 거의 안 변해야 한다.

    합계만 보면 어떤 장이 늘고 어떤 장이 줄어 상쇄된다 — 그래서 장별로 본다.
    `die_dropout` 은 다이의 2.0% 를 지운다. 회전이 그 수준이면 같은 것이 된다.
    """
    x = disc(64)
    out = random_rotate(x, rng=np.random.default_rng(0), max_deg=180)
    before = (x > 0).sum(axis=(1, 2))
    after = (out > 0).sum(axis=(1, 2))
    rel = np.abs(after - before) / before
    assert rel.max() < 0.02, f"장별 다이 수가 최대 {rel.max():.2%} 움직였다"


def test_ninety_degree_rotation_preserves_every_fail_die_exactly():
    """90도의 배수는 격자를 자기 자신으로 보내므로 **불량 다이가 한 칸도 안 없어진다.**

    이것이 dihedral 이 이 프로젝트 최대 이득(+0.101)인 이유이고,
    아래 자유 각도 시험과의 대비가 E23 기각의 근거다.
    """
    x = disc(64)
    for deg in (90.0, 180.0, 270.0):
        out = random_rotate(x, rng=None, angles=np.full(len(x), deg))
        assert np.array_equal((out == 2).sum(axis=(1, 2)),
                              (x == 2).sum(axis=(1, 2))), f"{deg}도에서 불량이 변했다"


def test_free_angle_rotation_destroys_fail_dies_like_dropout():
    """**자유 각도 회전은 성긴 불량 다이를 재표본에서 잃는다. 기각의 근거다.**

    보간 때문이 아니다(보간은 안 쓴다). 최근접 역사상에서 **어떤 원본 칸도 가리키지
    않는 출력 칸**이 생기고, 불량 다이는 낱개로 흩어져 있어 그 손실을 그대로 맞는다.
    다이 영역 자체는 멀쩡한데(위 시험) 그 위의 표식만 사라진다.

    실측(실제 WM-811K, 장별): 자유 회전이 불량 다이의 2.3~3.3% 를 잃고
    `die_dropout`(rate=0.02)이 2.0% 를 잃는다. **같은 등급이다.**
    dropout 은 이 프로젝트에서 누출을 1,925 -> 4,014 로 늘린 증강이다.

    이 시험은 "고쳐야 할 버그" 가 아니라 **이산 명목 격자의 성질**을 박아 둔 것이다.
    구현이 바뀌어 이 손실이 사라지면 그때 E23 을 다시 열어야 한다.
    """
    x = disc(256)
    out = random_rotate(x, rng=np.random.default_rng(0), max_deg=180)
    before = (x == 2).sum(axis=(1, 2)).astype(float)
    after = (out == 2).sum(axis=(1, 2)).astype(float)
    rel = np.abs(after - before) / before
    assert rel.mean() > 0.01, (
        "자유 회전이 더 이상 불량 다이를 잃지 않는다 — E23 기각 근거가 사라졌으니 "
        "candidate 문서를 다시 열어라")
    # 다이 영역은 멀쩡하다는 것을 같은 시험 안에서 대비시킨다.
    dies = np.abs((out > 0).sum(axis=(1, 2)) - (x > 0).sum(axis=(1, 2))) / (x > 0).sum(axis=(1, 2))
    assert dies.mean() < 0.01, "다이 영역까지 무너지면 다른 이야기가 된다"


def test_the_die_region_stays_connected_not_speckled():
    """회전이 다이 영역 안에 구멍을 뚫으면 안 된다.

    재표본이 원본 칸을 건너뛰면 **다이 영역 안에** 0 이 생긴다. 그것이
    `die_dropout` 이 하는 일이고, 웨이퍼 모양 자체가 바뀐다.
    원판의 안쪽(반지름 80% 이내)에는 다이 없음 칸이 없어야 한다.
    """
    x = disc(1, size=65)
    out = random_rotate(x, rng=None, angles=np.array([37.0]))
    size = 65
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(yy - 32, xx - 32)
    inner = r <= (size // 2 - 2) * 0.8
    holes = int(((out[0] == 0) & inner).sum())
    assert holes == 0, f"다이 영역 안쪽에 구멍이 {holes}칸 생겼다"
