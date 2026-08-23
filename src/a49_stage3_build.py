"""3단계 평가 자료를 짓는다 — 이미지 넷, 프롬프트 넷, 정답표.

규약은 `docs/experiments/candidate/ood_stage3_map_handoff.md` 에 **실행 전에** 박았다.
이 스크립트는 **MLLM 을 호출하지 않는다.** 다음 세션이 바로 돌릴 수 있게 자료만 만든다.

## 봉인 (문서 §6-1)

맵 npz 12,894장 중 **9,509장만 `test_dev`** 다. `cache_index` 를 파티션과 교집합해
거르고, `assert_not_sealed` 로 한 번 더 확인한다. **봉인 3,385장은 만지지 않는다.**

## 척도 (문서 §5-1)

요소 맵의 표시 범위는 **train-none 에서** 잡는다. test 통계를 쓰면 범위 A 가 깨진다.
그래서 train-none 29,149장의 맵을 실제로 계산해 1/99 백분위를 낸다.

## 위약 (문서 §4)

P1 은 **다른 웨이퍼의** 맵 셋과 스칼라를 쓴다. 이미지 수와 토큰 수가 E1 과 같으므로
E1 − P1 이 **맵 내용의 기여**를 준다. 이 대조가 없으면 판정이 무효다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "src")

from a22_ood_template import assert_one_class  # noqa: E402
from a23_ood_template_eval import NAMES  # noqa: E402
from a24_ood_residual import local_fail_count_map  # noqa: E402
from a27_line_filter import line_density_map  # noqa: E402
from a29_radial_calibration import band_index, calibrate_map, fit_band_reference  # noqa: E402
from a38_sealed_holdout import assert_not_sealed, load_partition  # noqa: E402
from a48_stage3_render import (  # noqa: E402
    ARM_IMAGE_ORDER, build_prompt, fit_component_scale, placebo_pairing,
    render_component, render_wafer,
)

MAPS = Path("result/ood/o1_pipeline_maps/pipeline_maps_pad.npz")
OUT = Path("result/ood/o3_stage3")
CHUNK, N_BANDS, L_LINE = 2000, 32, 11
PER_CLASS = 70                      # 문서 §6-3. 최소 클래스 Near-full 70 에 맞춘다.
SAMPLE_SEED, PLACEBO_SEED = 20260823, 7
T0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    X = d["X"]
    trn = sp["train"][y[sp["train"]] == 0]
    assert_one_class(y[trn])
    assert_not_sealed(trn, "train-none")
    pad_tr = np.ascontiguousarray(X[trn])
    del X, d
    log("train-none %d" % len(trn))

    # --- 척도를 train-none 에서 잡는다 (문서 §5-1) --------------------------------
    die_tr = (pad_tr > 0)
    v_tr = local_fail_count_map(pad_tr, k=5, chunk=CHUNK).astype(np.float32)
    ref = fit_band_reference(v_tr, pad_tr, y=y[trn], n_bands=N_BANDS, seed=0)
    a_tr = np.empty_like(v_tr)
    for i in range(0, len(pad_tr), CHUNK):
        xb = pad_tr[i:i + CHUNK]
        a_tr[i:i + len(xb)] = calibrate_map(
            local_fail_count_map(xb, k=5, chunk=CHUNK), band_index(xb, N_BANDS), ref)
    del v_tr
    b_tr = line_density_map(pad_tr, length=L_LINE, n_orient=8, chunk=CHUNK,
                            min_dies=L_LINE).astype(np.float32)
    c_tr = local_fail_count_map(pad_tr, k=3, chunk=CHUNK).astype(np.float32)
    scales = {"A": fit_component_scale(a_tr, die_tr),
              "B": fit_component_scale(b_tr, die_tr),
              "C": fit_component_scale(c_tr, die_tr)}
    del a_tr, b_tr, c_tr, die_tr, pad_tr
    log("척도(train-none 1/99 백분위): " + "  ".join(
        "%s [%.5f, %.5f]" % (k, *v) for k, v in scales.items()))

    # --- 평가 집합: test_dev 안에서 클래스당 70장 (문서 §6-3) ----------------------
    z = np.load(MAPS, allow_pickle=True)
    ci = z["cache_index"]
    ymap = z["y"].astype(np.int64)
    dev = set(load_partition("test_dev").tolist())
    in_dev = np.array([int(i) in dev for i in ci])
    rng = np.random.default_rng(SAMPLE_SEED)
    pick = []
    for c in range(1, 9):
        pool = np.flatnonzero((ymap == c) & in_dev)
        if len(pool) < PER_CLASS:
            raise ValueError("%s 가 %d 장뿐이다" % (NAMES[c], len(pool)))
        pick.append(rng.choice(pool, PER_CLASS, replace=False))
    pick = np.sort(np.concatenate(pick))
    assert_not_sealed(ci[pick], "3단계 평가 집합")
    log("평가 집합 %d 장 (클래스당 %d)" % (len(pick), PER_CLASS))

    wafer = z["wafer"][pick]
    maps = {"A": z["A_radialcal_k2k5"][pick].astype(np.float32),
            "B": z["B_line_L11_full"][pick].astype(np.float32),
            "C": z["C_k2_k3"][pick].astype(np.float32)}
    labels = ymap[pick]
    die = wafer > 0

    n_fail = (wafer == 2).reshape(len(pick), -1).sum(1)
    bmax = maps["B"].reshape(len(pick), -1).max(1)
    cmax = maps["C"].reshape(len(pick), -1).max(1)
    derived = np.log(np.maximum(bmax, 1e-6)) - np.log(np.maximum(cmax, 1e-6))
    scal = [{"n_fail": int(n_fail[i]), "log_b_minus_log_c": float(derived[i])}
            for i in range(len(pick))]

    # --- 렌더링 --------------------------------------------------------------------
    for i in range(len(pick)):
        Image.fromarray(render_wafer(wafer[i])).save(OUT / "images" / ("%04d_wafer.png" % i))
        for k in ("A", "B", "C"):
            Image.fromarray(render_component(maps[k][i], die[i], *scales[k])).save(
                OUT / "images" / ("%04d_%s.png" % (i, k)))
    log("이미지 %d 장 저장" % (len(pick) * 4))

    # --- arm 넷의 프롬프트 ----------------------------------------------------------
    pair = placebo_pairing(len(pick), seed=PLACEBO_SEED)
    for arm in ("C0", "E1", "P1", "S1"):
        rows = []
        for i in range(len(pick)):
            src = int(pair[i]) if arm == "P1" else i
            imgs = ["images/%04d_%s.png" % (i if k == "wafer" else src, k)
                    for k in ARM_IMAGE_ORDER[arm]]
            rows.append({"i": i, "arm": arm, "images": imgs,
                         "prompt": build_prompt(arm, scal[src]),
                         "truth": NAMES[int(labels[i])],
                         "scalar_source": src, "cache_index": int(ci[pick[i]])})
        (OUT / ("prompts_%s.jsonl" % arm)).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        log("%s 프롬프트 %d 개" % (arm, len(rows)))

    (OUT / "manifest.json").write_text(json.dumps({
        "per_class": PER_CLASS, "n": int(len(pick)),
        "sample_seed": SAMPLE_SEED, "placebo_seed": PLACEBO_SEED,
        "scales": {k: list(v) for k, v in scales.items()},
        "class_counts": {NAMES[c]: PER_CLASS for c in range(1, 9)},
        "note": "MLLM 은 호출하지 않았다. 자료만 만들었다.",
    }, ensure_ascii=False, indent=2))
    log("저장 완료 → %s" % OUT)
