"""파이프라인 2~3단계로 넘길 다이 단위 맵을 내보내고 대표 사례를 그린다.

`a26_ood_residual_export.py` 를 대체한다(그쪽은 O1 template 시절 산출물이라 남겨만 둔다).

## 무엇을 내보내는가 — 현재 최선의 **세 요소 맵**

탐지 점수는 세 신호의 융합이고(`docs/reports/OOD.md`), 각 신호는 **서로 다른 결함을 본다.**
그러니 하류에 넘길 것도 하나가 아니라 셋이다.

| 맵 | 무엇을 잘 보는가 |
|---|---|
| `A_radialcal_k2k5` | 전반 (Loc, Edge). 창 5x5 k² 밀도를 반경 대역별 정상 백분위로 |
| `B_line_L11_full` | **Scratch.** 온전창 방향성 선 필터 (길이 11, 8방향, 다이 11개 창만) |
| `C_k2_k3` | **Center.** 창 3x3 k² 밀도 |

**셋 다 넘기는 이유**: 어느 것이 하류 8종 분류를 돕는지는 **아직 모른다**(기획서 §9.4).
그리고 탐지에서 셋이 상보적이라는 것은 이미 보였으니, 분류에서도 그럴 가능성이 있다.

## 좌표계

**pad(원 좌표)** 를 쓴다. 리사이즈는 작은 웨이퍼를 확대해 창 하나가 덮는 물리적 크기를
웨이퍼마다 다르게 만든다 — 국소 창을 쓰는 순간 그건 교란이다.

## 픽셀 단위 정답은 만들지 않는다

기획서 §9.3. 그림은 **정성 평가**용이다. 합성 마스크로 P-AUROC 를 내지 않는다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from a21_ood_metrics import fail_ratio  # noqa: E402
from a22_ood_template import assert_one_class  # noqa: E402
from a23_ood_template_eval import ecdf_percentile, fisher  # noqa: E402
from a24_ood_residual import local_fail_count_map, local_fail_count_max  # noqa: E402
from a26_ood_residual_export import select_cases, subset_for_export  # noqa: E402
from a27_line_filter import line_density_map, line_density_max  # noqa: E402
from a29_radial_calibration import band_index, calibrate_map, fit_band_reference  # noqa: E402

OUT = Path("result/ood/o1_pipeline_maps")
FIG = Path("docs/research/ood_pipeline_maps/evidence")
NAMES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch",
         "Near-full"]
WAFER_CMAP = ListedColormap(["#f2f2f2", "#3b6ea5", "#d1495b"])
L_LINE, N_BANDS, CHUNK = 11, 32, 4000
T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def _panel(ax, img, title, kind, die):
    if kind == "wafer":
        ax.imshow(img, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
    else:
        v = float(np.max(img[die])) if die.any() else 1.0
        ax.imshow(np.where(die, img, np.nan), cmap="inferno", vmin=0,
                  vmax=max(v, 1e-6), interpolation="nearest")
        ax.set_facecolor("#e8e8e8")
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def figure_for_class(x, maps, score, y, cls, path, n=3):
    cases = select_cases(score, y, cls, n)
    groups = [("caught (top)", cases["caught"]), ("MISSED (bottom)", cases["missed"]),
              ("normal, false alarm", cases["false_alarm"]),
              ("normal, typical", cases["typical_normal"])]
    rows = sum(len(g[1]) for g in groups)
    fig, axes = plt.subplots(rows, 4, figsize=(8.0, 2.05 * rows))
    axes = np.atleast_2d(axes)
    r = 0
    for gname, idxs in groups:
        for i in idxs:
            die = x[i] > 0
            _panel(axes[r, 0], x[i], "%s\n%s  fail=%.3f"
                   % (gname, NAMES[y[i]], fail_ratio(x[i][None])[0]), "wafer", die)
            _panel(axes[r, 1], maps["A"][i], "A radius-calibrated k2 5x5", "m", die)
            _panel(axes[r, 2], maps["B"][i], "B full-window line L=11", "m", die)
            _panel(axes[r, 3], maps["C"][i], "C k2 3x3", "m", die)
            r += 1
    fig.suptitle("%s — original (pad) coords, score = 3-signal fusion" % NAMES[cls],
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    tr, te = sp["train"], sp["test"]
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    trn = tr[y[tr] == 0]
    assert_one_class(y[trn])
    X = d["X"]
    pad_tr = np.ascontiguousarray(X[trn])
    yte = y[te]
    keep_local = subset_for_export(yte, n_normal=5000, seed=0)
    keep_global = te[keep_local]
    x = np.ascontiguousarray(X[keep_global])
    del X
    log("적재 완료. 내보낼 %d장 (결함 %d / 정상 %d)"
        % (len(x), int((yte[keep_local] != 0).sum()), int((yte[keep_local] == 0).sum())))

    v_tr = local_fail_count_map(pad_tr, k=5, chunk=CHUNK).astype(np.float32)
    ref = fit_band_reference(v_tr, pad_tr, y=y[trn], n_bands=N_BANDS, seed=0)
    del v_tr

    def a_map(xb):
        return calibrate_map(local_fail_count_map(xb, k=5, chunk=CHUNK),
                             band_index(xb, N_BANDS), ref)

    maps = {
        "A": a_map(x).astype(np.float32),
        "B": line_density_map(x, length=L_LINE, n_orient=8, chunk=CHUNK,
                              min_dies=L_LINE).astype(np.float32),
        "C": local_fail_count_map(x, k=3, chunk=CHUNK).astype(np.float32),
    }
    log("맵 3종 완료")

    # 융합 점수 — 보정은 train none 만
    comps_tr = [a_map(pad_tr).reshape(len(pad_tr), -1).max(1),
                line_density_max(pad_tr, length=L_LINE, n_orient=8, chunk=CHUNK,
                                 min_dies=L_LINE),
                local_fail_count_max(pad_tr, k=3, chunk=CHUNK)]
    comps_te = [maps["A"].reshape(len(x), -1).max(1),
                maps["B"].reshape(len(x), -1).max(1),
                maps["C"].reshape(len(x), -1).max(1)]
    score = fisher([ecdf_percentile(a, b) for a, b in zip(comps_tr, comps_te)])
    log("융합 점수 완료")

    meta = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    np.savez_compressed(
        OUT / "pipeline_maps_pad.npz",
        cache_index=keep_global.astype(np.int64), y=yte[keep_local].astype(np.int8),
        die_size=meta["die_size"][keep_global].astype(np.float32),
        lot_name=meta["lot_name"][keep_global], wafer=x,
        A_radialcal_k2k5=maps["A"].astype(np.float16),
        B_line_L11_full=maps["B"].astype(np.float16),
        C_k2_k3=maps["C"].astype(np.float16),
        score_fusion3=score.astype(np.float32), classes=np.array(NAMES))
    log("저장 완료 → %s" % (OUT / "pipeline_maps_pad.npz"))

    ysub = yte[keep_local]
    for cname in ("Scratch", "Center", "Loc", "Edge-Ring"):
        figure_for_class(x, maps, score, ysub, NAMES.index(cname),
                         FIG / ("cases_%s.png" % cname.lower().replace("-", "")))
        log("그림 저장: cases_%s.png" % cname.lower())

    (FIG / "export_summary.json").write_text(json.dumps({
        "n_exported": int(len(x)), "n_defect": int((ysub != 0).sum()),
        "n_normal": int((ysub == 0).sum()),
        "coords": "pad (original size, centred), 64x64",
        "maps": {"A_radialcal_k2k5": "반경 대역 보정 k² 5x5 — 전반",
                 "B_line_L11_full": "온전창 선 L=11 — Scratch",
                 "C_k2_k3": "k² 3x3 — Center"},
        "score": "3-signal Fisher fusion, blocked AUPR 0.8325, FPR@95TPR 0.1719",
        "note": "픽셀 단위 정답 없음 — 정성 평가 전용 (기획서 §9.3)",
    }, ensure_ascii=False, indent=2))
    log("완료")
