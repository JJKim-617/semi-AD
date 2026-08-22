"""잔차/밀도 맵을 파일로 내보내고 대표 사례를 그린다. 파이프라인 2~3단계가 쓸 산출물.

기획서 §9.2 — 다이 단위 맵이 1급 산출물이다.
스칼라만 최적화하면 뭉개는 방식만 튜닝하게 되고 하류가 쓸 맵의 품질은 방치된다.

## 무엇을 내보내는가 (두 장)

실측으로 정해졌다(`docs/experiments/candidate/ood_residual_pooling.md` 판정 절).

1. **`density` — 창 5x5 안 불량 다이 비율.** template 이 없다. 학습이 0 이다.
2. **`radial_residual` — 반경 template 잔차를 같은 창으로 평활한 것.**
   정상이 이미 자주 죽는 자리(정중앙 다이, 가장자리 링)를 깎아 준다.
3. **`radial_cal` — 창 7x7 밀도를 같은 반경 대역 정상 다이 분포에서의 백분위로 바꾼 것.**
   **지금 가장 강한 탐지기다**(AUPR 0.7407, FPR@95TPR 0.3603) 그리고
   Scratch/Center/Loc 을 동시에 최고로 만든 유일한 arm 이다.
   값이 "이 자리 치고 얼마나 드문가" 라서 사람이 읽기도 쉽다.

셋 다 내보내는 이유: 어느 쪽이 하류 8종 분류를 돕는지는 **아직 모른다.**
기획서 §9.4 의 간접 평가로 판정할 일이고, 그러려면 다 있어야 한다.

## 좌표계

**pad(원 좌표)** 를 쓴다. 리사이즈 좌표는 작은 웨이퍼를 확대하므로 5x5 창이
웨이퍼마다 다른 물리적 크기를 덮는다. 국소 창을 쓰는 순간 그건 교란이다.
실측에서도 T-RESIZE 계열이 같은 창 크기에서 뒤졌다.

## 픽셀 단위 정답은 만들지 않는다

기획서 §9.3. 여기 그림은 **정성 평가**용이다. 합성 마스크로 P-AUROC 를 내지 않는다.
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
from a22_ood_template import assert_one_class, fit_radial_template  # noqa: E402
from a29_radial_calibration import (  # noqa: E402
    band_index,
    calibrate_map,
    fit_band_reference,
)
from a24_ood_residual import (  # noqa: E402
    local_fail_density_map,
    local_fail_density_max,
    residual_map,
    smooth_map,
)

OUT = Path("result/ood/o1_maps")
FIG = Path("docs/research/ood_template/evidence")
NAMES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch",
         "Near-full"]
WAFER_CMAP = ListedColormap(["#f2f2f2", "#3b6ea5", "#d1495b"])  # 0 없음 / 1 정상 / 2 불량
K_MAP = 5
K_SCORE = 7
T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def select_cases(score, y, cls: int, n: int = 3) -> dict:
    """한 결함 클래스에서 잘 잡은 것/놓친 것, 그리고 정상 쪽 대조를 고른다.

    - `caught` 점수 상위 n — 제대로 뜨는 사례
    - `missed` 점수 하위 n — **왜 안 잡히는지 보려면 이쪽을 봐야 한다**
    - `false_alarm` 정상 중 점수 상위 n — 헛경보의 정체
    - `typical_normal` 정상 중 중앙값 근처 n — 기준선
    """
    score = np.asarray(score, np.float64)
    y = np.asarray(y)
    m = np.flatnonzero(y == cls)
    order = m[np.argsort(-score[m], kind="mergesort")]
    normal = np.flatnonzero(y == 0)
    n_order = normal[np.argsort(-score[normal], kind="mergesort")]
    mid = len(n_order) // 2
    half = min(n, len(n_order))
    return {
        "caught": order[:n],
        "missed": order[len(order) - min(n, len(order)):][::-1],
        "false_alarm": n_order[:min(n, len(n_order))],
        "typical_normal": n_order[mid:mid + half],
    }


def subset_for_export(y, n_normal: int = 5000, seed: int = 0) -> np.ndarray:
    """결함은 전부, 정상은 표본으로. 118,595장 전부 저장하면 2 GB 다."""
    y = np.asarray(y)
    defect = np.flatnonzero(y != 0)
    normal = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    k = min(n_normal, len(normal))
    keep = rng.choice(normal, k, replace=False) if k < len(normal) else normal
    return np.sort(np.concatenate([defect, keep]))


def _panel(ax, img, title, kind, die=None):
    """다이가 없는 칸만 비운다.

    처음에는 `img == 0` 을 비웠는데, 그러면 **깨끗한(밀도 0) 다이**와 **다이가 없는 칸**이
    같은 색으로 보인다. 국소화 그림에서 그 둘을 못 가르면 그림이 거짓말을 한다.
    """
    if kind == "wafer":
        ax.imshow(img, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
    elif kind == "density":
        ax.imshow(np.where(die, img, np.nan), cmap="inferno", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_facecolor("#e8e8e8")
    elif kind == "surprisal":
        # 백분위 u 를 그대로 그리면 1 근처에 몰려 온통 노랗다 (다이 800개의 최대는
        # 귀무에서도 1-1/800 근처다). -log10(1-u) 로 펴야 사람이 읽을 수 있다.
        v = -np.log10(np.clip(1.0 - img, 1e-6, 1.0))
        ax.imshow(np.where(die, v, np.nan), cmap="inferno", vmin=0, vmax=6,
                  interpolation="nearest")
        ax.set_facecolor("#e8e8e8")
    else:
        v = np.abs(img[die]).max() or 1.0
        ax.imshow(np.where(die, img, np.nan), cmap="RdBu_r", vmin=-v, vmax=v,
                  interpolation="nearest")
        ax.set_facecolor("#e8e8e8")
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def figure_for_class(x, dens, cal, score, y, cls: int, path: Path, n=3):
    cases = select_cases(score, y, cls, n)
    groups = [("caught (top score)", cases["caught"]),
              ("MISSED (bottom score)", cases["missed"]),
              ("normal, false alarm", cases["false_alarm"]),
              ("normal, typical", cases["typical_normal"])]
    rows = sum(len(g[1]) for g in groups)
    fig, axes = plt.subplots(rows, 3, figsize=(6.2, 2.05 * rows))
    axes = np.atleast_2d(axes)
    r = 0
    for gname, idxs in groups:
        for i in idxs:
            _panel(axes[r, 0], x[i], "%s\n%s  fail=%.3f" % (
                gname, NAMES[y[i]], fail_ratio(x[i][None])[0]), "wafer")
            die = x[i] > 0
            _panel(axes[r, 1], dens[i],
                   "density %dx%d (max %.2f)" % (K_MAP, K_MAP, dens[i].max()),
                   "density", die)
            _panel(axes[r, 2], cal[i],
                   "radius-calibrated surprisal", "surprisal", die)
            r += 1
    fig.suptitle("%s — original (pad) coords. score = radius-calibrated density max %dx%d"
                 % (NAMES[cls], K_SCORE, K_SCORE), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_radial_template(t, path: Path):
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    centres = (np.arange(len(t.p)) + 0.5) / len(t.p)
    ax.plot(centres, t.p, marker="o", ms=3)
    ax.set_xlabel("normalized radius (wafer's own die extent)")
    ax.set_ylabel("P(fail) among normal wafers")
    ax.set_title("Normal wafers fail at BOTH centre and edge\n"
                 "centre-most die fails 58.9%% of the time", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    tr, te = sp["train"], sp["test"]
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y_all = d["y"].astype(np.int64)
    trn = tr[y_all[tr] == 0]
    assert_one_class(y_all[trn])
    X = d["X"]
    x_tr = np.ascontiguousarray(X[trn])
    t_rad = fit_radial_template(x_tr, y=y_all[trn], n_bins=32, smoothing=1.0)
    log("반경 template 적합 (정상 %d장만). 전역 불량률 %.5f" % (len(trn), t_rad.global_rate))
    figure_radial_template(t_rad, FIG / "radial_template.png")

    yte = y_all[te]
    keep_local = subset_for_export(yte, n_normal=5000, seed=0)
    keep_global = te[keep_local]
    x = np.ascontiguousarray(X[keep_global])
    del X, d
    log("내보낼 부분집합 %d장 (결함 %d + 정상 %d)"
        % (len(keep_local), int((yte[keep_local] != 0).sum()),
           int((yte[keep_local] == 0).sum())))

    dens = local_fail_density_map(x, k=K_MAP).astype(np.float32)
    rres = smooth_map(residual_map(x, t_rad, mode="nll"), x > 0, k=K_MAP).astype(np.float32)
    v_tr = local_fail_density_map(x_tr, k=K_SCORE).astype(np.float32)
    ref = fit_band_reference(v_tr, x_tr, y=y_all[trn], n_bands=32, seed=0)
    del v_tr
    cal = calibrate_map(local_fail_density_map(x, k=K_SCORE),
                        band_index(x, 32), ref).astype(np.float32)
    score = cal.reshape(len(cal), -1).max(1)
    log("맵 3종 + 채점 완료 (점수 = 반경 보정 밀도의 최대)")

    meta = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    np.savez_compressed(
        OUT / "o1_maps_pad.npz",
        cache_index=keep_global.astype(np.int64),
        y=yte[keep_local].astype(np.int8),
        die_size=meta["die_size"][keep_global].astype(np.float32),
        lot_name=meta["lot_name"][keep_global],
        wafer=x,
        density_k5=dens.astype(np.float16),
        radial_residual_k5=rres.astype(np.float16),
        radial_cal_k7=cal.astype(np.float16),
        score_radial_cal_max_k7=score.astype(np.float32),
        radial_template=t_rad.p.astype(np.float32),
        classes=np.array(NAMES))
    log("맵 저장 완료 → %s" % (OUT / "o1_maps_pad.npz"))

    ysub = yte[keep_local]
    for cname in ("Scratch", "Center", "Loc", "Edge-Ring"):
        figure_for_class(x, dens, cal, score, ysub, NAMES.index(cname),
                         FIG / ("cases_%s.png" % cname.lower().replace("-", "")))
        log("그림 저장: cases_%s.png" % cname.lower())

    (FIG / "export_summary.json").write_text(json.dumps({
        "n_exported": int(len(keep_local)),
        "n_defect": int((ysub != 0).sum()),
        "n_normal": int((ysub == 0).sum()),
        "coords": "pad (original size, centred), 64x64",
        "maps": {"density_k5": "창 5x5 불량 다이 비율, template 없음",
                 "radial_residual_k5": "반경 template NLL 잔차를 5x5 평활",
                 "radial_cal_k7": "창 7x7 밀도의 반경 대역별 정상 백분위 (최선)"},
        "ranking_score": "radius-calibrated density max 7x7 (AUPR 0.7407)",
        "note": "픽셀 단위 정답 없음 — 정성 평가 전용 (기획서 §9.3)",
    }, ensure_ascii=False, indent=2))
    log("완료")
