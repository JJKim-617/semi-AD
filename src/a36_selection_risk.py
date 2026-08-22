"""선택 편향 감사 — test 를 여덟 사이클 보고 고른 것이 얼마나 위험한가.

반증 조건은 `candidate/ood_selection_risk.md` 에 실행 전에 박았다.
공식 test 를 **lot 단위로 두 반쪽**으로 갈라 같은 arm 을 다시 잰다.
**아무것도 다시 적합하지 않는다** — 보정 참조는 train-none 그대로다.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import aupr_blocked, fail_ratio  # noqa: E402
from a22_ood_template import (  # noqa: E402
    assert_one_class,
    fit_position_template,
    score_nll,
)
from a23_ood_template_eval import ecdf_percentile, fisher  # noqa: E402
from a24_ood_residual import (  # noqa: E402
    local_fail_count_map,
    local_fail_count_max,
    local_fail_density_max,
)
from a27_line_filter import line_density_max  # noqa: E402
from a29_radial_calibration import band_index, calibrate_map, fit_band_reference  # noqa: E402

OUT = Path("result/ood/o1_selection_risk")
CHUNK, N_BANDS, L_LINE = 4000, 32, 11
T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def lot_half(lot_names: np.ndarray) -> np.ndarray:
    """lot 이름 해시로 두 반쪽. **lot 단위여야** 같은 lot 의 쌍둥이가 양쪽에 안 갈린다."""
    return np.array([int(hashlib.md5(str(n).encode()).hexdigest(), 16) & 1
                     for n in lot_names], dtype=np.int64)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sp = np.load("data/wm811k/cache/splits_v1.npz")
    tr, te = sp["train"], sp["test"]
    d = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)
    y = d["y"].astype(np.int64)
    lots = d["lot_name"]
    trn = tr[y[tr] == 0]
    assert_one_class(y[trn])
    X = d["X"]
    pad_tr = np.ascontiguousarray(X[trn])
    pad_te = np.ascontiguousarray(X[te])
    res_tr = None
    del X
    dres = np.load("data/wm811k/cache/wm811k_64.npz", allow_pickle=True)
    Xr = dres["X"]
    res_tr = np.ascontiguousarray(Xr[trn])
    res_te = np.ascontiguousarray(Xr[te])
    del Xr, dres, d
    yte = y[te]
    is_def = (yte != 0).astype(np.int64)
    half = lot_half(lots[te])
    log("적재 완료. 반쪽 크기 %d / %d, 결함비율 %.4f / %.4f"
        % (int((half == 0).sum()), int((half == 1).sum()),
           float(is_def[half == 0].mean()), float(is_def[half == 1].mean())))

    v_tr = local_fail_count_map(pad_tr, k=5, chunk=CHUNK).astype(np.float32)
    ref = fit_band_reference(v_tr, pad_tr, y=y[trn], n_bands=N_BANDS, seed=0)
    del v_tr

    def a_score(x):
        out = np.empty(len(x))
        for a in range(0, len(x), CHUNK):
            xb = x[a:a + CHUNK]
            u = calibrate_map(local_fail_count_map(xb, k=5, chunk=CHUNK),
                              band_index(xb, N_BANDS), ref)
            out[a:a + len(xb)] = u.reshape(len(xb), -1).max(1)
        return out

    A_te, A_tr = a_score(pad_te), a_score(pad_tr)
    B_te = line_density_max(pad_te, length=L_LINE, n_orient=8, chunk=CHUNK, min_dies=L_LINE)
    B_tr = line_density_max(pad_tr, length=L_LINE, n_orient=8, chunk=CHUNK, min_dies=L_LINE)
    C_te = local_fail_count_max(pad_te, k=3, chunk=CHUNK)
    C_tr = local_fail_count_max(pad_tr, k=3, chunk=CHUNK)
    log("요소 3종 완료")

    t_res = fit_position_template(res_tr, y=y[trn], smoothing=1.0)

    def fu(keys):
        m = {"A": (A_tr, A_te), "B": (B_tr, B_te), "C": (C_tr, C_te)}
        return fisher([ecdf_percentile(*m[k]) for k in keys])

    scores = {
        "채택 fuse3 (A+B+C)": fu(["A", "B", "C"]),
        "채택 A+B (6차)": fu(["A", "B"]),
        "탐색 B+C": fu(["B", "C"]),
        "채택 A 단독 (5차)": A_te,
        "채택 die k=7 (2차 바닥)": local_fail_density_max(pad_te, k=7, chunk=CHUNK),
        "E0 불량 다이 비율 (원 바닥)": fail_ratio(pad_te),
        "기각 T-RESIZE S_nll (2차)": score_nll(res_te, t_res),
        "기각 선 L=7 가드없음 (3차)": line_density_max(pad_te, length=7, n_orient=8,
                                                chunk=CHUNK),
    }
    log("arm %d개 준비 완료" % len(scores))

    rows = {}
    for name, s in scores.items():
        full = aupr_blocked(s, is_def)
        h0 = aupr_blocked(s[half == 0], is_def[half == 0])
        h1 = aupr_blocked(s[half == 1], is_def[half == 1])
        rows[name] = {"full": full, "half0": h0, "half1": h1, "gap": abs(h0 - h1)}
        log("%-28s 전체 %.4f  반쪽 %.4f / %.4f  차이 %.4f" % (name, full, h0, h1, abs(h0 - h1)))

    names = list(rows)
    print("\n== 반쪽별 blocked AUPR ==")
    print("%-28s %9s %9s %9s %9s" % ("arm", "전체", "반쪽1", "반쪽2", "|차이|"))
    for n in sorted(names, key=lambda k: -rows[k]["full"]):
        r = rows[n]
        print("%-28s %9.4f %9.4f %9.4f %9.4f" % (n, r["full"], r["half0"], r["half1"], r["gap"]))

    o0 = sorted(names, key=lambda k: -rows[k]["half0"])
    o1 = sorted(names, key=lambda k: -rows[k]["half1"])
    print("\n== 조건 1: 두 반쪽의 최고 arm ==")
    print("  반쪽1 최고: %s (%.4f)" % (o0[0], rows[o0[0]]["half0"]))
    print("  반쪽2 최고: %s (%.4f)" % (o1[0], rows[o1[0]]["half1"]))
    print("  → %s" % ("같다" if o0[0] == o1[0] else "**다르다 — 최선을 한 arm 으로 못 지목한다**"))

    top = o0[0]
    second_full = sorted(names, key=lambda k: -rows[k]["full"])[1]
    lead = rows[sorted(names, key=lambda k: -rows[k]["full"])[0]]["full"] - rows[second_full]["full"]
    print("\n== 조건 2: 최고 arm 의 반쪽 간 차이 대 2위와의 격차 ==")
    print("  최고 arm 반쪽 차이 %.4f, 1위-2위 격차 %.4f  → %s"
          % (rows[top]["gap"], lead,
             "순위가 잡음 안이다" if rows[top]["gap"] > lead else "격차가 변동보다 크다"))

    rho = spearman([rows[n]["half0"] for n in names], [rows[n]["half1"] for n in names])
    print("\n== 조건 3: 반쪽 간 순위 Spearman = %.4f  → %s"
          % (rho, "불안정" if rho < 0.9 else "안정"))

    floor = rows["E0 불량 다이 비율 (원 바닥)"]["full"]
    best = rows[sorted(names, key=lambda k: -rows[k]["full"])[0]]["full"]
    gain = best - floor
    maxgap = max(r["gap"] for r in rows.values())
    print("\n== 조건 4: 바닥 대비 이득 %.4f, 최대 반쪽 변동 %.4f, 배수 %.1f  → %s"
          % (gain, maxgap, gain / max(maxgap, 1e-9),
             "큰 주장 안전, 미세 순위만 위험" if gain > 5 * maxgap else "**전체 주장까지 흔들린다**"))

    print("\n== 조건 5: 기각한 arm 이 어느 반쪽에서든 채택 arm 을 넘는가 ==")
    adopted = [n for n in names if n.startswith("채택")]
    worst_adopted = min(min(rows[n]["half0"], rows[n]["half1"]) for n in adopted)
    viol = [n for n in names if n.startswith("기각")
            and max(rows[n]["half0"], rows[n]["half1"]) > worst_adopted]
    print("  채택 arm 의 반쪽 최저 %.4f, 이를 넘는 기각 arm: %s"
          % (worst_adopted, viol if viol else "없음"))

    (OUT / "selection_risk.json").write_text(
        json.dumps({"rows": rows, "spearman": rho, "gain": gain, "max_gap": maxgap},
                   ensure_ascii=False, indent=2))
    log("저장 완료 → %s" % OUT)
