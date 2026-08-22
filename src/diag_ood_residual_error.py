"""95% 재현율에 남은 오류가 모델 탓인가 라벨 탓인가 — 진단.

반증 조건은 `candidate/ood_residual_error.md` 에 실행 전에 박았다.
**판정은 fuse3 점수를 쓰지 않고 내린다** — 자기 점수로 자기 오류를 설명하면 순환이다.
문턱 결정에만 쓰고, 집단 성격은 **불량 다이 개수/비율**이라는 독립적인 자로 잰다.
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from a21_ood_metrics import fail_ratio, operating_points  # noqa: E402
from a22_ood_template import assert_one_class  # noqa: E402
from a23_ood_template_eval import NAMES, ecdf_percentile, fisher  # noqa: E402
from a24_ood_residual import local_fail_count_map, local_fail_count_max  # noqa: E402
from a27_line_filter import line_density_max  # noqa: E402
from a29_radial_calibration import band_index, calibrate_map, fit_band_reference  # noqa: E402

OUT = Path("result/ood/o1_residual_error")
CHUNK, N_BANDS, L_LINE = 4000, 32, 11
T0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


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
    del X, d
    yte = y[te]
    is_def = (yte != 0).astype(np.int64)
    lot_te = lots[te]
    log("적재 완료")

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

    pairs = [(a_score(pad_tr), a_score(pad_te)),
             (line_density_max(pad_tr, length=L_LINE, n_orient=8, chunk=CHUNK,
                               min_dies=L_LINE),
              line_density_max(pad_te, length=L_LINE, n_orient=8, chunk=CHUNK,
                               min_dies=L_LINE)),
             (local_fail_count_max(pad_tr, k=3, chunk=CHUNK),
              local_fail_count_max(pad_te, k=3, chunk=CHUNK))]
    score = fisher([ecdf_percentile(a, b) for a, b in pairs])
    log("fuse3 점수 완료")

    op = operating_points(score, is_def, (0.95,))[0]
    thr = op["threshold"]
    flagged = score >= thr
    fp = flagged & (is_def == 0)
    fn = (~flagged) & (is_def == 1)
    log("문턱 %.4f — 헛경보 정상 %d장, 놓친 결함 %d장"
        % (thr, int(fp.sum()), int(fn.sum())))

    # --- 독립적인 자 ---------------------------------------------------------
    ratio = fail_ratio(pad_te)
    true_def_q25 = float(np.percentile(ratio[is_def == 1], 25))
    normal_median = float(np.median(ratio[is_def == 0]))
    print("\n== 독립적인 자: 불량 다이 비율 ==")
    print("  참 결함 하위 25%% = %.4f, 정상 중앙값 = %.4f" % (true_def_q25, normal_median))

    frac1 = float((ratio[fp] > true_def_q25).mean())
    print("\n== 조건 1: 헛경보 정상이 독립적인 자로도 결함처럼 보이는가 ==")
    print("  헛경보 정상 중 '참 결함 하위 25%% 보다 불량률이 높은' 비율: %.4f" % frac1)
    print("  (참고) 전체 정상 중 같은 비율: %.4f" % float((ratio[is_def == 0] > true_def_q25).mean()))
    print("  → %s" % ("**과반. 천장은 상당 부분 라벨 탓이다**" if frac1 > 0.5
                     else "과반 미만. 모델 탓이 크다"))

    frac2 = float((ratio[fn] < normal_median).mean())
    print("\n== 조건 2: 놓친 결함이 웨이퍼맵에 볼 것이 있는가 ==")
    print("  놓친 결함 중 '정상 중앙값보다 불량률이 낮은' 비율: %.4f" % frac2)
    print("  놓친 결함 불량률 분위 [10,50,90]: %s"
          % np.round(np.percentile(ratio[fn], [10, 50, 90]), 4))
    print("  전체 결함 불량률 분위 [10,50,90]: %s"
          % np.round(np.percentile(ratio[is_def == 1], [10, 50, 90]), 4))
    print("  → %s" % ("**과반. 표현을 바꿔도 못 잡는다**" if frac2 > 0.5
                     else "과반 미만. 아직 볼 것이 남아 있다"))

    print("\n== 조건 3: 놓친 결함의 클래스 분포 ==")
    print("%-11s %8s %8s %10s %10s" % ("클래스", "전체", "놓침", "놓침률", "전체대비"))
    rows = {}
    for c in range(1, 9):
        tot = int((yte == c).sum())
        miss = int((fn & (yte == c)).sum())
        rows[NAMES[c]] = {"total": tot, "missed": miss,
                          "miss_rate": miss / max(tot, 1),
                          "share": miss / max(int(fn.sum()), 1),
                          "share_all": tot / int(is_def.sum())}
        print("%-11s %8d %8d %10.4f %10.4f"
              % (NAMES[c], tot, miss, rows[NAMES[c]]["miss_rate"], rows[NAMES[c]]["share"]))
    print("  (놓침률 = 그 클래스에서 놓친 비율, 전체대비 = 놓친 것 중 그 클래스의 몫)")

    print("\n== 조건 4: 헛경보가 lot 에 몰려 있는가 ==")
    cnt = collections.Counter(lot_te[fp])
    top = cnt.most_common(10)
    top_share = sum(c for _, c in top) / max(int(fp.sum()), 1)
    n_lots_fp = len(cnt)
    n_lots_all = len(set(lot_te[is_def == 0]))
    print("  헛경보가 걸린 lot %d개 (전체 정상 lot %d개)" % (n_lots_fp, n_lots_all))
    print("  상위 10개 lot 이 헛경보의 %.4f 를 차지" % top_share)
    print("  상위 5개 lot: %s" % [(str(l), c) for l, c in top[:5]])
    print("  → %s" % ("**lot 단위 현상이다**" if top_share > 0.20 else "웨이퍼 단위 현상이다"))

    (OUT / "residual_error.json").write_text(json.dumps({
        "threshold": thr, "n_fp": int(fp.sum()), "n_fn": int(fn.sum()),
        "cond1_frac": frac1, "cond2_frac": frac2, "per_class_missed": rows,
        "top10_lot_share": top_share, "n_lots_fp": n_lots_fp,
    }, ensure_ascii=False, indent=2))
    log("저장 완료 → %s" % OUT)
