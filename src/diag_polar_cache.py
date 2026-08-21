"""극좌표 캐시가 학습에 쓸 만한지 확인한다. E15 가 이걸로 46분을 쓰기 전에."""
import numpy as np

NAMES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full"]

pol = np.load("data/wm811k/cache/wm811k_64polar.npz", allow_pickle=True)
pad = np.load("data/wm811k/cache/wm811k_64pad.npz", allow_pickle=True)

Xp, yp = pol["X"], pol["y"]
print("shape %s  dtype %s  값 %s" % (Xp.shape, Xp.dtype, np.unique(Xp)))
print("라벨 분포가 pad 캐시와 같은가:", np.array_equal(yp, pad["y"]))
print("lot_name 정렬 같은가:", np.array_equal(pol["lot_name"], pad["lot_name"]))
print("split 정렬 같은가:", np.array_equal(pol["split"], pad["split"]))

die = (Xp > 0)
print()
print("다이 점유율 (0 이 아닌 셀의 비율)")
print("  극좌표 평균 %.3f   pad 평균 %.3f" % (die.mean(), (pad["X"] > 0).mean()))
print("  극좌표에서 다이가 아예 없는 웨이퍼: %d 장" % (die.reshape(len(Xp), -1).sum(1) == 0).sum())

print()
print("클래스별 불량 다이 비율 — 클래스가 구분되는 신호가 남아 있는가")
print("%-11s %8s %12s %12s" % ("클래스", "n", "극좌표", "pad"))
rng = np.random.default_rng(0)
for c in range(9):
    idx = np.flatnonzero(yp == c)
    if len(idx) == 0:
        continue
    take = idx if len(idx) <= 2000 else rng.choice(idx, 2000, replace=False)
    fp = (Xp[take] == 2).sum((1, 2)) / np.maximum(1, (Xp[take] > 0).sum((1, 2)))
    fd = (pad["X"][take] == 2).sum((1, 2)) / np.maximum(1, (pad["X"][take] > 0).sum((1, 2)))
    print("%-11s %8d %12.4f %12.4f" % (NAMES[c], len(idx), np.median(fp), np.median(fd)))

print()
print("Edge-Ring 이 실제로 행 방향 띠가 되는가 (극좌표의 핵심 주장)")
for c, name in [(4, "Edge-Ring"), (1, "Center"), (7, "Scratch")]:
    idx = np.flatnonzero(yp == c)[:300]
    prof = (Xp[idx] == 2).mean((0, 2))          # 행(반지름)별 불량 비율
    top = int(np.argmax(prof))
    print("  %-10s 불량이 가장 몰린 행 %2d/64 (비율 %.3f), 중심부 행0~10 평균 %.3f, 외곽 행53~63 평균 %.3f"
          % (name, top, prof[top], prof[:11].mean(), prof[53:].mean()))
