"""추론 시 증강 (TTA). 학습 때 쓴 대칭을 추론에서도 쓴다.

이 프로젝트에서 가장 크게 통한 기법은 dihedral 증강(+0.101)이고, 다른 모든 기법을
합친 것보다 컸다. 웨이퍼의 회전, 반사 대칭이 실제 구조라는 뜻이다. 그렇다면
추론에서도 쓸 수 있다 — 한 웨이퍼의 변환본들을 각각 넣고 확률을 평균한다.
라벨이 변환에 불변이라 역변환이 필요 없다. **학습이 전혀 없다.**

표현마다 맞는 변환이 다르다. 카르테시안(pad, resize)에는 dihedral 8개.
극좌표에는 열 방향 순환이동과 뒤집기 — rot90 을 걸면 반지름 축과 각도 축이
섞여 의미가 깨진다.

미리 하는 예측: **극좌표 모델은 TTA 이득이 작아야 한다.** 극좌표에서 회전은
평행이동이고 ResNet 의 global average pooling 이 그것을 근사적 불변으로 만들어
이미 갖고 있는 성질이기 때문이다. 틀리면 극좌표가 의도한 불변성을 실제로는
제공하지 못한다는 뜻이다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def dihedral_variants(x: np.ndarray, flips: bool = True):
    """(N,H,W) 배열의 dihedral 변환본을 내놓는다. 색인 재배열뿐이라 값이 안 변한다."""
    for k in range(4):
        r = np.rot90(x, k, axes=(1, 2))
        yield np.ascontiguousarray(r)
        if flips:
            yield np.ascontiguousarray(r[:, :, ::-1])


def angular_variants(x: np.ndarray, n: int = 8, flips: bool = False):
    """극좌표용. 열(각도) 축만 순환이동한다. 행(반지름)은 건드리지 않는다."""
    width = x.shape[2]
    for i in range(n):
        s = np.ascontiguousarray(np.roll(x, i * width // n, axis=2))
        yield s
        if flips:
            yield np.ascontiguousarray(s[:, :, ::-1])


def average_probs(parts) -> np.ndarray:
    parts = list(parts)
    if not parts:
        raise ValueError("평균낼 확률이 없다")
    return np.mean(parts, axis=0)


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def cache_tta_logits(ckpt: str, cache: str, splits: str, out_dir: str, tag: str,
                     scheme: str = "dihedral", n_angle: int = 8, flips: bool = True,
                     batch_size: int = 512, backbone: str = "resnet18",
                    density_ks=(), density_shuffle_seed=None, line_ls=()) -> str:
    """TTA 평균 확률을 로그로 되돌려 a6 와 같은 npz 형식으로 저장한다.

    확률을 평균한 뒤 log 를 취한다. 앙상블 도구가 로짓에 softmax 를 다시 걸어도
    같은 분포가 나오도록 하기 위해서다.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _runtime import setup
    setup("logits")
    import torch
    from a3_train_wm811k_cls import build_model, encode_device

    density_ks, line_ls = tuple(density_ks), tuple(line_ls)
    path = Path(out_dir) / f"{tag}_logits.npz"
    if path.exists():
        return str(path)

    d, sp = np.load(cache), np.load(splits)
    n_cls = len(d["classes"])
    X, y_all = d["X"], d["y"]          # npz 지연 로딩을 배치 루프 밖에서 한 번만
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(num_classes=n_cls, pretrained=False, backbone=backbone,
                        in_channels=3 + len(density_ks) + len(line_ls))
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()

    def variants(batch):
        if scheme == "dihedral":
            return dihedral_variants(batch, flips=flips)
        if scheme == "angular":
            return angular_variants(batch, n=n_angle, flips=flips)
        raise ValueError(f"모르는 변환 방식: {scheme}")

    store = {}
    with torch.no_grad():
        for split in ("val", "test"):
            idx = sp[split]
            out = np.empty((len(idx), n_cls), dtype=np.float32)
            for i in range(0, len(idx), batch_size):
                b = idx[i:i + batch_size]
                raw = X[b]
                # 밀도 맵은 dihedral 과 정확히 교환되므로 변환본 위에서 그대로 다시 계산한다.
                probs = [
                    _softmax(model(encode_device(v, density_ks, device,
                                                 density_shuffle_seed, line_ls))
                             .cpu().numpy().astype(np.float64))
                    for v in variants(raw)
                ]
                out[i:i + len(b)] = np.log(np.clip(average_probs(probs), 1e-12, None))
            store[f"{split}_logits"] = out
            store[f"{split}_y"] = y_all[idx].astype(np.int64)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **store)
    return str(path)


def main() -> None:
    import argparse
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from a4_eval_wm811k_cls import evaluate

    p = argparse.ArgumentParser(description="TTA 로짓 캐시와 비교")
    p.add_argument("--manifest", default="docs/experiments/ensemble_members.json")
    p.add_argument("--out-dir", default="result/posthoc")
    p.add_argument("--splits", default="data/wm811k/cache/splits_v1.npz")
    p.add_argument("--only", nargs="*", default=None, help="이 태그들만")
    p.add_argument("--rotations-only", action="store_true", help="변환 4개(회전만)")
    a = p.parse_args()

    entries = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    if a.only:
        entries = [e for e in entries if e["tag"] in set(a.only)]

    print("%-22s %10s %10s %9s" % ("run", "기본", "TTA", "차이"))
    rows = []
    for e in entries:
        scheme = "angular" if e["representation"].startswith("polar") else "dihedral"
        suffix = "tta4" if a.rotations_only else "tta"
        tta_tag = f"{e['tag']}_{suffix}"
        cache_tta_logits(e["ckpt"], e["cache"], a.splits, a.out_dir, tta_tag,
                         scheme=scheme, flips=not a.rotations_only,
                         backbone=e.get("backbone", "resnet18"),
                         density_ks=tuple(e.get("density_ks", ())),
                         density_shuffle_seed=e.get("density_shuffle"),
                         line_ls=tuple(e.get("line_ls", ())))

        base = np.load(Path(a.out_dir) / f"{e['tag']}_logits.npz")
        tta = np.load(Path(a.out_dir) / f"{tta_tag}_logits.npz")
        nc = base["test_logits"].shape[1]
        b_f1 = evaluate(base["test_y"], base["test_logits"].argmax(1), nc)["macro_f1"]
        t_f1 = evaluate(tta["test_y"], tta["test_logits"].argmax(1), nc)["macro_f1"]
        rows.append((e["tag"], e["representation"], b_f1, t_f1))
        print("%-22s %10.4f %10.4f %+9.4f" % (e["tag"], b_f1, t_f1, t_f1 - b_f1))

    if rows:
        deltas = [t - b for _, _, b, t in rows]
        print("\n평균 이득 %+.4f   부호 일관 %s (%d/%d 개선)" % (
            np.mean(deltas), "예" if all(d > 0 for d in deltas) else "아니오",
            sum(d > 0 for d in deltas), len(deltas)))
        print("\n표현별 평균 이득")
        for rep in sorted({r for _, r, _, _ in rows}):
            v = [t - b for _, r, b, t in rows if r == rep]
            print("  %-10s %+.4f  (n=%d)" % (rep, np.mean(v), len(v)))


if __name__ == "__main__":
    main()
