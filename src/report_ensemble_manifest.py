"""새 실행을 앙상블 구성원 매니페스트에 붙인다 (18차).

`docs/experiments/ensemble_members.json` 은 `a17_tta`(TTA 로짓)와
`bench_ensemble`(앙상블 집계)이 함께 읽는 단일 목록이다. 손으로 넣다가 `backbone` 을
빠뜨리면 체크포인트를 **3채널 resnet18 로 잘못 읽는다.** 규칙을 코드로 두고 시험으로 박는다.

사용:
    python src/report_ensemble_manifest.py            # 끝난 e22/e23 실행을 전부 붙인다
    python src/report_ensemble_manifest.py --dry-run  # 무엇이 붙을지만 본다
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST = Path("docs/experiments/ensemble_members.json")
OUT_DIR = "result/cls_baseline"
CACHE_PAD = "data/wm811k/cache/wm811k_64pad.npz"

# 백본 이름에 밑줄과 숫자가 들어가므로(`efficientnet_b0`) 태그를 순진하게 쪼개면 깨진다.
# 아는 이름만 접두사로 맞춘다.
BACKBONES = ["shufflenet_v2", "mobilenet_v3", "efficientnet_b0", "convnext_tiny"]
E23_ARMS = {"rot": "rotate", "rottr": "rotate+translate"}


def entry_for_tag(tag: str) -> dict:
    """태그 하나에서 매니페스트 항목을 만든다. **모르는 태그는 거절한다.**

    조용히 resnet18 기본값을 씌우면 잘못된 모델로 로짓을 만들고, 그 사실이
    숫자에는 드러나지 않는다. 이 프로젝트에서 가장 비싼 종류의 사고다.
    """
    base = dict(tag=tag, ckpt=f"{OUT_DIR}/{tag}_best.pt",
                cache=CACHE_PAD, representation="pad64")
    if tag.startswith("e22_"):
        rest = tag[len("e22_"):]
        for bb in BACKBONES:
            if rest.startswith(bb + "_tr_s"):
                return {**base, "architecture": bb, "backbone": bb,
                        "augment": "translate"}
        raise ValueError(f"E22 태그에서 아는 백본을 못 찾았다: {tag}")
    if tag.startswith("e23_"):
        rest = tag[len("e23_"):]
        for arm, recipe in E23_ARMS.items():
            if rest.startswith(arm + "_s"):
                return {**base, "architecture": "resnet18", "backbone": "resnet18",
                        "augment": recipe}
        raise ValueError(f"E23 태그에서 아는 팔을 못 찾았다: {tag}")
    raise ValueError(f"이 도구가 모르는 태그다: {tag}")


def merge_entries(existing: list, new_tags) -> list:
    """기존 목록은 그대로 두고 **없는 것만 뒤에 붙인다.**

    기존 항목을 다시 쓰면 과거 결과의 재현이 깨진다. 두 번 돌려도 같은 결과여야 한다.
    """
    have = {e["tag"] for e in existing}
    out = list(existing)
    for t in new_tags:
        if t in have:
            continue
        out.append(entry_for_tag(t))
        have.add(t)
    return out


def finished_tags(prefixes=("e22_", "e23_"), out_dir: str = OUT_DIR) -> list:
    """**학습이 끝난** 실행만 고른다.

    `{tag}_best.pt` 는 val 이 좋아질 때마다 덮어써지므로 존재만으로는 부족하다.
    a4 평가가 남기는 `{tag}_best_test.json` 이 있어야 그 실행이 끝난 것이다.
    """
    d = Path(out_dir)
    tags = []
    for p in sorted(d.glob("*_best_test.json")):
        t = p.name[: -len("_best_test.json")]
        if t.startswith(prefixes):
            tags.append(t)
    return tags


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    path = Path(a.manifest)
    existing = json.loads(path.read_text(encoding="utf-8"))
    tags = finished_tags()
    merged = merge_entries(existing, tags)
    added = [e["tag"] for e in merged[len(existing):]]
    print(f"끝난 실행 {len(tags)}개, 새로 붙는 것 {len(added)}개")
    for t in added:
        print(f"  + {t}")
    if a.dry_run:
        print("(--dry-run 이라 쓰지 않았다)")
        return
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(f"[저장] {path}  총 {len(merged)}개")


if __name__ == "__main__":
    main()
