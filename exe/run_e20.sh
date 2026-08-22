#!/usr/bin/env bash
# E20 — 국소 불량 밀도 입력 채널.
#
# 반증 조건은 실행 전에 docs/experiments/candidate/local_density_channels.md 에 박았다.
# cls_e17.yaml(= translate 증강 조건)과 config diff 가 density_ks, tag, seed 뿐이라
# 대조군은 재학습하지 않고 e17_translate_s0~s3 4 seed(평균 0.7645)를 그대로 쓴다.
#
# 사용: ./exe/run_e20.sh dens_s0 dens_s1
#       ./exe/run_e20.sh dens_s2 shuf_s0
source "$(dirname "$0")/_common.sh"
CACHE=data/wm811k/cache/wm811k_64pad.npz

for spec in "$@"; do
  case "$spec" in
    dens_s*) seed="${spec#dens_s}"; extra="" ;;
    # 셔플 대조군: 밀도 채널을 다른 웨이퍼에서 가져온다. 채널 수와 통계는 같고
    # one-hot 과의 대응만 깨진다 — 이득이 정보 때문인지 용량 때문인지를 가른다.
    shuf_s*) seed="${spec#shuf_s}"; extra="--density-shuffle 12345" ;;
    *) echo "모르는 spec: $spec"; exit 1 ;;
  esac
  tag="e20_${spec}"
  if [ -f "result/cls_baseline/${tag}_best_test.json" ]; then
    echo "이미 있음: $tag"
    continue
  fi
  echo "=========== $tag ==========="
  $PY src/a3_train_wm811k_cls.py --config configs/cls_e17.yaml \
      --tag "$tag" --seed "$seed" --extra-augment translate --density-ks 5 7 $extra
  $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" \
      --split test --cache "$CACHE" --density-ks 5 7 $extra
done
echo "=========== E20 DONE: $* ==========="
