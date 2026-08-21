#!/usr/bin/env bash
# 앞선 실험과 미라벨 캐시가 끝난 뒤 자기지도 사전학습과 fine-tune 을 돌린다.
source "$(dirname "$0")/_common.sh"
while pgrep -f "run_e11.sh|mk_unlabeled_cache.sh" > /dev/null; do sleep 30; done
ALL=data/wm811k/cache/wm811k_64pad_all.npz
PAD=data/wm811k/cache/wm811k_64pad.npz
[ -f "$ALL" ] || { echo "미라벨 캐시 없음: $ALL"; exit 1; }

echo "=========== 사전학습: 전체 811K ==========="
$PY src/a11_ssl_pretrain.py --tag ssl_all --cache "$ALL"
echo "=========== 사전학습: 라벨만 172K (대조군) ==========="
$PY src/a11_ssl_pretrain.py --tag ssl_lab --cache "$ALL" --labeled-only

for pair in "e12_ssl:ssl_all" "e12_ssl_lab:ssl_lab"; do
  tag=${pair%%:*}; enc=${pair##*:}
  cfgname=$([ "$tag" = "e12_ssl" ] && echo cls_e12_ssl || echo cls_e12_ssl_lab)
  echo "=========== fine-tune: $tag (인코더 $enc) ==========="
  $PY src/a3_train_wm811k_cls.py --config "configs/$cfgname.yaml" \
      --tag "$tag" --init-encoder "result/ssl/${enc}_encoder.pt"
  $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" \
      --split test --cache "$PAD"
done
echo "=========== E12 DONE ==========="
