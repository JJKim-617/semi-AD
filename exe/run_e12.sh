#!/usr/bin/env bash
# E12 — 미라벨 638,507장이 도움이 되는가. 대조군은 라벨 172K 로만 같은 사전학습.
#
# 2026-08-22 개정. 원래 조건당 1 seed 였는데, E10 에서 run-to-run 범위가 0.050 으로
# 측정돼 단일 실행 비교로는 부호조차 확정할 수 없다는 것이 확인됐다. 조건당 2 seed 로
# 늘리고 평균과 범위를 함께 본다. 2 seed 로도 부족하지만 예산 안에서의 타협이다.
source "$(dirname "$0")/_common.sh"
while pgrep -f "run_e11.sh|run_e15.sh|mk_unlabeled_cache.sh" > /dev/null; do sleep 30; done
ALL=data/wm811k/cache/wm811k_64pad_all.npz
PAD=data/wm811k/cache/wm811k_64pad.npz
[ -f "$ALL" ] || { echo "미라벨 캐시 없음: $ALL"; exit 1; }

echo "=========== 사전학습: 전체 811K ==========="
$PY src/a11_ssl_pretrain.py --tag ssl_all --cache "$ALL"
echo "=========== 사전학습: 라벨만 172K (대조군) ==========="
$PY src/a11_ssl_pretrain.py --tag ssl_lab --cache "$ALL" --labeled-only

for pair in "e12_ssl:ssl_all:cls_e12_ssl" "e12_ssl_lab:ssl_lab:cls_e12_ssl_lab"; do
  base=${pair%%:*}; rest=${pair#*:}; enc=${rest%%:*}; cfgname=${rest##*:}
  for seed in 0 1; do
    tag="${base}_s${seed}"
    echo "=========== fine-tune: $tag (인코더 $enc, seed $seed) ==========="
    $PY src/a3_train_wm811k_cls.py --config "configs/$cfgname.yaml" \
        --tag "$tag" --seed "$seed" --init-encoder "result/ssl/${enc}_encoder.pt"
    $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" \
        --split test --cache "$PAD"
  done
done

# 기준선도 같은 2 seed 로 잡혀 있어야 비교가 성립한다. e8_pad(seed 0), e10_pad_s1(seed 1)이
# 정확히 같은 설정의 seed 0,1 이므로 그대로 대조군으로 쓴다. 재학습하지 않는다.
echo "=========== 기준선(재학습 없음): e8_pad 0.7485, e10_pad_s1 0.6991, 평균 0.7238 ==========="

echo "=========== 앙상블: 조건별 2 seed ==========="
$PY src/a8_ensemble.py --tags e12_ssl_s0 e12_ssl_s1 --cache "$PAD"
$PY src/a8_ensemble.py --tags e12_ssl_lab_s0 e12_ssl_lab_s1 --cache "$PAD"
echo "=========== E12 DONE ==========="
