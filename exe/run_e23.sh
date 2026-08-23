#!/usr/bin/env bash
# E23 — 자유 각도 회전. dihedral(유한군 D4)을 연속 회전군으로 넓힌다.
# 반증 조건은 실행 전에 docs/experiments/candidate/rotation_symmetry.md 에 박았다.
#
# E22 큐가 끝나기를 기다린 뒤 시작한다. `pkill -f` 는 금지이고 `pgrep -f` 도
# 자기 명령줄을 잡을 수 있어 쓰지 않는다 — E22 스크립트가 로그에 남기는
# 완료 표시만 본다.
#
# 사용: ./exe/run_e23.sh              (seed 0 1 2, 두 팔 전부)
#       ./exe/run_e23.sh --now 0      (기다리지 않고 seed 0 만)
source "$(dirname "$0")/_common.sh"
CACHE=data/wm811k/cache/wm811k_64pad.npz

WAIT=1
if [ "${1:-}" = "--now" ]; then WAIT=0; shift; fi
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)

if [ "$WAIT" = "1" ]; then
  echo "E22 큐가 끝나기를 기다린다..."
  until grep -q "E22 DONE" /tmp/e22.log 2>/dev/null; do sleep 60; done
  echo "E22 끝남 ($(date +%H:%M:%S))"
  # 큐를 연달아 6시간 물고 있지 않도록 **일부러 틈을 둔다.**
  # GPU 0 은 OOD 에이전트와 공유하는 하나뿐인 슬롯이다.
  echo "다른 작업을 위해 10분 비워 둔다."
  sleep 600
fi

# GPU 0 이 실제로 비었는지 확인한다. 남이 올라와 있으면 겹쳐서 둘 다 OOM 으로 죽는다.
for _ in $(seq 1 120); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$used" -lt 2000 ] && break
  echo "GPU0 에 $used MiB 가 올라와 있다. 60초 뒤 다시 본다."
  sleep 60
done

for seed in "${SEEDS[@]}"; do
  # 팔 A = 회전 단독, 팔 B = 회전 + 이동. 두 팔을 seed 마다 짝지어 돌려
  # 중간에 끊겨도 고르게 남게 한다.
  for arm in "rot:rotate" "rottr:rotate translate"; do
    name="${arm%%:*}"; recipe="${arm#*:}"
    tag="e23_${name}_s${seed}"
    if [ -f "result/cls_baseline/${tag}_best_test.json" ]; then
      echo "이미 있음: $tag"; continue
    fi
    echo "=========== $tag  (extra: $recipe)  ($(date +%H:%M:%S)) ==========="
    $PY src/a3_train_wm811k_cls.py --config configs/cls_e17.yaml \
        --tag "$tag" --seed "$seed" --extra-augment $recipe
    $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" \
        --split test --cache "$CACHE"
  done
done
echo "=========== E23 DONE: ${SEEDS[*]}  ($(date +%H:%M:%S)) ==========="
