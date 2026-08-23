#!/usr/bin/env bash
# E24 — 제로 패딩이 절대 위치의 출처인가. **기전의 개입 시험이다.**
# 이번 사이클이 낸 기전("위치 민감도가 누출을 몬다")은 모델 사이의 상관이라
# 개입이 아니었다. 여기서 원인 후보를 직접 끈다.
# 반증 조건은 실행 전에 docs/experiments/candidate/padding_position_leak.md 에 박았다.
#
# 팔 A = reflect 패딩만(증강 없음) — 핵심. 팔 B = reflect + translate.
# seed 를 바깥 루프에 둬서 중간에 끊겨도 두 팔이 고르게 남는다.
#
# 사용: ./exe/run_e24.sh              (E22 를 기다린 뒤 seed 0 1 2)
#       ./exe/run_e24.sh --now 0      (기다리지 않고 seed 0)
source "$(dirname "$0")/_common.sh"
CACHE=data/wm811k/cache/wm811k_64pad.npz

WAIT=1
if [ "${1:-}" = "--now" ]; then WAIT=0; shift; fi
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)

if [ "$WAIT" = "1" ]; then
  echo "E22 큐가 끝나기를 기다린다..."
  # `pkill -f` 는 금지이고 `pgrep -f` 도 자기 명령줄을 잡을 수 있어 쓰지 않는다.
  until grep -q "E22 DONE" /tmp/e22.log 2>/dev/null; do sleep 60; done
  echo "E22 끝남 ($(date +%H:%M:%S)). 다른 작업을 위해 10분 비워 둔다."
  sleep 600
fi

# GPU 0 은 OOD 에이전트와 공유하는 하나뿐인 슬롯이다. 남이 올라와 있으면 기다린다.
for _ in $(seq 1 120); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$used" -lt 3000 ] && break
  echo "GPU0 에 $used MiB. 60초 뒤 다시 본다."
  sleep 60
done

# **팔을 바깥 루프에 둔다.** 팔 A(reflect 단독)가 이 실험의 핵심이고
# 사전 등록이 3 seed 를 요구하므로, 시간이 모자라면 팔 B 를 통째로 버리는 편이
# 두 팔을 2 seed 씩 남기는 것보다 낫다 — 이 프로젝트는 n=2 로 판정했다가 세 번 뒤집혔다.
for arm in "pad:" "padtr:translate"; do
  name="${arm%%:*}"; aug="${arm#*:}"
  if [ "$name" = "padtr" ]; then
    # 한 큐가 4시간을 넘지 않게 팔 사이에 틈을 둔다.
    echo "팔 A 끝. 다른 작업을 위해 10분 비워 둔다. ($(date +%H:%M:%S))"
    sleep 600
  fi
  for seed in "${SEEDS[@]}"; do
    tag="e24_${name}_s${seed}"
    if [ -f "result/cls_baseline/${tag}_best_test.json" ]; then
      echo "이미 있음: $tag"; continue
    fi
    echo "=========== $tag  (reflect, extra: '${aug:-없음}')  ($(date +%H:%M:%S)) ==========="
    if [ -z "$aug" ]; then
      $PY src/a3_train_wm811k_cls.py --config configs/cls_e17.yaml \
          --tag "$tag" --seed "$seed" --padding-mode reflect
    else
      $PY src/a3_train_wm811k_cls.py --config configs/cls_e17.yaml \
          --tag "$tag" --seed "$seed" --padding-mode reflect --extra-augment "$aug"
    fi
    # **평가에도 반드시 같은 padding-mode 를 준다.** 빼면 zeros 모델로 실려 조용히 틀린다.
    $PY src/a4_eval_wm811k_cls.py --ckpt "result/cls_baseline/${tag}_best.pt" \
        --split test --cache "$CACHE" --padding-mode reflect
  done
done
echo "=========== E24 DONE: ${SEEDS[*]}  ($(date +%H:%M:%S)) ==========="
