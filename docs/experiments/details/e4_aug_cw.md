# E4b — 증강 + class weight

설정 `configs/cls_e4_aug_cw.yaml` · 결과 `result/cls_baseline/e4_aug_cw_*` · seed 0

두 기법을 동시 적용. 효과가 더해지는지 겹치는지 확인한다.

## 조건

| 항목 | 값 |
|---|---|
| 인코더 | ResNet-18 (stem 3x3 s1, maxpool 제거), pretrained=False |
| 입력 | one-hot 3채널, 64x64 |
| 증강 | dihedral |
| 불균형 처리 | 역빈도 class weight |
| 최적화 | Adam lr 0.001, batch 256, 15 epoch |
| 소요 | 518.1초 (cuda) |

## 결과

| 분할 | macro-F1 | accuracy |
|---|---:|---:|
| val (체크포인트 선택용) | 0.9076 | — |
| **test (공식)** | **0.6545** | 0.9494 |

test accuracy 는 전부 none 으로 찍는 자명한 분류기(93.34%)와 비교해서 읽어야 한다.

### 클래스별 (test)

| 클래스 | support | recall | F1 |
|---|---:|---:|---:|
| none | 110,701 | 0.976 | 0.978 |
| Center | 832 | 0.523 | 0.482 |
| Donut | 146 | 0.507 | 0.576 |
| Edge-Loc | 2,772 | 0.605 | 0.582 |
| Edge-Ring | 1,126 | 0.733 | 0.777 |
| Loc | 1,973 | 0.486 | 0.440 |
| Random | 257 | 0.747 | 0.722 |
| Scratch | 693 | 0.391 | 0.441 |
| Near-full | 95 | 0.968 | 0.893 |

### 학습 추이

- 최고 val macro-F1 0.9076 (ep 10)
- 최저 val macro-F1 0.1613 (ep 4, accuracy 0.1429)
- 최종 loss 0.1477 (ep1 1.2593)

ep 4 에서 학습이 붕괴했다 회복했다. 역빈도 가중치가 극소수 클래스에 큰 배율을 걸어 gradient 분산이 커진 것으로 보인다.

재현

```bash
./exe/a3_train_cls.sh configs/cls_e4_aug_cw.yaml
./exe/a4_eval_cls.sh result/cls_baseline/e4_aug_cw_best.pt --split test
```
