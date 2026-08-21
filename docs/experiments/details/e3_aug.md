# E3 — dihedral 증강

설정 `configs/cls_e3_aug.yaml` · 결과 `result/cls_baseline/e3_aug_*` · seed 0

90도 배수 회전과 좌우 반전. 웨이퍼의 회전 대칭성을 이용한다. 격자 인덱스 재배열이라 명목형 셀 값이 보존된다.

## 조건

| 항목 | 값 |
|---|---|
| 인코더 | ResNet-18 (stem 3x3 s1, maxpool 제거), pretrained=False |
| 입력 | one-hot 3채널, 64x64 |
| 증강 | dihedral |
| 불균형 처리 | 없음 |
| 최적화 | Adam lr 0.001, batch 256, 15 epoch |
| 소요 | 515.3초 (cuda) |

## 결과

| 분할 | macro-F1 | accuracy |
|---|---:|---:|
| val (체크포인트 선택용) | 0.9021 | — |
| **test (공식)** | **0.6350** | 0.9392 |

test accuracy 는 전부 none 으로 찍는 자명한 분류기(93.34%)와 비교해서 읽어야 한다.

### 클래스별 (test)

| 클래스 | support | recall | F1 |
|---|---:|---:|---:|
| none | 110,701 | 0.962 | 0.974 |
| Center | 832 | 0.612 | 0.565 |
| Donut | 146 | 0.404 | 0.494 |
| Edge-Loc | 2,772 | 0.587 | 0.584 |
| Edge-Ring | 1,126 | 0.930 | 0.394 |
| Loc | 1,973 | 0.510 | 0.578 |
| Random | 257 | 0.844 | 0.728 |
| Scratch | 693 | 0.439 | 0.521 |
| Near-full | 95 | 0.789 | 0.877 |

### 학습 추이

- 최고 val macro-F1 0.9021 (ep 9)
- 최저 val macro-F1 0.5581 (ep 1, accuracy 0.9223)
- 최종 loss 0.0406 (ep1 0.3147)

재현

```bash
./exe/a3_train_cls.sh configs/cls_e3_aug.yaml
./exe/a4_eval_cls.sh result/cls_baseline/e3_aug_best.pt --split test
```
