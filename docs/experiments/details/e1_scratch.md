# E1 — 순수 베이스라인

설정 `configs/cls_e1_baseline.yaml` · 결과 `result/cls_baseline/e1_scratch_*` · seed 0

증강 없음, 불균형 처리 없음. 이후 모든 실험의 기준선.

## 조건

| 항목 | 값 |
|---|---|
| 인코더 | ResNet-18 (stem 3x3 s1, maxpool 제거), pretrained=False |
| 입력 | one-hot 3채널, 64x64 |
| 증강 | 없음 |
| 불균형 처리 | 없음 |
| 최적화 | Adam lr 0.001, batch 256, 15 epoch |
| 소요 | 538.2초 (cuda) |

## 결과

| 분할 | macro-F1 | accuracy |
|---|---:|---:|
| val (체크포인트 선택용) | 0.8861 | — |
| **test (공식)** | **0.5339** | 0.8719 |

test accuracy 는 전부 none 으로 찍는 자명한 분류기(93.34%)와 비교해서 읽어야 한다.

### 클래스별 (test)

| 클래스 | support | recall | F1 |
|---|---:|---:|---:|
| none | 110,701 | 0.900 | 0.935 |
| Center | 832 | 0.535 | 0.451 |
| Donut | 146 | 0.336 | 0.438 |
| Edge-Loc | 2,772 | 0.447 | 0.438 |
| Edge-Ring | 1,126 | 0.412 | 0.572 |
| Loc | 1,973 | 0.571 | 0.184 |
| Random | 257 | 0.599 | 0.670 |
| Scratch | 693 | 0.237 | 0.203 |
| Near-full | 95 | 0.863 | 0.916 |

### 학습 추이

- 최고 val macro-F1 0.8861 (ep 11)
- 최저 val macro-F1 0.5991 (ep 1, accuracy 0.9499)
- 최종 loss 0.0091 (ep1 0.2492)

재현

```bash
./exe/a3_train_cls.sh configs/cls_e1_baseline.yaml
./exe/a4_eval_cls.sh result/cls_baseline/e1_scratch_best.pt --split test
```
