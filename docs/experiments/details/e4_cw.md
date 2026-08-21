# E4 — 역빈도 class weight

설정 `configs/cls_e4_cw.yaml` · 결과 `result/cls_baseline/e4_cw_*` · seed 0

증강 없이 불균형 처리만의 효과를 분리해서 본다.

## 조건

| 항목 | 값 |
|---|---|
| 인코더 | ResNet-18 (stem 3x3 s1, maxpool 제거), pretrained=False |
| 입력 | one-hot 3채널, 64x64 |
| 증강 | 없음 |
| 불균형 처리 | 역빈도 class weight |
| 최적화 | Adam lr 0.001, batch 256, 15 epoch |
| 소요 | 514.5초 (cuda) |

## 결과

| 분할 | macro-F1 | accuracy |
|---|---:|---:|
| val (체크포인트 선택용) | 0.8790 | — |
| **test (공식)** | **0.6328** | 0.9467 |

test accuracy 는 전부 none 으로 찍는 자명한 분류기(93.34%)와 비교해서 읽어야 한다.

### 클래스별 (test)

| 클래스 | support | recall | F1 |
|---|---:|---:|---:|
| none | 110,701 | 0.974 | 0.976 |
| Center | 832 | 0.446 | 0.523 |
| Donut | 146 | 0.514 | 0.564 |
| Edge-Loc | 2,772 | 0.589 | 0.600 |
| Edge-Ring | 1,126 | 0.665 | 0.744 |
| Loc | 1,973 | 0.544 | 0.459 |
| Random | 257 | 0.572 | 0.664 |
| Scratch | 693 | 0.382 | 0.310 |
| Near-full | 95 | 0.989 | 0.855 |

### 학습 추이

- 최고 val macro-F1 0.8790 (ep 11)
- 최저 val macro-F1 0.2868 (ep 4, accuracy 0.1802)
- 최종 loss 0.0851 (ep1 1.3840)

ep 4 에서 학습이 붕괴했다 회복했다. 역빈도 가중치가 극소수 클래스에 큰 배율을 걸어 gradient 분산이 커진 것으로 보인다.

재현

```bash
./exe/a3_train_cls.sh configs/cls_e4_cw.yaml
./exe/a4_eval_cls.sh result/cls_baseline/e4_cw_best.pt --split test
```
