# a3 — WM-811K 9-class 분류 실험군

이 문서는 **진입점**이다. 수치의 최신 상태는 [reports/CLASSIFICATION.md](../reports/CLASSIFICATION.md),
개별 실행 상세는 [details/](details/) 에 있다.

주 지표는 **공식 test split 의 macro-F1**. val 은 체크포인트 선택에만 쓴다
(이유는 [specs/splits.md](../specs/splits.md)).

## 실험 목록

| 실험 | 조건 | test macro-F1 | 상세 |
|---|---|---:|---|
| E1 | 순수 베이스라인 | 0.5339 | [e1_scratch](details/e1_scratch.md) |
| E3 | + dihedral 증강 | 0.6350 | [e3_aug](details/e3_aug.md) |
| E4 | + 역빈도 class weight | 0.6328 | [e4_cw](details/e4_cw.md) |
| **E4b** | 증강 + class weight | **0.6545** | [e4_aug_cw](details/e4_aug_cw.md) |
| E5 | focal loss 계열 | 진행 중 | — |

공통 조건: ResNet-18(stem 3x3 s1, maxpool 제거), one-hot 3채널 64x64, Adam lr 1e-3, batch 256, 15 epoch, seed 0.
인코더 선택 근거는 [research/encoder_selection](../research/encoder_selection/README.md).

## 이 실험군에서 확인된 것

1. **val 이 두 번 방향을 잘못 알려줬다.** E3 은 val +0.016 인데 test +0.101, E4 는 val 이
   내려갔는데(-0.007) test 는 +0.099 올랐다. val 은 Training 분포를 물려받아
   일반화 추정치가 아니다.
2. **증강과 class weight 는 효과가 겹친다.** 각각 +0.10 인데 합쳐도 +0.12 다.
   다만 살리는 클래스가 다르다. 증강은 Edge-Ring(0.412->0.930), class weight 는
   Near-full(0.863->0.989).
3. **class weight 가 학습을 불안정하게 만든다.** E4 는 ep4 에 accuracy 0.180,
   E4b 는 0.143 까지 붕괴했다 회복했다.
4. **Loc 은 어떤 기법도 듣지 않는다.** E1 0.571 에서 E4b 0.486 으로 오히려 떨어졌다.

해석과 근거는 [reports/CLASSIFICATION.md](../reports/CLASSIFICATION.md) 에 유지한다.

## 진행 중

- E5 focal loss 계열 3종 (`configs/cls_e5_*.yaml`) — 가설은 [candidate/focal_loss.md](candidate/focal_loss.md)
- 결과 진단 (`research/e4_diagnosis/`)
