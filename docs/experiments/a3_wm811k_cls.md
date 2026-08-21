# a3 — WM-811K 9-class 분류

## E1 순수 베이스라인 (2026-08-21)

ResNet-18(stem 3x3 s1 + maxpool 제거), one-hot 3채널, scratch. 증강 없음, 불균형 처리 없음.
15 epoch, batch 256, Adam lr 1e-3, seed 0. GPU 0, 538초.

설정 `configs/cls_e1_baseline.yaml`, 결과 `result/cls_baseline/e1_scratch_*`.

### 결과 — val 과 test 의 격차가 이 실험의 핵심 발견

| 분할 | n | macro-F1 | accuracy |
|---|---:|---:|---:|
| val (Training 내 lot 분할) | 11,132 | **0.8861** | 0.9737 |
| **test (공식 trianTestLabel)** | 118,595 | **0.5339** | 0.8719 |

**macro-F1 이 0.886 에서 0.534 로 무너진다.** val 만 보면 목표(0.85)를 이미 넘긴 것처럼 보이지만
공식 Test 에서는 절반 수준이다.

### 클래스별 비교

| 클래스 | val support | val recall | test support | test recall |
|---|---:|---:|---:|---:|
| none | 7,581 | 0.999 | 110,701 | 0.900 |
| Center | 682 | 0.966 | 832 | 0.535 |
| Donut | 82 | 0.720 | 146 | 0.336 |
| Edge-Loc | 502 | 0.761 | 2,772 | 0.447 |
| Edge-Ring | 1,774 | 0.967 | 1,126 | 0.412 |
| Loc | 282 | 0.894 | 1,973 | 0.571 |
| Random | 128 | 0.930 | 257 | 0.599 |
| Scratch | 92 | 0.772 | 693 | 0.237 |
| Near-full | 9 | 0.778 | 95 | 0.863 |

Edge-Ring 이 0.967 에서 0.412 로 떨어진 것이 특히 크다. Scratch 는 0.237 로 가장 낮다.
Near-full 만 test 에서 오히려 올랐는데 support 가 9장과 95장이라 표본이 작다.

### 해석

원인은 과적합이라기보다 **분할 간 분포 이동**으로 보인다. a2 단계에서 실측했듯
Training 은 none 67.6%, Test 는 none 93.3% 이고 Edge-Ring 은 Training 8,554장 vs Test 1,126장으로 역전돼 있다.
val 은 Training 에서 떼어냈으므로 Training 분포를 그대로 물려받는다.
즉 val 은 낙관적인 추정치이며, 일반화 성능은 test 로만 판단해야 한다.

lot 겹침은 train/val 0개, Training/Test 0개로 확인됐으므로 데이터 누수는 아니다.

### 문헌 대비 위치

공식 split + 불균형 유지 조건의 공개 수치는 macro 0.7 수준이 확인된다
([Defect_KAN](https://github.com/judahobi/Defect_KAN) 공식 분할 macro-F1 0.703, 단 학습 시
공식 Test 행을 배제하지 않은 누수가 있어 낙관 편향).
현재 0.534 는 그보다 낮고, 증강과 불균형 처리를 아직 적용하지 않은 상태다.

98~99% 대 수치는 대부분 balanced 파생 벤치마크이거나 결함 8클래스 조건이라 직접 비교 대상이 아니다.
자세한 것은 [인코더 선택 조사](../research/encoder_selection/README.md) 참조.

### 다음

1. **주 지표를 test macro-F1 로 고정**한다. val 은 체크포인트 선택에만 쓴다.
2. E3 증강(dihedral)과 E4 불균형 처리(class weight)를 적용해 test macro-F1 이 오르는지 본다.
   현재 파이프라인에 `--augment`, `--class-weight` 플래그로 이미 준비돼 있다.
3. Scratch(0.237), Donut(0.336), Edge-Ring(0.412) 이 우선 개선 대상이다.
4. E5 에서 `--pretrained` 로 동일 조건 비교. 17만 장 규모라 이득이 크지 않을 것이라는
   가설을 검증한다.
