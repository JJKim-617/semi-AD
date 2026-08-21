# AD 베이스라인 실험 기획 — WM-811K 웨이퍼맵

- 작성일: 2026-08-21, 브랜치: `exp/ad-baseline`
- 범위: **이상탐지(AD)/분류 파트만 다룬다.** MLLM 설명 코파일럿은 이 문서의 범위 밖이며, 여기서 만든 탐지기와 스코어맵이 향후 MLLM 파트의 입력이 된다는 연결만 전제한다.
- 선행 작업: [a1 전처리](../experiments/a1_wm811k_preprocess.md) 완료. 이 문서는 그 다음 단계(a2~)의 계획이다.

## 1. 데이터 현황 요약

| 항목 | 값 |
|---|---|
| 캐시 | `data/wm811k/cache/wm811k_64.npz` (X: 172,950 x 64 x 64 uint8, 값 {0,1,2}) |
| 로딩 | 1.74s, 메모리 708MB — 전량 인메모리 학습 가능 |
| 클래스 | none 147,431 (85.2%) / 결함 8종 25,519 (Edge-Ring 9,680 ~ Near-full 149) |
| 불균형 | 최다/최소 클래스 비 약 1000:1 |
| 미라벨 | 638,507장, 캐시 미포함 (`--include-unlabeled`로 추출 가능) |

원본 해상도는 26x26(38%), 33x29(21%)가 최빈으로 64x64는 대부분 업샘플링이다. 즉 실질 정보량이 낮은 저해상도 데이터라는 점이 이후 모델 선택의 전제가 된다.

## 2. 문제 정의 — 두 갈래와 우선순위

같은 데이터로 두 가지 문제를 정의할 수 있고, 요구하는 프로토콜과 지표가 다르므로 처음부터 구분한다.

| | (a) 9-class 분류 | (b) one-class 이상탐지 |
|---|---|---|
| 학습 데이터 | 라벨 172,950장 전부 | none만 (Training 내 36,730장) |
| 출력 | 9개 클래스 확률 | 이상 점수 (정상 vs 결함) |
| 지표 | macro-F1, per-class recall | AUROC, AUPRC |
| 비교 가능성 | 표준 벤치마크, 문헌 수치와 직접 비교 | 프로토콜이 문헌마다 달라 비교 어려움 |
| 서사 | "높은 정확도" 목표에 부합 | 3D-AD 연구(training-free AD) 경험과 직결 |

**권장 순서: (a) 먼저, 그 다음 (b).** 근거는 데이터 분포다.

- Donut 555장, Near-full 149장 같은 소수 클래스는 지도 분류로도 빠듯하다. 분류를 먼저 돌려 보면 "라벨을 전부 쓰고도 어느 클래스가 얼마나 어려운지"의 상한선이 나오고, 이것이 (b)의 결과를 해석하는 기준선이 된다.
- (a)의 산출물(학습된 인코더, 정규화된 파이프라인, 검증 프로토콜)은 (b)에서 그대로 재사용된다. 역방향은 성립하지 않는다.
- (b)는 none 147,431장이라는 풍부한 정상 데이터 덕에 자연스러운 프레이밍이지만(1000:1 불균형에서 소수 클래스를 "배우는" 대신 "이상으로 검출"), 절대 성능의 좋고 나쁨을 판단할 문헌 기준이 약하다. (a)의 정상-vs-결함 이진화 성능을 상한 참조로 두면 판단이 가능해진다.

## 3. 검증 프로토콜

수치의 신뢰도를 좌우하는 부분이므로 실험 설계보다 먼저 확정한다.

### 3.1 실측 결과 (2026-08-21, 캐시 기준)

원본 `trianTestLabel`(데이터셋 자체 오타, 그대로 참조) split을 실측했다.

- Training 54,355장 (lot 5,809개) / Test 118,595장 (lot 4,953개)
- **두 split 간 lot 겹침: 0개.** 우려했던 "같은 lot이 train/test에 섞이는 누수"는 원본 split에는 없다.
- 대신 **split 간 분포 이동이 크다**: none 비율이 Training 67.6% vs Test 93.3%이고, Center는 Training 3,462 vs Test 832, Edge-Ring은 8,554 vs 1,126으로 역전돼 있다. Test가 훨씬 불균형하다.

### 3.2 프로토콜 제안

1. **원본 split을 기본으로 유지한다.** 문헌 수치와의 직접 비교가 가능한 유일한 선택지이고, lot 누수도 실측상 없다.
2. **검증셋은 Training 내부에서 lot 단위로 분리한다** (예: lot 기준 80/20). 웨이퍼 단위 랜덤 분할은 같은 lot의 거의 동일한 웨이퍼맵이 train/val에 나뉘어 검증 수치를 낙관적으로 왜곡한다. 누수 위험은 원본 split이 아니라 우리가 새로 만드는 분할에서 생긴다.
3. **lot-wise 5-fold를 병기 측정한다** (여유가 되면). 원본 Test의 분포 이동이 커서 단일 split 수치만으로는 일반화를 말하기 어렵다. 리포트에는 "원본 split 수치(비교용) + lot-wise CV 수치(신뢰도용)"를 나란히 적는다.
4. one-class AD도 동일 원칙: Training의 none으로 학습, 원본 Test 전체(정상+결함)로 AUROC 측정. 임계값 튜닝이 필요하면 검증셋에서만 한다.

split 인덱스는 시드 고정 후 npz로 저장해 모든 실험이 같은 분할을 쓰게 한다.

## 4. 평가 지표

accuracy는 none만 찍어도 85%(원본 Test 기준으로는 93%)가 나오므로 단독으로는 무의미하다.

- **분류 (a)**: macro-F1 (주 지표), per-class recall 전체 표, 소수 클래스(Donut, Near-full) recall을 별도 명시. accuracy는 문헌 비교용으로만 병기.
- **AD (b)**: AUROC (주 지표), AUPRC (결함이 소수인 상황에서 AUROC보다 민감), 참고로 FPR@95TPR.
- 모든 결과는 json(재분석용) + md(요약)를 `result/`에 함께 남긴다 (CLAUDE.md 규칙).

## 5. 실험 설계

베이스라인부터 단계적으로 쌓고, 각 단계가 검증하는 질문을 하나로 좁힌다.

### 5.1 모델 선택의 전제

- **인코더는 scratch small CNN이 1순위.** 64x64 저해상도에 원본이 대부분 26x26대라 실질 정보량이 낮아 ResNet50급은 과잉이다. WM-811K 문헌도 0.15M~수 M 파라미터급 경량 CNN이 SOTA 근처다 (7절 출처 참조).
- **LoRA는 이 단계에 부적합.** LoRA는 "큰 사전학습 모델 + 적은 타깃 데이터" 상황의 도구인데, 여기는 17만 장으로 데이터가 충분하고, 모델이 작아도 되며, ImageNet 표현과 웨이퍼맵의 도메인 갭이 크다. LoRA의 자리는 향후 MLLM 파트다. 단 **사전학습 인코더 자체는 비교군으로 실험할 가치가 있다** (ImageNet ResNet-18 fine-tune, frozen feature + linear).
- **성능을 좌우하는 것은 인코더 크기보다 증강과 불균형 처리라고 본다.** 웨이퍼맵은 회전, 반전 대칭성이 있으므로 90도 단위 회전 + 플립(dihedral)이 안전한 증강이다 (임의 각도 회전은 명목형 셀 값에 보간이 개입하므로 신중히). 불균형 처리는 class weight, oversampling, focal loss를 축으로 비교한다.

### 5.2 입력 표현 후보

| | 표현 | 비고 |
|---|---|---|
| A | raw {0,1,2} 1채널 | 가장 단순, 순서 관계를 암묵 가정 |
| B | one-hot 3채널 | 명목형을 정직하게 표현, 3채널이라 사전학습 모델에도 바로 꽂힘 |
| C | 불량 이진 1채널 + 웨이퍼 마스크 채널 | 결함 패턴과 웨이퍼 형상을 분리 |

### 5.3 단계와 ablation 축

| 단계 | 내용 | 검증하는 질문 |
|---|---|---|
| E0 | 다수결/랜덤 예측, 결함 픽셀 수 기반 규칙 | 지표 하한선. 파이프라인 배선 검증 |
| E1 | small CNN + 표현 A, 증강 없음, 불균형 처리 없음 | 순수 베이스라인 |
| E2 | E1 + 입력 표현 A/B/C 비교 | 표현이 성능에 미치는 영향 |
| E3 | E2 최선 + 증강 유무 (90도 회전, 플립) | 증강 효과 크기 |
| E4 | E3 최선 + 불균형 처리 (class weight vs oversampling vs focal) | 소수 클래스 recall 개선 |
| E5 | E4 조건에서 인코더 교체 (scratch CNN vs ImageNet ResNet-18 fine-tune vs frozen + linear) | "작은 모델로 충분한가" 검증 |
| E6 | one-class AD 베이스라인 (none만 학습) | (b) 프레이밍의 첫 수치 |

E6의 방법 후보는 E5까지의 결과를 보고 확정한다. 후보: E5 인코더의 feature 기반 kNN/Mahalanobis (3D-AD에서 쓴 training-free 계열과 동일 발상), 오토인코더 재구성 오차, DeepSVDD류. 우선순위는 feature 기반 training-free 계열 — 학습된 분류 인코더를 재사용할 수 있고 작성자의 기존 연구 방법론과 이어진다.

### 5.4 코드와 산출물 배치 (CLAUDE.md 규칙 준수)

- `src/` (평평하게): `a2_split_wm811k.py` (split 인덱스 생성, 시드 고정), `a3_train_wm811k_cls.py`, `a4_eval_wm811k_cls.py`, `a5_ad_wm811k.py`, 보조로 `viz_wm811k_samples.py`, `report_wm811k_results.py`
- `configs/`: 실험 조건은 전부 yaml로 (`cls_e1_baseline.yaml`, `cls_e3_aug.yaml`, ...). 하드코딩 금지.
- `exe/`: 실험 1건 = 스크립트 1개 (`a3_train_cls_e1.sh`, ...)
- `result/`: 실험군별 하위 폴더 (`result/cls_baseline/`, `result/ad_baseline/`)에 json + md. 실체는 `/mnt/HYDHC` 심링크이므로 용량 걱정 없이 체크포인트도 여기에.
- `docs/experiments/`: 실험군 요약 md (`a3_wm811k_cls.md` 등), 개별 실행 상세는 `details/`로.

## 6. 성공 기준

문헌 수치 조사 결과 (2026-08 웹 검색):

| 출처 | 조건 | 수치 |
|---|---|---|
| [Defect_KAN (EfficientNet-B0 + SPP + KAN)](https://github.com/judahobi/Defect_KAN) | **원본 split, 불균형 유지** | acc 89.8%, macro recall 0.943 (balanced split에선 acc 93.0%, macro-F1 0.913) |
| [AE 증강 + CNN (arXiv 2411.11029)](https://arxiv.org/abs/2411.11029) | 증강 후 분류 | acc 98.56% |
| [경량 CNN (MobileNetV3 계열, PMC9960339)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9960339/) | 경량화 비교 | acc 98%, F1 89.5% |
| [CBAM 경량 CNN (Frontiers 2026)](https://www.frontiersin.org/journals/electronics/articles/10.3389/felec.2026.1750707/full) | **balanced 파생 벤치마크** | acc 99.88%, 파라미터 0.15M |

주의할 점: 문헌의 고정확도(98~99%대)는 상당수가 balanced 파생 벤치마크나 자체 리샘플링 위에서 나온 수치라 원본 split 수치와 직접 비교하면 안 된다. 원본 split + 불균형 유지 조건의 공개 수치는 macro 기준 0.9 초반대가 확인되는 수준이다. one-class AD 쪽은 프로토콜이 논문마다 달라 표준화된 AUROC 기준을 찾지 못했다.

이를 근거로 다음을 목표로 제안한다 (달성치가 아니라 제안이며, E1 결과를 보고 조정한다):

- **분류**: 원본 split에서 E4까지 macro-F1 0.80 이상을 1차 목표, 0.85 이상이면 문헌 대비 손색없는 수준으로 판단. 소수 클래스(Donut, Near-full) recall 0.7 이상을 별도 조건으로.
- **AD**: 절대 기준 대신 이중 기준 — (i) E5 분류기의 정상-vs-결함 이진화 AUROC를 상한 참조로 두고, (ii) AUROC 0.90 이상을 1차 기준으로 제안.

## 7. 리스크와 함정

| 리스크 | 내용 | 대응 |
|---|---|---|
| 검증 분할 누수 | 원본 split은 lot 겹침 0으로 확인됐지만, 우리가 만드는 val 분할, CV에서 웨이퍼 단위 랜덤 분할을 쓰면 lot 누수가 생긴다 | 자체 분할은 전부 lot 단위 (3절) |
| Test 분포 이동 | 원본 Test는 none 93.3%로 Training과 분포가 크게 다르다. 단일 수치 과신 금지 | lot-wise CV 병기, per-class 표 필수 |
| oversampling 왜곡 | Near-full 149장을 수십 배 복제하면 사실상 같은 그림을 암기한다. val/test에 절대 리샘플링 금지 | 증강과 결합해서만 oversample, 학습셋에만 적용 |
| 업샘플링 아티팩트 | 26x26 → 64x64 nearest 확대가 패턴을 계단화한다 | `mode: pad` 캐시(a1에 옵션 존재)와의 비교를 E2 확장으로 검토 |
| 미라벨 638K 활용 | 자기지도 사전학습 유혹이 있으나 범위가 커진다 | 이번 2일 범위에서 제외, 향후 과제로만 기록 |
| 재현성 | 시드, split 인덱스, config가 흩어지면 수치를 재현 못 한다 | split 인덱스 파일 고정, 실험 1건 = exe 스크립트 1개 + config 1개, 결과 json에 config 경로와 git hash 기록 |

## 8. 작업 순서 (2일)

| 순서 | 작업 | 산출물 |
|---|---|---|
| 1일차 오전 | `a2_split_wm811k.py`: lot-wise val 분할 + 인덱스 저장, E0 규칙 베이스라인 | `data/wm811k/cache/splits_v1.npz`, `result/cls_baseline/e0_*.{json,md}` |
| 1일차 오후 | `a3_train_wm811k_cls.py` + config/exe 골격, E1 학습, `a4_eval_wm811k_cls.py`로 macro-F1, per-class recall 산출 | `result/cls_baseline/e1_*.{json,md}` |
| 2일차 오전 | E2(표현 A/B/C), E3(증강) — 인메모리 데이터라 small CNN 기준 회전이 빠르므로 순차 실행 가능 | `result/cls_baseline/e2_*, e3_*` |
| 2일차 오후 | E4(불균형 처리) 중 1~2개 조건, 실험군 요약 작성 | `result/cls_baseline/e4_*`, `docs/experiments/a3_wm811k_cls.md` |
| 이후 | E5(인코더 비교), E6(one-class AD), lot-wise CV | `result/ad_baseline/`, `docs/experiments/` 갱신 |

2일 안에 E4까지가 현실적 목표이고, E5~E6은 다음 사이클로 넘겨도 문서와 코드 구조상 이어받는 데 문제가 없도록 위 배치 규칙을 지킨다.

## 9. 범위 밖 (향후 과제)

- 미라벨 638K를 이용한 자기지도/준지도 학습
- MixedWM38(복합 결함) 확장
- MLLM 설명 코파일럿: 여기서 만든 탐지기의 스코어맵과 클래스 출력을 입력으로 받는 후속 파트. LoRA 등 PEFT는 그 단계에서 검토한다.
