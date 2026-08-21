# 인코더 선택 조사 — WM-811K 분류

조사일 2026-08-21. 목적은 새 아키텍처 설계가 아니라 검증된 인코더를 골라 바로 적용하는 것.

## 결론

**ResNet-18, stem 수정(3x3 stride 1 + maxpool 제거), 3채널 one-hot 입력, 사전학습 여부는 플래그로 두고 둘 다 측정한다.**

근거는 아래 세 가지다.

### 1. 이 데이터셋에서 백본 선택은 성능을 거의 가르지 않는다

동일 조건에서 7개 백본을 비교한 연구에서 정확도가 0.979~0.980으로 사실상 동일했다.
파라미터는 0.4M(ShuffleNetV2 0.5x)에서 20.3M(EfficientNetV2)까지 50배 차이인데도 그렇다.

| 백본 | 파라미터 | 정확도 | F1 |
|---|---|---|---|
| ResNet18 | 11.2M | 0.980 | 0.895 |
| EfficientNetV2 | 20.3M | 0.979 | 0.885 |
| ShuffleNetV2 | 1.3M | 0.980 | 0.892 |
| ShuffleNetV2 0.5x | 0.4M | 0.979 | 0.884 |
| MobileNetV2 | 2.2M | 0.979 | 0.886 |
| MobileNetV3 | 1.5M | 0.980 | 0.895 |
| CNN-WDI | 2.7M | 0.979 | 0.895 |

출처: [Sensors 2023, PMC9960339](https://pmc.ncbi.nlm.nih.gov/articles/PMC9960339/). 단 이 수치는 결함 8클래스를 각 10,000장으로 증강한 조건이다.

→ 백본 고민에 시간을 쓸 이유가 없다. 표준적이고 수정이 쉬운 ResNet-18을 택한다.

### 2. 17만 장 규모에서 ImageNet 사전학습의 최종 성능 이득은 사라진다

- Transfusion(NeurIPS 2019): 대규모 의료영상 두 과제에서 전이 이득이 미미. Retina(약 25만 장)에서 ResNet-50 random 96.4% vs transfer 96.7%. [출처](https://arxiv.org/abs/1902.07208)
- Matsoukas et al.(CVPR 2022): 전이 이득은 데이터가 작을수록 커지며, CheXpert(224,316장) 같은 대규모에서 ResNet 계열의 이득은 제한적. 반대로 APTOS2019(3,662장)에서는 뚜렷. [출처](https://arxiv.org/abs/2203.01825)
- Rethinking ImageNet Pre-training: 사전학습은 초기 수렴을 가속할 뿐 최종 정확도를 반드시 높이지 않는다. [출처](https://arxiv.org/abs/1811.08883)
- 웨이퍼맵 실측 사례: 동일 조건에서 MobileNetV2 fine-tune이 scratch CNN보다 크게 나빴다.
  scratch CNN(0.9M) macro F1 **83.14** vs MobileNetV2 fine-tuned **59.46**. [출처](https://github.com/lorenzolecci/deep-learning-wafer-defects)

→ 사전학습을 쓰더라도 **반드시 동일 조건 scratch를 함께 돌려 비교**한다. 남는 이득은 최종 성능이 아니라 수렴 속도일 가능성이 높다.

### 3. 입력 표현과 stem 처리가 백본보다 중요하다

**3채널 one-hot** (배경 / 정상 / 불량 각 한 채널). 웨이퍼맵 선행 연구가 정확히 같은 구조를 채택했고,
die 단위 이산성을 보존하려 스무딩과 강도 정규화를 적용하지 않았다.
[출처](https://www.frontiersin.org/journals/electronics/articles/10.3389/felec.2026.1750707/full)

- 정수 단일 채널(0/1/2를 밝기로)은 명목형에 인위적 순서와 거리를 주입한다. 실증적으로 아티팩트가 확인된다. [출처](https://arxiv.org/pdf/2212.10446)
- gray→RGB 값 복제는 정보를 추가하지 않고 순서 아티팩트만 복사한다.
- ImageNet mean/std 정규화는 적용하지 않는다. 입력이 이미 자연영상 통계와 무관하다.
- 리사이즈는 nearest neighbor 고정. bilinear/bicubic은 존재하지 않는 중간 범주를 만든다.

**stem 수정**은 원 ResNet 논문까지 거슬러 올라가는 표준이다. He et al.이 CIFAR용 ResNet의 첫 층을 3x3으로 정의했고,
SimCLR도 CIFAR 실험에서 7x7 stride 2를 3x3 stride 1로 바꾸고 max pooling을 제거한다고 명시한다.
[ResNet](https://ar5iv.labs.arxiv.org/html/1512.03385) · [SimCLR](https://ar5iv.labs.arxiv.org/html/2002.05709) · [Lightning CIFAR10 레시피](https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html)

수정하지 않으면 64x64 입력이 stem에서 16x16으로 줄고 layer2~4의 stride 2를 거쳐 **최종 feature map이 2x2**가 된다.
GAP 직전 공간 해상도가 사실상 소멸한다.

주의: timm의 `stem_type='deep'`은 저해상도 대책이 아니다. 3x3 conv 3개를 쓰지만 총 stride 4는 그대로다.
[timm resnet.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py)

## frozen 인코더를 쓸 경우의 예외

linear probe나 frozen feature 기반 AD(E6)에서는 stem을 건드리지 말고 **nearest로 128 또는 224까지 업샘플**한 뒤
원본 stem을 쓰는 쪽이 낫다. 네이티브 64x64인 EuroSAT에서 ImageNet ResNet-50 임베딩 + KNN 평가 시
64x64 그대로 82.09% vs 224 리사이즈 91.17%로 약 9pp 차이가 났다. [출처](https://arxiv.org/abs/2305.13456)

사전학습 해상도와 테스트 해상도를 맞추는 것이 이유이며, full fine-tuning에서는 네트워크가 적응할 기회가 있어 격차가 줄어든다.

## 공개 성능 수치를 목표로 삼으면 안 되는 이유

같은 WM-811K인데 보고 성능이 79%에서 99.88%까지 갈린다. 원인은 아키텍처가 아니라 실험 조건이며 축이 세 개다.

1. **'none' 포함 여부** — 결함 전용 8클래스(25,519장)로 하면 'none'이 라벨의 85%라 정확도가 통째로 달라진다.
2. **테스트셋 균형화** — 클래스당 N개로 balanced 재구성하면 자연 분포 성능과 비교 불가.
   한 레포가 같은 모델로 acc 95.68%, balanced acc 87.32%, macro F1 83.14%를 동시에 보고한 것이 이 격차를 보여준다.
3. **공식 split 사용 여부** — 다수 레포가 자체 랜덤 분할을 쓴다.

→ 우리 기준은 **공식 `trianTestLabel` 분할 + 9클래스 + 자연 분포 + macro F1**로 고정한다.

## 참고할 만한 공개 구현

| 레포 | 쓸모 | 주의 |
|---|---|---|
| [WaPIRL](https://github.com/hgkahng/WaPIRL) | 전처리가 가장 정확하다. 전부 INTER_NEAREST 리사이즈, `decouple_mask`가 [결함 마스크, 다이 존재 마스크] 2채널 분리. IEEE TSM 게재 논문 원저자 구현 | pytorch-lightning 0.8.5 고정, 포팅 필요 |
| [xalzh/WM-811k](https://github.com/xalzh/WM-811k) | ImageNet 사전학습 백본 8종이 1채널로 교체되어 있어 백본 비교를 바로 돌릴 수 있다. 공식 split 사용 | `anti_aliasing=True` 리사이즈가 이산값을 뭉갠다. 테스트셋이 클래스당 1,000장 balanced |

WaPIRL의 2채널 분리는 우리가 택한 3채널 one-hot과 사실상 같은 발상이다(배경 채널을 명시하느냐 차이).

## 이 조사가 실험 계획에 미치는 영향

`docs/plans/ad_baseline_plan.md`의 E1~E5를 다음과 같이 조정한다.

- E1 베이스라인을 손수 만든 small CNN이 아니라 **ResNet-18(stem 수정, scratch)**로 시작한다. 검증된 표준이고 사전학습 토글이 플래그 하나다.
- E2 입력 표현 비교에서 **B(one-hot 3채널)를 기본 후보로 승격**한다. 문헌 선례가 있다.
- E5 인코더 비교를 **scratch vs ImageNet 사전학습(동일 ResNet-18)**의 통제 비교로 좁힌다. 백본 종류를 늘리는 실험은 근거상 가치가 낮다.

## 확인하지 못한 것

- 64x64 **명목형** 데이터에서 stem 교체 + 사전학습 유지 vs 업샘플 + 원본 stem을 직접 비교한 통제 실험은 없다. 위 권고는 인접 도메인(EuroSAT, CIFAR, MedMNIST) 결과의 외삽이다.
- one-hot 3채널을 ImageNet 사전학습 인코더에 넣었을 때의 정량 비교 자료 없음. 웨이퍼맵 one-hot 선례는 scratch 학습이었다.
- WM-811K one-class AD의 표준화된 AUROC 기준을 찾지 못했다.
