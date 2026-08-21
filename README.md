# semi-ad — 반도체 이상탐지 + MLLM 설명 코파일럿

## 목표
정상만 학습해 결함을 OOD로 검출(2D 웨이퍼맵 + 3D 점군)하고,
agentic MLLM이 결함을 자연어로 설명하고 근본원인까지 제시하는 코파일럿.
- 연계: 3D-AD 학부연구(MARS) + HEIR agentic 파이프라인을 반도체 도메인으로 확장.

## 데이터셋 후보
- 2D 웨이퍼맵: WM-811K(811,457장, 9클래스, None ~85%), MixedWM38
- 2D 시각: MVTec-AD(transistor), VisA(PCB)
- 3D: Real3D-AD, Anomaly-ShapeNet
- 공정 센서: SECOM

## 참고 논문
AnomalyGPT(AAAI 2024) · WamGLM · WaferSAGE · SemiFA · LDU-Bench · EIAD · AnomalyR1

## 구조
`CLAUDE.md`의 배치 규칙을 따른다. (src/ exe/ configs/ result/ docs/ tests/ data/)
