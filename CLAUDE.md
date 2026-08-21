# semi-ad — 작업 규칙 (Claude 활용 + 실험 로그 축적)

반도체 이상탐지(2D 웨이퍼맵 + 3D 점군) + agentic MLLM 설명 코파일럿.
구조는 `JJ/part/MARS/learned_reg`의 세팅을 이식했다. **아래 배치 규칙을 반드시 지킬 것.**

## 루트는 얇게
루트에는 `CLAUDE.md`, `README.md`, `.gitignore`만 둔다. 산출물은 전부 하위 디렉토리로.

## 배치 규칙
| 위치 | 넣는 것 | 규칙 |
|---|---|---|
| `src/` | 파이썬 코드 | **평평하게(중첩 금지)**. 이름 규칙으로 분류한다. |
| `exe/` | `*.sh` 실행 스크립트 | 실험 1건 = 스크립트 1개. 재현 가능한 실행 단위. |
| `configs/` | `*.yaml` | 실험 조건은 코드가 아니라 여기에. 하드코딩 금지. |
| `result/` | 결과 | **심링크** → `/mnt/HYDHC/ryukimlee/etc/outputs`. `*.json`(지표) + `*.md`(요약)를 같이 남긴다. 실험군별 하위 폴더. |
| `data/` | 데이터셋 | **심링크** → `/mnt/HYDHC/ryukimlee/etc/data`. git에 올리지 않는다. |
| `tests/` | pytest | 회귀 방지. |
| `docs/` | 문서 | 아래 참조. |

## 저장소 위치 (중요)
`/mnt/sdf`는 여유가 적다(96% 사용). **semi-ad의 대용량 산출물은 전부 `/mnt/HYDHC`에 둔다.**
- `data/`   → `/mnt/HYDHC/ryukimlee/etc/data`
- `result/` → `/mnt/HYDHC/ryukimlee/etc/outputs`
`/mnt/sdf` 쪽에는 코드·설정·문서만 남긴다. 새 대용량 경로가 필요하면 `/mnt/HYDHC/ryukimlee/etc/` 아래에 만들고 심링크로 연결한다.

## 실행 환경 — 반드시 지킬 것

### GPU 는 0번만 사용한다
이 서버의 GPU 는 공용이다. **다른 번호의 GPU 를 절대 점유하지 않는다.**
모든 실행은 `CUDA_VISIBLE_DEVICES=0` 을 고정한 상태로 한다.

- 파이썬 진입점은 **torch import 보다 먼저** `_runtime.setup()` 을 호출한다.
  ```python
  import sys; sys.path.insert(0, "src")
  from _runtime import setup
  setup("train")          # GPU 0 고정 + 프로세스 이름 변경
  import torch            # 반드시 setup() 이후
  ```
  `setup()` 은 `CUDA_VISIBLE_DEVICES` 가 0 이 아닌 값으로 이미 설정돼 있으면 예외를 던진다.
- 쉘 스크립트는 첫 줄에서 `source exe/_common.sh` 로 같은 설정을 받는다.
- `torch.device("cuda")`, `cuda:0` 만 쓴다. `cuda:1` 같은 명시적 인덱스를 코드에 적지 않는다.

### 프로세스 이름을 중립적으로 유지한다
`nvitop`, `ps` 의 COMMAND 에 프로젝트 경로나 이름이 노출되지 않게 한다.
`_runtime.setup(name)` 이 프로세스 표시 이름을 `wafer-<name>` 으로 바꾼다
(예: `wafer-train`, `wafer-eval`). 의존성은 `setproctitle` 이며 없으면 조용히 무시된다.

- 스크립트를 절대 경로로 실행하지 않는다. `exe/_common.sh` 가 프로젝트 루트로 `cd` 하므로
  `python3 src/a3_....py` 처럼 상대 경로로 호출한다.
- `setup()` 에 넘기는 이름도 중립적으로 짓는다. 프로젝트명이나 데이터셋 코드명을 쓰지 않는다.

## src/ 파일명 규칙
- `a<N>_<name>.py` — 파이프라인 단계 (예: `a1_load_wm811k.py`, `a5_score.py`)
- `diag_<name>.py` — 진단/탐색 실험
- `bench_<name>.py` — 성능·지연 측정
- `report_<name>.py` — 결과 표/문서 생성
- `viz_<name>.py` — 시각화
- `_<name>.py` — 진입점이 아닌 공용 헬퍼 (`_runtime.py`). 밑줄로 시작해 구분한다.

## docs/ 계층
- `reports/` — **대표 종합 리포트(SSOT)**. `LATENCY.md`, `EVALUATION.md`, `SOTA.md`처럼 주제별 단일 문서.
  여러 실험을 가로질러 종합한 **현재 결론**을 담으며, 새 파일을 늘리지 말고 **같은 파일을 갱신**한다.
  "지금 이 프로젝트의 latency는?"에 항상 한 곳에서 답할 수 있게 유지한다.
- `specs/` 사양 · `plans/` 계획 · `adr/` 설계 결정 기록
- `experiments/` — **요약과 상세를 분리**
  - 최상위: 실험군 요약 md (먼저 읽는 곳)
  - `candidate/`: 가설·후보 아이디어
  - `details/`: 개별 실행 상세 로그 (많아져도 요약이 진입점)
- `research/<topic>/` — 주제별 조사. 각 토픽은 **결론 md 1개 + `evidence/`(원자료: json/jsonl/png/py)**.
  주장과 근거를 반드시 짝지어 둔다.

## 원칙
1. **요약 먼저, 상세는 아래로** — 컨텍스트를 아끼기 위해 진입점은 항상 요약 md.
2. **주장에는 근거를** — 수치를 쓰면 `evidence/`에 원자료를 남긴다.
3. **결과는 이중 포맷** — json(재분석용) + md(읽기용).
4. **문서 성격을 섞지 않는다** — `result/`는 실행 산출물(자동 생성), `docs/experiments/details/`는 시점 기록(불변),
   `docs/reports/`는 큐레이션된 종합 결론(계속 갱신).
5. **빈 디렉토리도 유지** — `.gitkeep`으로 자리를 고정해 산출물이 흩어지지 않게 한다.
