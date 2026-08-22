# 봉인 홀드아웃 — 원자료

11차 사이클(2026-08-23). 결론은 [`docs/reports/OOD.md`](../../reports/OOD.md) 의
**봉인 홀드아웃** 절, 판정 전문은
[`candidate/ood_sealed_holdout.md`](../../experiments/candidate/ood_sealed_holdout.md) 에 있다.

## 한 문장

**여덟 사이클의 순위와 이득 크기는 봉인 자료에서도 유지되고,
수준 0.8325 는 여전히 편향된 숫자이며 지금 있는 자료로는 편향을 뺄 수 없다.**

| arm | `test_dev`(편향) | `test_sealed` | `val_unseen` 유병률 맞춤(편향 없음) |
|---|---:|---:|---:|
| **채택 fuse3** | **0.8304** | **0.8384** | **0.9291** |
| die k=7 (진짜 바닥) | 0.7043 | 0.7248 | 0.8205 |
| E0 불량 다이 비율 | 0.3823 | 0.3939 | 0.2570 |

순위 Spearman 1.0000(두 쌍 다), `fuse3 − 바닥` 격차 +0.1260 / +0.1137 / +0.1086.

## evidence

| 파일 | 무엇 |
|---|---|
| `sealed_eval.json` | 세 파티션 × 여섯 arm 전 지표(CI, 운영 지점, 클래스별, 고유값), 재현 확인 |
| `sealed_eval.txt` | 위를 낸 실행의 콘솔 출력 그대로 |
| `holdout_shift.json` | val 이 왜 더 쉬운가 — 유병률만 맞춤 대 클래스 구성까지 맞춤 |
| `unseal_log.jsonl` | **봉인 해제 감사 로그.** 언제 무엇을 왜 열었는지 |
| `sealed_lots.txt` | 봉인된 lot 이름 1,285개 |

## 봉인은 파일이 없어져도 복원된다

레지스트리 `result/ood/holdout/sealed_holdout_v1.npz` 는 `result/` 아래라 git 에 없다.
그래도 **분할은 `SALT="ood_sealed_v1"` 과 `THRESHOLD=2500` 만으로 결정**되고
그 두 상수는 `tests/test_a38_sealed_holdout.py` 로 박혀 있다.
파일이 지워져도 `build_registry()` 가 **같은 lot** 을 다시 뽑는다.
`sealed_lots.txt` 는 그 사실을 나중에 검증할 수 있게 남기는 사본이고,
**`test_sealed_lots_match_the_list_committed_to_git` 가 매번 대조한다** —
상수만 박으면 해시 방식을 바꿔도 통과하기 때문이다.
(실제로 `SALT` 를 바꿔 보면 lot 1,200개 중 288개만 겹친다. 테스트가 판별한다.)

**단 감사 로그는 복원되지 않는다.** `unseal_log.jsonl` 이 지워지면
누가 언제 열었는지는 사라진다. 그래서 사이클마다 여기에 사본을 남긴다.
