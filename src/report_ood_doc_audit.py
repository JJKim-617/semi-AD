"""사이클 종료 조건을 기계적으로 확인한다 (CLAUDE.md 원칙 6).

사이클마다 문서 3종을 채웠는지 눈으로 확인하다 보면 빠뜨린다.
실제로 옆 워크스트림이 cycle_log 를 여덟 사이클 동안 방치한 적이 있다.

확인하는 것:
- 후보 문서마다 **반증 조건**(실행 전)과 **판정**(실행 후)이 둘 다 있는가
- 사이클 로그에 사이클이 빠짐없이 있는가
- SSOT 에 필수 절이 다 있는가
- 기획서의 범위 선언 절이 살아 있는가

`python3 src/report_ood_doc_audit.py` — 프로젝트 루트에서.
"""
import io
import os
import re

print("=== 후보 문서: 반증 조건 + 판정 둘 다 있는가 ===")
cand = sorted(f for f in os.listdir("docs/experiments/candidate") if f.startswith("ood_"))
for f in cand:
    s = io.open(os.path.join("docs/experiments/candidate", f), encoding="utf-8").read()
    has_fals = "반증 조건" in s
    has_verd = "## 판정" in s
    empty = "(아직 없음)" in s
    flag = "OK " if (has_fals and has_verd and not empty) else "**확인필요**"
    print("  %-34s 반증조건 %s  판정 %s  %s"
          % (f, "O" if has_fals else "X", "O" if has_verd and not empty else "X", flag))

print("\n=== 사이클 로그 ===")
s = io.open("docs/experiments/details/ood_cycle_log.md", encoding="utf-8").read()
cyc = re.findall(r"^# (\d+)차 사이클", s, re.M)
print("  기록된 사이클:", cyc)
print("  줄 수:", s.count("\n"))

print("\n=== SSOT ===")
r = io.open("docs/reports/OOD.md", encoding="utf-8").read()
for sec in ["전체 요약", "운영 지점", "클래스별", "통한 것", "기각된 것", "정정 기록",
            "국소화 산출물", "재현 지도", "미완"]:
    print("  %-14s %s" % (sec, "O" if sec in r else "**없음**"))
print("  정정 항목 수:", len(re.findall(r"^\d+\. \*\*", r.split("## 정정 기록")[1]
                                    .split("##")[0], re.M))
      if "## 정정 기록" in r else "?")

print("\n=== 기획서 ===")
pl = io.open("docs/plans/2026-08-22-ood-oneclass-preregistration.md", encoding="utf-8").read()
for sec in ["# 9. 범위 확장", "# 10. \"정상만 사용\""]:
    print("  %-24s %s" % (sec, "O" if sec in pl else "**없음**"))

print("\n=== 결과 디렉토리 ===")
for d in sorted(os.listdir("result/ood")):
    n = len(os.listdir(os.path.join("result/ood", d)))
    print("  %-22s 파일 %d개" % (d, n))
