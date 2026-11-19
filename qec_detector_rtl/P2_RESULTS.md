# P2 결과: 고정 RTL과 기본 testbench

- 작성일: 2026-11-18, Asia/Seoul
- 상태: **로컬 구현·기본 검증·서버 전송 묶음 준비 완료. 수업 서버의 VCS·Verdi 검증은 대기 중이다.**
- 입력: P1 frozen run `d3_r3_z_v1_20261118_001`
- 정상 로컬 run: `artifacts/qec_detector_rtl/p2_local_20261118_005/`
- 로컬 도구: Icarus Verilog 12.0 (devel), Python 3.10.20

## 1. 구현

[RTL](rtl/measurement_to_detection.v)은 33비트 기록 register, 수락한 측정 수,
다음 detector ID와 한 칸의 출력 register를 사용한다. P1의 고정 참조 표에 따라
D0–D23을 계산한다. 기준 패리티는 이 회로에서 모두 0이다.

clock의 posedge에서 입력을 수락하고, 새로 저장한 기록은 다음 edge의 계산부터 사용한다.
현재 shot의 입력 수집과 detector 출력을 겹쳐 처리한다. 출력 register가 대기 중이면
값·ID·valid를 유지하며, D23이 소비된 edge에서 진행 상태를 초기화한다.
다음 edge부터 새 shot을 받을 수 있다. shot 사이 기록 register 자체를 지우지는 않지만
측정 수를 초기화해 새 shot에서 아직 쓰지 않은 기록은 참조하지 않는다.

[기본 testbench](tb/tb_measurement_to_detection.sv)는 입력 33개와 기대 출력 24개를
`.01` 파일에서 읽는다. 왼쪽 문자가 인덱스 0이므로 packed vector의 방향을 명시적으로
변환한다. 입력은 매 cycle 제시하고 sink는 항상 ready로 둔다.
출력 전송 시 bit·ID·last·순서·개수·shot 경계를 검사하며, X/Z·timeout·추가 출력은 실패다.

[실행기](scripts/run_sim.py)는 입력 hash와 길이를 검증하고, 실제 source 사본을 run 폴더에
보관한 뒤 compile/run을 실행한다. 종료 코드와 로그를 남기고 실제 출력 파일을 frozen
정답과 다시 비교한다. compiler 실패는 실행 파일이 생겼더라도 실패로 처리한다.
VCS와 로컬 보조 Icarus 실행을 지원하며, 서버에는 Python 표준 라이브러리만 필요하다.

## 2. 로컬 검증 결과

| 입력 묶음 | shots | RTL 출력 bits | frozen 정답 대비 불일치 |
| --- | --- | --- | --- |
| 무잡음 | 256 | 6,144 | 0 |
| 기준 기록과 단일 반전 | 34 | 816 | 0 |
| 임의 비트 | 1,024 | 24,576 | 0 |
| SI1000 | 4,096 | 98,304 | 0 |
| 합계 | 5,410 | 129,840 | 0 |

전체 178,530개 측정 bit를 수락하고 detector 129,840개를 출력했다.
정답은 P1에서 같은 측정 기록에 대해 Stim과 Python 직접 XOR가 일치한 frozen 파일이다.
기대값의 첫 detector bit를 의도적으로 반전한 별도 실행은 `MISMATCH`와 종료 코드 1로
실패해 scoreboard의 오류 검출도 확인했다.

검사 결과와 로그는 다음에 있다.

- [검증 JSON](../artifacts/qec_detector_rtl/p2_local_20261118_005/verification.json)
- [컴파일 로그](../artifacts/qec_detector_rtl/p2_local_20261118_005/sim/compile.log)
- [단일 반전 실행 로그](../artifacts/qec_detector_rtl/p2_local_20261118_005/sim/single_flip.log)
- [의도한 실패 로그](../artifacts/qec_detector_rtl/p2_local_20261118_005/sim/expected_failure.log)
- [단일 반전 VCD](../artifacts/qec_detector_rtl/p2_local_20261118_005/sim/single_flip.vcd)

### 로컬 실행 환경의 구분

Codex sandbox 안에서는 compiler가 `0xC0000005`로 종료하는 문제가 반복됐다.
실패 run `001`–`004`를 보존했으며 성공 결과로 바꾸지 않았다.
같은 최종 source를 sandbox 밖에서 실행한 `005`는 compiler 종료 코드 0으로 통과했다.
P2 회귀 검사도 sandbox 밖에서 3개 모두 통과했다.

따라서 실행 환경의 영향을 의심할 근거는 있지만 메모리 오류의 근본 원인을 확정하지는
않는다. verbose 로그, 명시적 표준 입력, 상대 compile 경로만으로 sandbox 안의 오류가
해결됐다고 주장하지 않는다. 정상 로컬 결과가 서버 VCS 실행 결과를 대신하지도 않는다.

## 3. 대표 경계의 VCD 기록

단일 반전 묶음의 첫 shot은 기준 기록이다. 해당 shot의 VCD를 텍스트로 읽어
입력 수락, output valid와 실제 전송을 구분했다. clock period는 8 ns다.

| detector | 마지막 필요 입력 수락(ns) | 처음 valid(ns) | 출력 전송(ns) |
| --- | --- | --- | --- |
| D0 | 52 | 60 | 68 |
| D4 | 84 | 92 | 100 |
| D20 | 260 | 268 | 276 |
| D22 | 276 | 284 | 292 |
| D23 | 252 | 292 | 300 |

D20과 D22는 마지막 data 측정의 경계를 포함한다. D23은 필요한 측정이 일찍 들어와도
D22 뒤의 ID 순서를 기다린다. 이 표는 기본 조건의 한 shot을 설명하는 기록이며,
P3의 전체 latency·처리 간격 검증이나 합성 timing 측정이 아니다.
근거는 [대표 trace](../artifacts/qec_detector_rtl/p2_local_20261118_005/report/representative_trace.json)다.
GTKWave/Verdi GUI로 파형을 확인하지는 않았다.

## 4. 테스트와 실행 명령

[P2 테스트](../tests/test_detector_rtl.py) 3개가 통과했다. 입력 손상 거부,
전송 ZIP의 상대 경로·hash·덮어쓰기 방지, 실제 RTL의 frozen 정답 비교와 의도한 오답 검출을
검사했다. 기존 P1 테스트 5개도 다시 실행해 모두 통과했다. 새 Python 파일의 문법 검사도 통과했다.

저장소 루트에서 Anaconda `fp` Python을 사용했다.

```powershell
python -m py_compile qec_detector_rtl/scripts/run_sim.py qec_detector_rtl/scripts/prepare_server_bundle.py tests/test_detector_rtl.py
python -m unittest discover -s tests -p test_detector_rtl.py -v
python -m unittest discover -s tests -p test_measurement_to_detection.py -v
python -m qec_detector_rtl.scripts.run_sim --vectors artifacts/qec_detector_rtl/d3_r3_z_v1_20261118_001/vectors --out-root artifacts/qec_detector_rtl/p2_local_20261118_005 --simulator iverilog
python -m qec_detector_rtl.scripts.prepare_server_bundle --vectors artifacts/qec_detector_rtl/d3_r3_z_v1_20261118_001/vectors --out-root artifacts/qec_detector_rtl/p2_server_bundle_20261118_001
```

위 `run_sim` 명령과 P2 RTL 테스트는 sandbox 밖에서 실행했다. 명령의 `python`은
`C:/Users/82108/anaconda3/envs/fp/python.exe`를 가리킨다. 재실행에는 새 output ID를 쓴다.

## 5. 서버에서 이어갈 항목

[전송 ZIP](../artifacts/qec_detector_rtl/p2_server_bundle_20261118_001.zip)은 RTL·testbench,
표준 라이브러리 실행기, P1 입력·정답·manifest와 checksum을 포함한다.
원래 P1 파일을 변경하지 않았고 checksum으로 동일성을 확인할 수 있다.
묶음의 실행기는 site package를 읽지 않는 `python -S`에서도 CLI를 시작하는 것을 확인했다.

[서버 실행 안내](SERVER_RUN.md)에 VPN·MobaXterm 전송 후 명령과 회수할 파일을 정리했다.
사용자는 현재 서버에 접속하지 않은 상태라고 확인했다. 아직 파일 전송, VCS 실행,
FSDB dump와 Verdi 확인은 수행하지 않았다.

서버에서는 다음을 확인한다.

1. checksum과 VCS/Verdi 환경을 확인하고 VCS에서 네 묶음을 실행한다.
2. `--fsdb`로 단일 반전 FSDB를 생성하고 대표 경계와 shot 전환을 Verdi에서 확인한다.
3. source·입력 hash, compile/run 로그, 실제 출력과 파형을 회수해 로컬 결과와 대조한다.

이 단계까지 확인한 뒤 P2 전체 완료로 표시한다. P3의 정체·입력 공백·중간 reset stress와
상세 cycle 측정, P4 합성·SDF 검증은 아직 진행하지 않았다.

## 6. 수업 PDF 참고 후 안내 개정

사용자 제공 수업 PDF의 simulation·pre-sim·FSDB·X11 절차와 현재 reset의 차이를
안내에 반영했다. 서버 작업은 사용자만 수행하며 개인정보는 검증 자료에 포함하지 않는다.
안내 개정본 [전송 ZIP 002](../artifacts/qec_detector_rtl/p2_server_bundle_20261118_002.zip)을
새로 준비하고 기존 001을 보존한다. RTL·testbench·runner와 frozen 입력은 위 로컬
통과본과 동일하다. 이번 안내 개정으로 서버 검증이나 새 simulation을 수행한 것은 아니다.

## 7. 금요일 전 실행 준비

서버에 Python이 없어도 실행하도록 기존 셸·VCS·파일 비교 도구를 사용하는
`scripts/run_vcs.sh`를 추가했다. 가짜 VCS를 사용한 로컬 제어 검사 4개와 전송 묶음 검사
1개가 통과했으며, 실제 VCS/FSDB 검증은 여전히 대기 중이다. RTL·testbench·기존 Python
실행기와 frozen 벡터는 바꾸지 않았다.

최신 [전송 ZIP 003](../artifacts/qec_detector_rtl/p2_server_bundle_20261118_003.zip)과
[금요일 이어하기](FRIDAY_CHECKLIST.md)를 사용한다. 원래 묶음은 보존한다.
