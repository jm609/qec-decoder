# QEC 측정 기록 → detection event RTL 실습

이 폴더는 `d=3`, 3라운드 rotated surface-code memory 회로의 측정 기록을
detection event로 변환하는 Python 기준 모델과 RTL 블록을 개발하는 공간이다.
상세 범위, 인터페이스, 검증 순서와 완료 기준은 [PLAN.md](PLAN.md)에 정리한다.

**현재 상태: P2 RTL·기본 testbench의 로컬 검증과 서버 전송 묶음 준비 완료. VCS·Verdi 검증은 서버 접속 후 진행한다.**

[P2 결과](P2_RESULTS.md): 로컬 Icarus에서 5,410개 기록, 129,840 detector bits의
불일치는 0이다. P2 테스트 3개와 기존 P1 테스트 5개가 통과했다.
서버 전송 ZIP과 실행 절차는 [서버 실행 안내](SERVER_RUN.md)에 연결한다.

[P1 결과](P1_RESULTS.md): 회귀 검사 5개 통과. 5,410개 기록의 129,840 detector bits에서
Stim 정답과 직접 XOR 결과의 불일치는 0이다.

[환경 점검 결과](ENVIRONMENT_CHECK.md): Stim과 작은 RTL 기능 simulation은 실행됐다.
Icarus compiler는 sandbox 안에서 비정상 종료했으나 같은 최종 코드를 sandbox 밖에서
실행한 P2 검사는 통과했다. GTKWave 실행과 서버 VCS·Verdi 검증은 아직 확인하지 않았다.

앞으로 사용할 주 EDA 흐름은 **VCS → Verdi → Design Compiler, 8 ns**다.
사용자가 수업 도구를 사용할 수 있다고 확인했으며, 로컬 Lab3 자료에서도 sky130 HD
합성 설정과 리포트를 확인했다. 실행은 **VPN 연결 → MobaXterm으로 수업 서버 접속 →
파일 전송 → 서버에서 도구 실행** 방식이다. 서버 작업 경로와 환경 초기화·라이브러리 설정은
접속 단계에서 확인한다.
도구별 역할과 재사용 조건은 [Synopsys 흐름](ENVIRONMENT_CHECK.md#6-synopsys-수업-환경-반영)에 정리한다.

**서버 접속과 서버 작업은 사용자가 직접 수행한다. Codex는 서버에 접속하지 않는다.**
사용자 제공 수업 PDF는 기술 절차의 참고 자료이며, 계정·접속 주소·인증정보는
안내나 전송 묶음에 옮기지 않는다.
금요일(2026-11-20)에는 [이어하기 순서](FRIDAY_CHECKLIST.md)를 따른다.
최신 [전송 ZIP](../artifacts/qec_detector_rtl/p2_server_bundle_20261118_003.zip)은 Python 없는
셸 실행 경로를 포함한다. RTL·testbench·기존 Python 실행기는 로컬 통과본과 같다.
서버에 새 도구를 설치하지 않는다.

## 폴더 구성

| 경로 | 용도 |
| --- | --- |
| `PLAN.md` | 단계별 실행 계획과 최초 설계 명세 |
| `ENVIRONMENT_CHECK.md` | 설치 도구, 실제 실행 결과와 미해결 환경 문제 |
| `P1_RESULTS.md` | Python 기준 모델과 frozen 벡터의 검증 결과 |
| `P2_RESULTS.md` | RTL 기본 기능의 로컬 결과와 서버 확인 대기 항목 |
| `SERVER_RUN.md` | 전송 묶음의 VCS·FSDB 실행 및 결과 회수 절차 |
| `FRIDAY_CHECKLIST.md` | 금요일의 실행 순서·회수 파일·P2 완료 기준 |
| `model/` | 회로에서 XOR 참조를 추출하는 Python 기준 모델과 검증 벡터 생성기 |
| `rtl/` | 합성 가능한 Verilog 모듈 |
| `tb/` | RTL testbench와 입출력 규약 검사 |
| `scripts/` | 시뮬레이션 실행·결과 비교와 확정된 합성 flow의 실행 파일 |
| `../tests/test_measurement_to_detection.py` | Python 기준 모델·파일 검증의 회귀 검사 |
| `../tests/test_detector_rtl.py` | RTL 기본 기능·입력 무결성·전송 묶음 검사 |
| `../artifacts/qec_detector_rtl/` | 생성 회로·벡터·파형·로그·합성 결과·검증 보고서 |

기준 모델, RTL, 기본 testbench, 실행기와 서버 묶음 생성기를 구현했다.
Design Compiler 합성 스크립트와 P3 stress bench는 아직 작성하지 않았다.

## 현재 실행 명령

Anaconda `fp` 환경에서 저장소 루트를 작업 경로로 사용한다.

GitHub에서 새로 받은 경우에는 아래 명령으로 벡터부터 생성한다.
Python 기준 모델에는 저장소의 Python 의존성이 필요하고, RTL simulation에는
Icarus Verilog의 `iverilog`와 `vvp`가 PATH에 있어야 한다.
기존 폴더가 있으면 새 output 이름을 사용한다.

```powershell
python -m qec_detector_rtl.model.generate_vectors --out-root artifacts/qec_detector_rtl/example_p1
python -m qec_detector_rtl.scripts.run_sim --vectors artifacts/qec_detector_rtl/example_p1/vectors --out-root artifacts/qec_detector_rtl/example_p2 --simulator iverilog
```

아래 명령의 `d3_r3_z_v1_20261118_001` 벡터와 서버 전송 ZIP은 로컬 생성물이며
Git에는 포함하지 않는다. 새 clone에서 해당 생성물이 없으면 관련 회귀 검사는
건너뛸 수 있다.

```powershell
python -m unittest discover -s tests -p test_measurement_to_detection.py -v
python -m unittest discover -s tests -p test_detector_rtl.py -v
python -m qec_detector_rtl.scripts.run_sim --vectors artifacts/qec_detector_rtl/d3_r3_z_v1_20261118_001/vectors --out-root artifacts/qec_detector_rtl/p2_local_20261118_006 --simulator iverilog
python -m qec_detector_rtl.scripts.prepare_server_bundle --vectors artifacts/qec_detector_rtl/d3_r3_z_v1_20261118_001/vectors --out-root artifacts/qec_detector_rtl/p2_server_bundle_20261118_004
```

P1 `001` run은 검증된 입력으로 보관한다. 모든 output에는 아직 사용하지 않은 run ID를
지정한다. 위 예시의 폴더도 이미 존재하면 다른 ID를 사용한다. 기존 run은 덮어쓰지 않는다.
Codex sandbox 안에서 compiler가 실패하면 종료 코드를 무시하지 않는다.
위 로컬 RTL 검사와 P2 테스트의 정상 결과는 sandbox 밖에서 얻었다.

## 단계별 진행

1. 완료: 기존 builder를 재사용한 Python 기준 모델과 Stim 정답 비교, frozen 벡터 저장.
2. 현재: 고정 RTL·기본 testbench는 로컬 통과. 전송 묶음으로 수업 서버의 VCS·Verdi 검증을 진행한다.
3. 이후: 입출력 규약과 cycle 지연·처리 간격을 검증하고 Verdi에서 FSDB 파형을 확인한다.
4. 마지막: Design Compiler 8 ns 합성, SDF를 적용한 gate-level 검증 및 결과 보고를 진행한다.

로컬 Python 실행은 저장소 루트 `C:/Users/82108/fp`를 기준으로 한다.
RTL·testbench·서버 실행 스크립트가 준비되면 frozen 벡터와 함께 전송 묶음을 만들고,
VCS·Verdi·DC는 수업 서버의 별도 작업 폴더에서 실행한다. 로그·파형·리포트는 회수해
같은 run의 로컬 생성물 경로에 보관한다.
Python 테스트는 저장소의 `tests/`에 추가하고, 생성물은
`artifacts/qec_detector_rtl/<run_id>/`에 저장한다. 기존 run을 덮어쓰지 않는다.

기존 neural pre-decoder 연구의 정확도 평가와 이 블록의 기능·시간 검증은
각각의 결과로 보고한다. 이 블록만으로 전체 실시간 QEC 성능을 주장하지 않는다.
