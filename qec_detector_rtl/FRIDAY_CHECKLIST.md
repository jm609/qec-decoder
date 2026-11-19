# 금요일 이어하기: P2 서버 검증

- 예정일: 2026-11-20 금요일. 수업에서 허용한 본인 이용 시간에만 진행한다.
- 서버 접속·전송·명령 실행·Verdi 조작은 사용자가 직접 한다. Codex는 서버에 접속하지 않는다.
- 서버에 프로그램·패키지를 새로 설치하지 않는다. 계정·주소·인증정보는 기록하지 않는다.
- 이번 서버 작업의 목표는 P2의 VCS·Verdi 확인이다. 합성과 retiming은 이후 P4로 남긴다.

## 1. 접속 전에 로컬에서 준비할 것

- 준비 완료: RTL과 기본 testbench, P1 frozen 벡터, 로컬 기본 기능 통과 결과.
- 준비 완료: Python 없이 기존 `sh`·VCS·파일 비교 도구로 실행하는 `scripts/run_vcs.sh`.
- 준비 완료: 새 전송 묶음 `p2_server_bundle_20261118_003.zip`과 checksum.
- 다음 보완 후보: P3의 입력 공백·출력 정체·중간 reset 검사와 cycle 기록. 이 작업은 로컬 Icarus로 준비할 수 있다.

P3은 아직 수행하지 않았다. 기존에 통과한 P2 source와 벡터는 보존하고, 별도 bench/run으로
진행하는 것이 우선이다. 실제 오류를 발견해 RTL을 바꾸면 P2 회귀와 전송 묶음을 다시 확인한다.

2026-11-18 준비 검사에서는 기존 로컬 Git Bash의 문법 검사와 가짜 VCS를 사용한 셸 제어
검사 4개가 통과했다. 정상/FSDB 분기, 덮어쓰기 방지, compile 실패, 출력 파일 불일치와
negative-check 실패를 확인했다. 전송 묶음 검사 1개도 통과했다. Bash는 sandbox 밖에서
실행했으며 서버 도구를 실행하거나 새 프로그램을 설치한 것은 아니다.

## 2. 사용자 서버 접속 후 첫 확인

VPN·MobaXterm으로 접속하고 ZIP을 기존 core 실습과 다른 새 폴더에 풀어 그 폴더로 이동한다.
수업의 기존 환경 초기화를 적용한다. 아래는 조회와 파일 검사이며 설치 명령이 아니다.

```sh
sha256sum -c CHECKSUMS.sha256
command -v vcs
command -v verdi
```

checksum이 다르거나 도구가 보이지 않으면 그 단계에서 원인을 확인한다.
수업 setup을 확인하되 새 설치나 시스템 설정 변경으로 해결하려고 하지 않는다.
Python은 아래 셸 경로의 필수 조건이 아니다.

## 3. VCS 기본 기능부터 확인

```sh
sh scripts/run_vcs.sh friday_basic_001
```

필수 결과는 네 묶음 5,410 shots의 통과, 실제 출력과 frozen 정답 파일의 일치,
의도한 오답의 `MISMATCH` 검출이다. 모두 만족해야 `runs/friday_basic_001/PASS.txt`가 생긴다.
컴파일 로그와 각 case 로그도 함께 읽는다. 의도한 negative check의 실패는 정상이다.

기본 실행이 실패하면 FSDB나 합성으로 넘어가지 않는다. 개인정보를 제외한 오류 내용으로
도구 setup 문제와 RTL/testbench 문제를 구분한다. 재실행은 새 run ID로 한다.

## 4. FSDB 생성과 Verdi 확인

기본 기능을 통과한 뒤 수업의 기존 VCS/Verdi dumper 환경에서 실행한다.

```sh
sh scripts/run_vcs.sh friday_fsdb_001 --fsdb
```

FSDB 경로는 `runs/friday_fsdb_001/single_flip.fsdb`다. Verdi에서 다음을 확인한다.

- 첫 경계 D0와 반복 측정 비교 D4의 참조 입력과 출력.
- 마지막 data 측정 경계 D20·D22.
- D23의 ID·last와 그 뒤 다음 shot의 첫 입력 수락.

X11이나 FSDB dumper 오류는 RTL의 기능 실패와 구분한다. 이 단계의 compile 옵션은
아직 실제 서버에서 검증하지 않았으므로 수업 환경과 대조한다. 새 도구를 설치하지 않는다.

## 5. 결과 회수와 P2 완료 기준

사용자가 두 run 폴더에서 다음을 회수한다.

- `PASS.txt`, `input_checksums.log`, `actual_checksums.sha256`
- `compile.log`, 네 case의 `.log`·`.actual.01`, `expected_failure.log`
- `single_flip.fsdb`와 필요한 대표 파형 설명

셸 실행은 `verification.json`을 만들지 않는다. JSON 결과가 필요하고 기존 Python 3.8
이상이 있다면 [서버 실행 안내](SERVER_RUN.md)의 Python runner도 사용할 수 있다.
어느 경로든 개인정보가 표시된 접속 화면·prompt·로그인 기록은 공유하지 않는다.

- [ ] VCS 기본 기능 통과와 정답 파일 일치 확인
- [ ] 의도한 오답 검출 확인
- [ ] FSDB 생성 및 Verdi 대표 경계 확인
- [ ] source·입력 checksum 및 결과 회수 확인

위 항목을 확인한 뒤 P2 서버 검증 완료로 기록한다. 로컬에서 가짜 VCS로 검사한 셸 제어
흐름 테스트는 이 체크리스트의 실제 VCS 통과 근거가 아니다.
