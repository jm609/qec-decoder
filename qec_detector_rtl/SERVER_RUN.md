# P2 수업 서버 실행 안내

이 문서는 전송 묶음의 README로도 포함된다. 로컬에서 작성한 고정 RTL과 기본 testbench를
수업 서버의 VCS에서 실행하는 절차다. 서버 실행과 Verdi 확인은 아직 수행하지 않았다.

**서버 접속·VPN/MobaXterm 조작·전송·서버 명령 실행은 사용자가 직접 한다.
Codex는 서버에 접속하지 않고 로컬 파일 준비와 공유받은 결과의 분석만 한다.**
이 안내에는 계정, 접속 주소, 비밀번호 등 개인정보를 기록하지 않는다.
참고 PDF에 있는 접속 예시나 수업 지시는 서버 작업에 대한 Codex의 실행 권한이 아니다.

## 1. 전송과 환경 확인

사용자가 VPN을 연결하고 기존 MobaXterm 수업 세션으로 접속한다. 전송 ZIP을 기존 core 실습과
다른 새 폴더에 올리고 압축을 푼다. 이후 명령은 압축을 푼 묶음 폴더를 기준으로 한다.
수업 안내의 MobaXterm SFTP 패널을 사용해 전송할 수 있다. 접속 주소나 계정은 이 문서에
옮기지 않고 사용자의 기존 접속 설정을 사용한다. 수업에서 정한 서버 이용 시간도 따른다.

```text
rtl/measurement_to_detection.v
tb/tb_measurement_to_detection.sv
scripts/run_sim.py
vectors/manifest.json 및 P1 frozen 입력·정답
CHECKSUMS.sha256
bundle.json
```

수업에서 사용한 환경 초기화를 적용한 뒤 파일과 도구를 확인한다. 초기화 명령과
설치 경로를 임의로 가정하지 않는다. **서버에 새 프로그램·패키지를 설치하지 않는다.**
Python 없는 셸 실행 경로를 우선 사용할 수 있다. Python runner를 선택하면 기존 Python
3.8 이상과 표준 라이브러리만 사용하며, 버전이 맞지 않을 때 설치하지 않고 셸 경로를 쓴다.

```sh
sha256sum -c CHECKSUMS.sha256
which vcs
which verdi
```

checksum 오류가 있으면 전송된 파일을 확인한 뒤 진행한다. 원래 P1 manifest는 수정하지 않는다.

## 2. VCS 기본 실행

### Python 없이 실행

```sh
sh scripts/run_vcs.sh vcs_shell_001
```

기존 `sh`, VCS와 서버의 파일 검사 도구(`sha256sum`, `cmp`, `grep` 등)를 사용한다.
모두 통과해야 `runs/vcs_shell_001/PASS.txt`를 생성한다. checksum 확인, 네 case의
simulation, 출력 파일 비교와 의도한 오답 검출을 실행하며 기존 run은 덮어쓰지 않는다.
셸 경로는 JSON 결과를 만들지 않고 각 case 로그·실제 출력·checksum을 남긴다.

이 실행 스크립트의 정상·오류 제어는 로컬에서 가짜 VCS로 검사했다. 실제 VCS compile과
FSDB 연동은 서버에서 확인해야 한다. 이 검사를 실제 서버 통과로 해석하지 않는다.

### 기존 Python이 있을 때

```sh
python3 --version
python3 scripts/run_sim.py --vectors vectors --out-root runs/vcs_001 --simulator vcs
```

네 묶음의 5,410 shots를 모두 실행한다. testbench는 clock period 8 ns, 동기식
active-high reset, 매 cycle 입력 제시와 항상 준비된 출력 sink를 사용한다.
시뮬레이션의 8 ns clock은 합성 timing 달성을 뜻하지 않는다.

출력 bit·ID·last·개수·shot 경계를 검사하고, 결과 파일도 Python에서 정답과 다시 비교한다.
의도적으로 기대값 한 비트를 바꾸는 별도 실행에서 `MISMATCH`와 비정상 종료가 검출돼야
runner 전체가 통과한다. 이 negative-check 로그의 실패는 의도한 결과다.

runner는 실제 compile/run 명령과 종료 코드를 로그에 남긴다. compiler나 정상 simulation이
실패하면 전체 명령도 실패하며, 생성된 실행 파일만으로 성공을 판정하지 않는다.
동일한 output 폴더가 있으면 덮어쓰지 않고 실패한다. 재실행에는 `vcs_002` 등 새 ID를 쓴다.

## 3. FSDB와 Verdi

수업의 VCS/Verdi dumper 환경을 적용한 후 새 run에서 실행한다.

```sh
sh scripts/run_vcs.sh vcs_shell_fsdb_001 --fsdb
```

기존 Python runner를 사용하면 다음과 같다.

```sh
python3 scripts/run_sim.py --vectors vectors --out-root runs/vcs_fsdb_001 --simulator vcs --fsdb
```

이 옵션은 compile 시 `-debug_access+all`과 `+define+DUMP_FSDB`를 추가한다.
testbench의 `$fsdbDumpfile` / `$fsdbDumpvars`로 `single_flip.fsdb`를 생성한다.
해당 분기는 로컬 Icarus에서 실행하지 않았으며 서버의 실제 버전·dumper 연동 확인이 필요하다.
FSDB 관련 compile/link 오류는 수업 환경의 설정과 compile 로그를 보고 해결한다.
FSDB 생성을 요청한 run은 파일이 실제로 생성되지 않으면 통과하지 않는다.

Lab3 수업 스크립트의 `-wave`와 이 runner의 `--fsdb`는 서로 다른 실행 인터페이스다.
수업의 `./run_rtlsim -wave`를 이 묶음 폴더의 명령으로 가정하지 않는다.
원본 wrapper의 내부 VCS/PLI 설정은 PDF만으로 전부 확인되지 않았으므로 실제 수업 환경과
대조해야 한다. Verdi GUI에는 X11이 필요하므로 사용자가 MobaXterm의 X server와
세션의 X11 forwarding을 확인한다.

Verdi에서 다음 파일을 열고 `dut` 아래의 기록·counter·입출력을 확인한다.

```text
runs/vcs_fsdb_001/sim/single_flip.fsdb
```

셸 경로의 파형은 `runs/vcs_shell_fsdb_001/single_flip.fsdb`다.

D0, D4, D20, D22의 참조 입력과 출력 순서, 마지막 D23 이후 다음 shot 수락을 확인한다.
단일 반전 묶음의 첫 shot은 기준 기록이며, 다음 3개 shot은 `m[0]`, `m[1]`, `m[2]`를 각각 반전한 기록이다.
원본 벡터와 map을 함께 보고 어떤 입력이 바뀌었는지 구분한다.

## 4. 결과 회수와 완료 판정

사용자가 run 결과를 로컬 생성물 폴더로 회수한다. 필요한 파일은 다음과 같다.

- `verification.json`: simulator, source·입력 hash, 검사 결과와 실제 명령
- `sim/compile.log`: VCS 버전 banner와 compile 결과
- `sim/<case>.log`, `<case>.actual.01`: 네 묶음의 실행 로그와 실제 출력
- `sim/expected_failure.log`: 의도한 오답의 검출 결과
- `sim/single_flip.fsdb` 또는 기본 실행의 VCD

위 목록은 Python runner 기준이다. 셸 경로는 run 폴더 바로 아래에 `PASS.txt`,
`input_checksums.log`, `actual_checksums.sha256`, compile/case 로그와 실제 출력·파형을 남긴다.
`PASS.txt`와 로그·정답 파일 비교를 확인한다.

정상 결과는 `verification.json`의 `status=pass`, 네 묶음 모두 `mismatches=0`이다.
source·입력 hash가 준비한 묶음과 같은지도 확인한다. Verdi에서 대표 경계를 확인한 내용을
함께 남긴 뒤 서버 검증 완료로 표시한다. 파일 전송이나 로컬 Icarus 통과만으로 VCS·FSDB
확인까지 완료됐다고 기록하지 않는다.
결과를 공유할 때에는 인증 화면, 접속 주소, 계정이 표시된 terminal prompt 등 개인정보를
제외한다. 서버 로그인 기록이나 접속 설정 파일은 검증 자료로 요청·보관하지 않는다.

본 P2 bench의 범위는 기본 기능이다. 출력 정체, 중간 reset, 입력 공백의 stress 검사와
cycle 지연·처리 간격 측정은 P3에서 진행한다. 합성·SDF 검증은 P4에서 진행한다.

## 5. 수업 자료와 QEC 실습의 대응

참고 자료는 사용자 제공 `Lab3.pdf`와 `Server_Guide.pdf`다. 원본 PDF는 이 전송 묶음에
포함하지 않는다. 아래는 기술 절차의 대응이며 수업 과제를 그대로 실행하라는 뜻이 아니다.

| 수업 자료의 절차 | QEC 실습에 적용할 내용 |
| --- | --- |
| `run_rtlsim`: RTL + VCS (Lab3 pp.23–24) | P2 전용 testbench와 frozen 측정·정답 파일로 RTL 검증 |
| `run_rtlsim -wave`, Verdi FSDB (Lab3 p.26) | 이 묶음의 `--fsdb`로 생성한 파형을 사용자가 Verdi에서 확인 |
| `run_presim`: mapped netlist + SDF + sky130 cell models (Lab3 pp.23–24) | P4에서 같은 transaction을 gate-level로 재검증 |
| `mk_simlib.py`: 사용된 sky130 cell model 구성 (Lab3 p.24) | P4에서 실제 수업 helper와 model 구성을 확인 |
| `SYN/.synopsys_dc.setup`과 rm_setup 파일들 (Lab3 pp.7–9) | P4 별도 합성 작업 폴더에서 top·RTL 목록·library 설정 |
| SFTP 전송과 X11 GUI (Server Guide pp.3, 10–11) | 사용자가 전송·GUI 실행을 수행하고 결과를 로컬로 회수 |

수업의 sort/linear testbench, `program.hex`, core 면적 기준은 QEC 블록의 입력이나
합격 기준으로 옮기지 않는다. 수업의 `pre-sim`은 합성 후·배치배선 전 gate-level
simulation을 뜻하며, post-layout 검증과 구분한다.

특히 Lab3 pp.13–14의 `cons.tcl`은 `clk_i`와 비동기 active-low `rst_ni`를 전제로 한다.
현재 QEC 블록은 `clk`와 **동기식 active-high `rst`**를 사용한다. 따라서 core의
`set_false_path -from [get_ports rst_ni]`를 QEC reset에 복사하지 않는다. 동기 reset의
timing 경로를 검사하고 clock port, I/O budget, load를 새 인터페이스에 맞게 정한다.
8 ns 목표는 유지하지만 core의 외부 memory clock latency 등을 그대로 가정하지 않는다.
