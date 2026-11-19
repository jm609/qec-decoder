# RTL 시뮬레이션 환경 점검

- 점검일: 2026-11-18, Asia/Seoul
- 범위: 현재 Codex 실행 환경의 도구 실행, 작은 RTL의 compile/run, 파일 입출력, 실패 검출, 파형 저장, Stim sampling
- Windows 점검 판정: **기능 시뮬레이션 실행 가능. 컴파일러의 비정상 종료와 GTKWave 실행 문제는 미해결이다.**
- 후속 도구 선정: 사용자 확인에 따라 VCS·Verdi·Design Compiler를 주 흐름으로 사용한다. 이 환경의 실제 실행 점검은 별도다.
- 결과 기록: [result.json](../artifacts/qec_detector_rtl/toolchain_smoke_20261118_003/result.json)

## 1. Windows 초기 점검 도구와 결과

| 역할 | 도구와 경로 | 확인 결과 |
| --- | --- | --- |
| QEC 회로와 측정 기록 생성 | Stim 1.15.0, Anaconda `fp` 환경 | 무잡음·잡음 회로 sampling과 measurement-to-detector 변환 통과 |
| RTL 컴파일 | `C:/iverilog/bin/iverilog.exe`, Icarus Verilog 12.0 (devel), `s20150603-1539-g2693dd32b` | 최종 점검 3회 모두 종료 코드 0. 앞선 점검에서 `0xC0000005` 관측 |
| RTL 시뮬레이션 실행 | `C:/iverilog/bin/vvp.exe`, runtime 12.0 (devel) | XOR·reset·출력 정체 검사 통과, 오류 시 비정상 종료 확인 |
| 파형 저장 | testbench의 `$dumpfile` / `$dumpvars` | VCD 생성과 기본 파일 구조 확인 |
| 파형 GUI | `C:/iverilog/gtkwave/bin/gtkwave.exe` | 설치 경로 확인. `--version` 실행부터 WinError 623으로 실패 |
| 합성 및 물리 timing | 사용할 타깃·flow 미확정 | 이번에 실행하지 않음 |

Stim은 양자 회로의 측정 기록과 정답을 생성한다. Icarus Verilog는 작성할 디지털 RTL을
컴파일하고 실행한다. GTKWave는 생성된 파형을 보는 프로그램이다. 이 세 도구의 역할은
서로 다르며, Icarus의 기능 simulation 결과를 합성 timing·area 결과로 사용하지 않는다.
Icarus의 컴파일과 language flag는 [공식 CLI 문서](https://github.com/steveicarus/iverilog/blob/master/Documentation/usage/command_line_flags.rst)를 참고한다.

Python 확인 환경은 다음과 같다.

```text
Python executable: C:/Users/82108/anaconda3/envs/fp/python.exe
Python: 3.10.20, conda-forge
NumPy: 2.2.6
Stim: 1.15.0
```

기본 `python`이 같은 환경을 가리킨다고 가정하지 않는다. 이후 명령도 `fp` 환경을
활성화하거나 위 executable을 명시한다.

## 2. 실제 실행한 작은 RTL 검사

본 검사는 도구 점검용 5비트 parity register와 testbench를 사용했다.
계획의 33비트 측정 기록 변환기나 실제 QEC detector RTL은 구현하지 않았다.

| 검사 | 입력과 확인 방법 | 결과 |
| --- | --- | --- |
| 컴파일 | Verilog 모듈 + SystemVerilog testbench, `-g2012 -Wall -s tb_toolchain` | 최종 run에서 3회 모두 종료 코드 0, warning 출력 없음 |
| reset | 동기식 active-high reset 후 output valid 확인 | 통과 |
| XOR 기능 | 5비트 입력의 32개 조합 전체를 Python parity와 비교 | 출력 32개, mismatch 0 |
| 입력 파일 | `$readmemb`로 입력과 기대값 파일 읽기 | 32개 벡터 처리 통과 |
| 출력 정체 | sink ready를 주기적으로 낮추고 valid·값 유지 검사 | 28 stall cycles 관측, 누락·중복 없이 완료 |
| 결과 파일 | `$fdisplay`로 RTL 출력 기록 후 Python으로 다시 비교 | 통과 |
| 실패 검출 | 기대값 첫 비트를 의도적으로 반전 | `$fatal(1, ...)`, runtime 종료 코드 1 |
| 파형 저장 | `$dumpfile`, `$dumpvars`로 VCD 생성 | 7,619 bytes, 정의부와 시간 기록 존재 |

정상 testbench의 출력은 다음과 같다.

```text
PASS words=32 outputs=32 stall_cycles=28 cycles=104
```

104 cycles는 이 도구 점검용 testbench의 실행 길이다. QEC 블록의 latency나 initiation
interval로 인용하지 않는다. VCD의 기본 텍스트 구조는 검사했지만 GUI로 시각 검토하지는
못했다. 이 검사로 SystemVerilog 전체, SVA, UVM 등의 지원까지 확인한 것은 아니다.

소스와 로그는 아래에 보관한다.

- [Verilog 모듈](../artifacts/qec_detector_rtl/toolchain_smoke_20261118_003/parity_stage.v)
- [testbench](../artifacts/qec_detector_rtl/toolchain_smoke_20261118_003/tb_toolchain.sv)
- [첫 정상 컴파일 로그](../artifacts/qec_detector_rtl/toolchain_smoke_20261118_003/compile_0.log)
- [정상 실행 로그](../artifacts/qec_detector_rtl/toolchain_smoke_20261118_003/positive.log)
- [의도한 실패의 로그](../artifacts/qec_detector_rtl/toolchain_smoke_20261118_003/negative.log)
- [VCD 파형](../artifacts/qec_detector_rtl/toolchain_smoke_20261118_003/smoke.vcd)

아래는 최종 run 폴더에서 실제 사용한 명령이다. 다시 실행할 때에는 벡터와 source를
새 run 폴더에 복사한 뒤 그 폴더를 작업 경로로 사용한다. 기존 로그·출력·파형을 덮어쓰지 않는다.
`negative/` 하위 폴더도 미리 준비한다.

```powershell
& C:/iverilog/bin/iverilog.exe -g2012 -Wall -s tb_toolchain -o checked_0.vvp parity_stage.v tb_toolchain.sv
& C:/iverilog/bin/vvp.exe checked_0.vvp
& C:/iverilog/bin/vvp.exe checked_0.vvp +force_failure
```

마지막 명령은 종료 코드 1이 기대값인 실패 경로 검사다. 정상 실행 명령과 구분한다.

## 3. Stim 실행 점검

`rotated_memory_z`, `d=3`, 3라운드에서 무잡음·잡음 각각 128 shots를 생성하고,
각 측정 배열을 같은 회로의 `compile_m2d_converter()`에 전달했다.

| 회로 | 측정 배열 | detector 배열 | detection event 합계 |
| --- | --- | --- | --- |
| 무잡음 | 128 × 33 | 128 × 24 | 0 |
| 잡음 | 128 × 33 | 128 × 24 | 61 |

잡음 점검은 Stim generator의 `after_clifford_depolarization=0.001`,
`before_measure_flip_probability=0.005`, sampling seed 1101801을 사용했다.
이는 실행 환경을 확인하기 위한 작은 검사이며, 계획한 SI1000 검증 벡터 생성이나
독립 Python 기준 모델 구현을 완료한 것은 아니다.

## 4. 발견한 문제와 후속 처리

### 4.1 Icarus 컴파일러의 비정상 종료

초기 `toolchain_smoke_20261118_001`과 `002` 검사에서는 `.vvp` 파일이 생성됐지만
컴파일 종료가 `0xC0000005`였다. PowerShell에서는 `-1073741819`, Python에서는
`3221225477`로 관측됐다. `002`의 [컴파일 로그](../artifacts/qec_detector_rtl/toolchain_smoke_20261118_002/compile_0.log)에 원래 종료 코드를 보관했다.

이후 flag 조합을 바꾼 진단과 최종 `003` run의 동일 명령 3회는 모두 정상 종료했다.
작업 폴더 내부의 짧은 TMP/TEMP를 사용한 `002`에서도 실패했으므로, 임시 경로 변경을
해결책으로 간주하지 않는다. 원인이 특정 flag나 source라고 확정하지도 않는다.
후속 성공으로 앞선 크래시가 해결됐다고 판정하지 않는다.

향후 simulation runner는 컴파일 종료 코드가 0이 아니면 즉시 실패시켜야 한다.
출력 `.vvp`가 존재하거나 우연히 실행된다는 이유로 컴파일 실패를 무시하지 않는다.
본격적인 RTL 자동 검증 전에는 이 빌드의 안정성을 확인하거나 별도 안정 버전의
도구 실행 환경을 마련해야 한다. 현재 Codex 밖의 일반 PowerShell에서도 같은 오류가
발생하는지는 확인하지 않았다.

### 4.2 GTKWave 실행 실패

`gtkwave --version`부터 WinError 623, `잘못된 DLL 재할당`으로 실패했다.
원래 환경과 GTKWave 폴더에서 최소 PATH를 지정한 환경 모두에서 동일하게 관측했다.
단순히 PATH를 줄이는 것으로 해결하지 못했다.

현재 환경에서는 GTKWave의 GUI 사용 가능 여부를 통과로 표시하지 않는다.
VCD 저장과 텍스트 기반 결과 비교는 실행됐으므로 Python 모델 및 기능 검증 준비는
진행할 수 있다. 파형 GUI 사용이 필요한 단계에서는 별도로 실행 환경을 복구하거나
정상 실행되는 viewer를 확보한다. 이번 점검에서 새 도구 설치, 시스템 DLL 교체,
보안 설정 변경은 하지 않았다.

### 4.3 합성 도구

이번 PATH 조사에서는 `yosys`, `vivado`, `dc_shell`, `verilator`가 발견되지 않았다.
설치 폴더 전체나 학교 서버·별도 EDA 환경은 조사하지 않았으므로 미설치로 단정하지 않는다.
합성 타깃·library·constraints는 여전히 미확정이다.

## 5. 다음 단계의 판단

P1의 Python 기준 모델과 frozen 벡터는 이후 완료했다. 상세 결과는 [P1 결과](P1_RESULTS.md)에 있다.
Windows의 작은 RTL compile/run은 실행됐지만 compiler 크래시는 미해결이다.
후속 P2부터는 아래 Synopsys 흐름을 우선 사용한다. Icarus/GTKWave 점검 기록은 로컬
보조 환경의 이력으로 보관한다. 환경 점검 결과와 실제 detector RTL의 검증 결과를 구분한다.

## 6. Synopsys 수업 환경 반영

### 사용자 확인과 로컬 자료

2026-11-18 사용자가 Synopsys VCS, Verdi, Design Compiler를 사용할 수 있다고 확인했다.
첫 합성 목표는 수업과 같은 **8 ns, 125 MHz**로 정한다. 이는 목표 제약이며 새 QEC 블록에서
달성한 timing이 아니다. 현재 Windows PATH에서는 `vcs`, `verdi`, `dc_shell`을 찾지 못했으므로
이 세 도구를 현재 Codex에서 실행했다고 기록하지 않는다.

로컬 `C:/Users/82108/OneDrive/바탕 화면/asic/lab3/`에서 다음 자료를 읽었다.

| 자료 | 확인한 내용 |
| --- | --- |
| `rm_setup/dc_setup.tcl` | RTL 목록, target/link library 변수, reports/results 경로 설정 |
| `baseline/script/dc.tcl` | analyze/elaborate/link, constraints 적용, `compile_ultra -no_autoungroup`, timing·area·power 보고, mapped Verilog·SDF·SDC 출력 |
| `retime/script/dc.tcl` | 같은 흐름에 `-retime`을 추가한 비교 구성 |
| `baseline/reports/0_core_time.rpt` | Design Compiler S-2021.06-SP4, `tt_025C_1v80`, sky130 HD |
| `baseline/reports/1_core_area.rpt` | `sky130_fd_sc_hd__tt_025C_1v80`, 기존 서버 DB 경로 `/tools/PDK/sky130/lib/sky130_fd_sc_hd__tt_025C_1v80.db` |

이는 기존 RISC-V core의 자료다. 이 자료의 면적·timing을 QEC 블록의 결과로 가져오지 않는다.
기존 리포트의 DB 경로도 새 실행 호스트에서 실제 존재하는지 확인한 뒤 사용한다.
라이브러리 설정의 근거는 [Lab3 area 리포트](<C:/Users/82108/OneDrive/바탕 화면/asic/lab3/baseline/reports/1_core_area.rpt>)다.
이 초기 점검에서는 `cons.tcl`, 전체 common setup, VCS 실행 스크립트와 cell simulation
model 위치를 확보하지 못했다. 이후 제공 PDF에서 확인한 설정·절차는 8절에 보완한다.
실제 서버의 전체 파일을 확보하거나 원래 수업 파일을 수정한 것은 아니다.

### 단계별 사용

| 단계 | 도구 | 적용할 내용과 근거 |
| --- | --- | --- |
| P2–P3 기능 검증 | [VCS](https://www.synopsys.com/verification/simulation/vcs.html) | 전용 testbench와 frozen `.01` 입력/정답으로 값·ID·last·handshake 검증 |
| 파형 확인 | [Verdi](https://www.synopsys.com/verification/debug/verdi.html) | FSDB로 첫·마지막 경계, 출력 정체, reset과 cycle 관측 |
| P4 합성 | [Design Compiler](https://www.synopsys.com/implementation-and-signoff/rtl-synthesis-test/design-compiler.html) | 수업 sky130 HD 환경을 재사용해 8 ns 제약에서 timing·cell area 확인 |
| 합성 후 검증 | VCS + Verdi | mapped netlist, 대응 cell model, SDF로 같은 입출력 검사를 반복하고 FSDB 확인 |

`dc_setup.tcl`의 RTL 목록과 top을 새 블록으로 바꾸고, 별도 작업 폴더에서 실행한다.
수업의 sort·linear 프로그램 입력 자리는 이 블록의 측정 기록·detector 정답 벡터와
전용 testbench로 연결한다. 수업의 core용 I/O delay와 load는 QEC 인터페이스에 맞춰 검토한다.
FSDB dump 연동은 수업의 VCS/Verdi 버전과 PLI 설정을 확인해 구성한다.

본 계획에서는 RTL simulation과 SDF를 적용한 합성 후 gate-level simulation을 구분한다.
이후 Lab3 PDF에서 `pre-sim`의 명령 역할과 model 구성은 확인했다(8절).
기능 비교는 동일한 transaction의 값·순서·개수와 약속한 cycle 기준으로 수행하고,
SDF annotation 성공 여부와 timing-check violation을 별도로 검사한다.
배치·배선 전의 SDF 결과를 post-layout timing이나 sign-off 결과로 부르지 않는다.

`-retime`은 baseline 기능·timing을 확인한 뒤 필요한 경우 별도 run에서 비교한다.
첫 RTL 기능 검증의 필수 조건으로 추가하지 않는다. 적용하면 reset·handshake와 외부
cycle 동작을 다시 확인한다. power 리포트를 남길 경우 switching activity를 주었는지와
기본 추정 조건을 표시한다. 타깃·activity가 다른 리포트를 직접 비교하지 않는다.

### 접속 방식과 작업 분담

사용자가 확인한 실행 방식은 **VPN 연결 후 MobaXterm으로 수업 서버에 접속하고,
파일을 옮긴 뒤 서버에서 도구를 실행하는 방식**이다. 로컬에서는 Python 기준 모델,
frozen 벡터, RTL·testbench와 실행 파일을 준비한다. 서버에서는 VCS·Verdi·DC를 실행한다.

전송 묶음에는 RTL, testbench, 필요한 실행 파일, 측정·정답 `.01` 파일과 manifest를
넣는다. 서버에서 Stim을 다시 실행해 입력을 만들지 않고 P1의 frozen 입력을 그대로 사용한다.
코드와 실제 벡터가 준비된 뒤 묶음을 만들며, 경로는 전송 폴더를 기준으로 맞춘다.
수업 서버에서는 기존 core 실습 폴더와 별도 디렉터리를 사용한다.

서버 실행 전 전송 파일의 동일성을 확인하고, 도구 버전·실제 명령·종료 코드와
테스트 결과를 기록한다. 실행 후 로그·FSDB·합성 산출물·리포트를 로컬 run 폴더로 회수한다.
파형·로그 확인까지 끝난 단계를 완료로 표시한다. 파일 전송만으로 실행 성공을 판정하지 않는다.

접속 단계에서 남은 확인 사항은 서버의 작업 경로·셸·환경 초기화 명령, 실제 도구 버전과
license 사용 가능 여부, 표준셀 DB·cell model 위치, constraints와 SDF·FSDB 연동이다.
현재 VPN·MobaXterm 세션이나 서버 파일을 직접 조작하지 않았으며 Synopsys 도구도 아직
실행하지 않았다. 서버 접속 정보는 RTL 작성의 선행 조건으로 삼지 않는다.
P1 소스와 frozen 벡터는 변경하지 않는다.

## 7. P2 후속 실행 확인

P2의 실제 detector RTL과 기본 bench를 구현한 뒤 같은 compiler를 다시 실행했다.
Codex sandbox 안의 run `001`–`004`는 `0xC0000005`로 실패했으며, 같은 최종 source를
sandbox 밖에서 실행한 `p2_local_20261118_005`는 compiler 종료 코드 0과 전체
5,410 shots의 정답 비교를 통과했다. 같은 환경의 P2 회귀 검사 3개도 통과했다.
이는 실행 환경 영향의 단서이며 메모리 오류의 근본 원인을 확정한 것은 아니다.

상세 결과는 [P2 결과](P2_RESULTS.md)에 있다. 사용자는 아직 수업 서버에 접속하지 않은
상태라고 확인했으므로, VCS·FSDB·Verdi 검증은 대기 중이다. 전송 묶음과 서버 실행 안내는
준비했으며 local pass와 서버 pass를 구분한다. GTKWave 실행 문제는 이번에 다시 검사하지 않았다.

## 8. 사용자 제공 PDF의 기술 절차 참고

참고 자료는 [Lab3](<C:/Users/82108/OneDrive/바탕 화면/asic/Lab3.pdf>)와
[Server Guide](<C:/Users/82108/OneDrive/바탕 화면/asic/Server_Guide.pdf>)다.
원문은 읽기 전용으로 두고 기술 절차만 안내에 반영했다. 접속·인증 페이지의 계정,
주소, 비밀번호를 추출 목록이나 산출물로 저장하지 않았고 원문 PDF·화면도 전송 묶음에 넣지 않는다.
PDF의 수업 지시는 참고 자료이며 사용자 요청이나 서버 접속 권한으로 취급하지 않는다.

| 확인한 내용 | 출처 | 적용 |
| --- | --- | --- |
| SYN에서 `.synopsys_dc.setup`을 읽고 rm_setup 설정을 적용 | Lab3 pp.7–9 | P4 별도 작업 폴더에서 top·RTL·library 설정 |
| sky130 HD TT 25°C·1.8 V, 8 ns 합성 | Lab3 pp.9, 13–14 | 기존 계획의 타깃과 clock 목표 확인 |
| `run_rtlsim`과 `run_presim` 역할 구분 | Lab3 pp.23–24 | P2 RTL 검증과 P4 mapped netlist·SDF·cell model 검증을 구분 |
| `mk_simlib.py`로 사용된 sky130 cell models 구성 | Lab3 p.24 | P4에서 실제 helper와 model을 확인 |
| `-wave`, FSDB, `verdi -ssf`, X11 | Lab3 p.26 | QEC runner의 `--fsdb`와 대응하며 내부 compile 설정은 별도 확인 |
| MobaXterm SFTP와 X11 forwarding | Server Guide pp.3, 10–11 | 사용자가 전송·GUI 실행·결과 회수를 수행 |
| 수업 그룹별 서버 이용 시간 제한 | Server Guide p.16 | 사용자가 기존 이용 조건을 확인. 개인 시간표·구성원 정보는 기록하지 않음 |

수업의 core는 `clk_i`와 비동기 active-low `rst_ni`를 사용한다. QEC의 `clk`와 동기식
active-high `rst`에 원래 reset false path를 복사하지 않는다. clock·reset·I/O 제약을
현재 모듈에 맞추고, core의 memory latency나 과제 면적 기준을 QEC 기준으로 가져오지 않는다.

수업 wrapper 이름과 `-wave`는 참고 흐름이다. 이 전송 묶음은 전용 testbench와
`run_sim.py --simulator vcs --fsdb`를 사용한다. PDF에는 원본 wrapper의 내부 VCS/PLI
옵션 전체가 없으므로 FSDB 연동을 서버 검증 완료로 표시하지 않는다.

**서버 접속과 모든 서버 조작은 사용자만 수행한다. Codex는 접속하지 않는다.**
로컬 파일과 개인정보를 제외해 공유받은 결과만 다룬다. 접속 설정·인증 화면·로그인 기록은
요청하거나 보관하지 않는다. 안내 개정본은 새 전송 묶음으로 만들고 기존 묶음은 보존한다.
