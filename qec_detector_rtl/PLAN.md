# 측정 기록 → detection event RTL 블록 개발 계획

- 작성일: 2026-11-18, Asia/Seoul
- 상태: P1 완료. P2 RTL·기본 testbench 로컬 검증 및 서버 전송 묶음 준비 완료. 서버 VCS·Verdi 확인과 P3·P4는 미완료다.
- P1 근거: [결과 문서](P1_RESULTS.md). 회귀 검사 5개 통과, 5,410개 기록에서 Stim 대비 불일치 0.
- P2 근거: [로컬 결과](P2_RESULTS.md). RTL 불일치 0, P2 테스트 3개 통과. 서버 접속 대기 중이다.
- 환경 점검: 작은 RTL compile/run과 Stim 실행 확인. compiler 비정상 종료·GTKWave 실행 문제는 [환경 점검 결과](ENVIRONMENT_CHECK.md)에 기록한다.
- 주 EDA 흐름: VCS 기능 검증, Verdi FSDB 확인, Design Compiler 8 ns 합성 및 VCS SDF 검증. 사용자가 수업 도구의 사용 가능 여부를 확인했다.
- 실행 방식: 로컬 구현·벡터 준비 후 VPN·MobaXterm으로 수업 서버에 접속해 파일을 전송하고 EDA 도구를 실행한다. 결과는 로컬 run 폴더에 회수한다.
- 작업 분담: 서버 접속·전송·서버 명령 실행은 사용자만 수행한다. Codex는 서버에 접속하지 않고 로컬 준비와 공유받은 결과 분석을 담당한다. 개인정보는 수집·기록하지 않는다.
- 다음 서버 작업: 2026-11-20 금요일, 사용자의 수업 허용 시간. 새 설치 없이 진행하며 [이어하기 순서](FRIDAY_CHECKLIST.md)를 따른다.
- 작업 공간: `C:/Users/82108/fp/qec_detector_rtl/`
- 생성물: `C:/Users/82108/fp/artifacts/qec_detector_rtl/<run_id>/`
- 첫 목표: 고정된 `d=3`, 3라운드 회로에서 경계 조건을 포함한 detector 변환을 구현하고 기능, cycle 지연, 처리 간격, 합성 자원을 검증한다.

## 1. 연구와의 연결 및 주장 범위

기존 연구는 neural pre-decoder가 제안한 후보의 품질과 실제 후보 선택의 효과를 구분했다.
완성본은 독립 strong-ZZ 및 편향 잡음 평가에서 선택 정책의 추가 정확도 향상을 확립하지
못했다고 정리한다. 온라인 지연과 RTL 면적·지연도 미측정으로 남아 있다.
이번 작업은 측정값을 디코더 입력으로 바꾸는 디지털 처리 흐름을 직접 구현하는 후속 실습이다.
기존 선택기를 개선하거나 neural 모델을 하드웨어로 옮기는 단계는 포함하지 않는다.
근거는 [완성본 논문](../artifacts/Candidate_Quality_and_Logical_Transition_Selection_in_Neural_Pre_Decoding_for_Surface_Codes.pdf)의
VII-C, VII-D 및 VIII와 [연구 인계 문서](../docs/thesis/PREDECODER_CLEAN_HANDOFF.md)다.

실시간 FPGA 실험에서는 전체 응답 시간에 decoding뿐 아니라 측정값 전달, 재정렬,
패킷 구성과 제어가 포함된다. 인용 논문은 9 decoding rounds에서 전체 9.6 μs 중
3.1 μs를 통신·제어 지연으로 보고하고, 측정값 재정렬·패킷 구성에 약 400 ns까지
소요될 수 있어 전용 하드웨어로 옮기는 방향을 제안한다. 이번 블록은 그 처리 흐름의
한 부분과 연결된다. 해당 실험은 stability-8 구성이므로 이번 surface-code memory와
시간값을 직접 비교하지 않는다. [실시간 FPGA 오류정정 실험](https://www.nature.com/articles/s41467-026-73331-6)

본 실습에서 검증할 주장은 다음 세 가지다.

1. 같은 측정 기록에 대해 Stim, 독립 Python XOR 구현, RTL의 detector 출력이 비트 단위로 일치한다.
2. 정해진 입출력 규약에서 첫·마지막 경계, 입력 공백, 출력 정체, 연속 shot과 reset을 처리한다.
3. 명시한 타깃과 클록 제약에서 cycle 지연·처리 간격과 합성 timing·자원 사용량을 보고한다.

decoder 정확도 향상, 실물 QPU 연결, 전체 feedback latency, 실시간 QEC 달성은
이 실습의 검증 범위 밖이다. 기존 논문의 Bell-pair 회로 대신 단일 memory 회로로
시작하므로, 논문의 논리 프레임 출력을 재현하는 실험으로 해석하지 않는다.

## 2. 첫 회로와 고정 조건

### 2.1 재사용할 코드

| 기존 코드 | 재사용 내용 |
| --- | --- |
| `config.py`의 `CircuitConfig` | `distance=3`, `rounds=3`, `basis="z"`, `variant="stim_rotated"` |
| `circuits.py`의 `build_memory_circuit` | 이상적인 Stim rotated memory 회로 생성 |
| `config.py`의 `SI1000Config` | 최초 잡음 벡터의 설정과 실제 파생 오류 확률 |
| `noise_si1000.py`의 `apply_si1000_noise` | 측정 순서를 유지하는 잡음 회로 생성 |
| Stim `compile_sampler()` | 원시 측정 기록 생성 |
| Stim `compile_m2d_converter()` | 같은 기록의 정답 detection event 계산 |

`sample_dataset.py`의 기존 데이터셋 schema를 변경하지 않는다. 본 실습에서 필요한
측정 기록과 detector 벡터는 전용 생성기로 저장한다. `xzzx` 경로는 현재 같은
seed 회로를 재사용하므로 첫 회로의 이름이나 물리적 의미로 사용하지 않는다.

### 2.2 계획 작성 시 확인한 회로 규모

아래 값은 2026-11-18에 기존 builder와 설치된 Stim 1.15.0으로 확인했다.
완성된 Python 모델이나 RTL을 검증한 결과는 아니다. 구현 단계에서 회로 파일과 hash를
저장하고 다시 확인한 값을 기준 명세로 고정한다.

| 항목 | 값 |
| --- | --- |
| 회로 종류 | `surface_code:rotated_memory_z` |
| distance / 안정자 측정 rounds | 3 / 3 |
| data qubits / check ancillas | 9 / 8 |
| 전체 측정값 | 33비트 = 8 × 3 + 9 |
| detector | 24비트 = 4 + 8 + 8 + 4 |
| logical observable | 1개. 첫 RTL의 출력에는 포함하지 않음 |
| detector의 무잡음 기준 패리티 | 현재 회로에서는 24개 모두 0 |

33개 측정은 Stim 기록 순서로 `m[0]`부터 `m[32]`까지 번호를 붙인다.
큐비트 번호와 측정 인덱스는 다르다. 각 측정의 큐비트 번호, 명령, round를 별도로
저장해 순서 착오를 확인할 수 있게 한다.

| 구간 | 측정 인덱스, 양 끝 포함 | detector 인덱스 | 경계 처리 |
| --- | --- | --- | --- |
| 첫 round | 0–7 | D0–D3 | 초기 Z 상태에서 결정되는 Z-check만 정의됨 |
| 두 번째 round | 8–15 | D4–D11 | 대응하는 첫 round 측정과 XOR |
| 세 번째 round | 16–23 | D12–D19 | 대응하는 두 번째 round 측정과 XOR |
| 마지막 data 측정 | 24–32 | D20–D23 | data 측정 패리티와 마지막 Z-check 측정을 XOR |

무잡음이어도 원시 측정 기록은 항상 0이 아니다. 특히 초기 상태에서 결정되지 않는
X-check의 개별 측정값은 무작위일 수 있다. 모든 첫 round 측정에 detector를 만들거나
마지막 data 측정을 일반적인 이전 round 비교로 처리하지 않는다.

## 3. Python 기준 모델과 Stim 정답

### 3.1 변환식

각 detector j의 측정 인덱스 집합을 S_j라 하면 다음 식을 구현한다.

```text
d[j] = XOR(m[i] for i in S_j) XOR b[j]
b[j] = XOR(reference_sample[i] for i in S_j)
```

`b[j]`는 오류 없는 기준 기록에서 기대하는 패리티다. 이번 회로에서는 모두 0으로
확인됐지만, 구현은 기준값을 계산하고 검사한다. 기본 Stim 변환기는 무잡음 기준 기록을
사용하므로 `skip_reference_sample=True`를 근거 없이 설정하지 않는다.
API는 [Stim 공식 문서](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md)를
참고하되, 개발 버전 문서의 신규 API를 그대로 가정하지 않고 고정한 설치 버전의
docstring과 동작을 확인한다.

### 3.2 측정 참조 추출

1. 회로의 `flattened()` 결과를 순회해 `REPEAT`와 좌표 이동을 펼친다.
2. 지금까지 발생한 측정 수를 `n_seen`으로 센다. 측정 명령별 비트 수를 반영한다.
3. `DETECTOR`를 만나면 `rec[-k]`를 절대 인덱스 `n_seen - k`로 바꾼다.
4. 각 인덱스가 이미 정의된 측정 범위 안에 있는지 검사한다. 참조 중복은 XOR의 홀짝으로 처리한다.
5. detector 등장 순서대로 ID를 붙이고 참조 인덱스, 좌표, `b[j]`, `required_count=max(S_j)+1`을 저장한다.
6. `OBSERVABLE_INCLUDE`는 detector 목록에 넣지 않는다. 첫 버전에서 지원하지 않는 참조 형식은 명시적으로 실패시킨다.

고정한 첫 회로에 필요한 추출만 구현한다. 임의 Stim 프로그램의 실행기나 범용 RTL 생성기를
만들지 않는다. Python XOR 계산은 Stim 변환 결과를 복사하지 않고 추출한 참조로 직접 계산한다.

### 3.3 동일한 shot의 정답을 저장하는 순서

```python
measurements = circuit.compile_sampler(seed=seed).sample(shots=shots)
expected = circuit.compile_m2d_converter().convert(
    measurements=measurements,
    append_observables=False,
)
```

`compile_detector_sampler()`를 별도로 실행한 결과를 위 `measurements`의 정답으로
사용하지 않는다. 독립 sampling으로 얻은 두 배열은 동일한 shot의 짝이 아니다.
seed만 같게 설정해도 서로 다른 sampler의 기록이 대응한다고 가정하지 않는다.

잡음 회로도 그 회로에서 생성한 측정 기록을 그 회로의 변환기에 넣는다. 이상 회로와 잡음
회로의 측정 순서, detector 참조, 기준 패리티가 같은지도 비교한다.
계획 작성 시 기존 SI1000 주입 후 회로를 `without_noise().flattened()`로 비교했을 때
이상 회로의 `flattened()`와 같았다. 펼치기 전의 문자열 또는 구조 비교만으로 판정하지 않는다.

### 3.4 최초 검증 벡터

| 묶음 | 계획한 규모 | 목적 |
| --- | --- | --- |
| `noiseless` | 256 shots | 측정의 무작위성과 별개로 detector가 모두 0인지 확인 |
| `single_flip` | 기준 기록 1개 + 각 위치 단일 반전 33개 | 각 입력 비트의 detector 연결과 경계 검사 |
| `random_bits` | 1,024개의 33비트 기록 | 물리적 sampling 분포 밖에서도 XOR 배선 검사 |
| `si1000` | p=0.001, 4,096 shots | 기존 회로 잡음 경로에서 실제 측정 기록 처리 |

`single_flip`과 `random_bits`는 변환기의 디지털 기능 검사용이다. 물리적으로 생성되는
오류 분포나 logical error rate의 근거로 사용하지 않는다. 단일 반전 검사는 affine XOR
변환의 각 입력 연결을 확인하지만, RTL의 시간 순서·상태 검증을 대신하지 않는다.

벡터 생성 seed는 `noiseless=1101801`, `random_bits=1101802`, `si1000=1101803`을
초기 제안값으로 사용한다. `single_flip`은 저장한 `reference_sample()`에서 만든다.
프로토콜 지연용 난수 seed는 데이터 seed와 별도로 저장한다. 버전과 파일 hash를 함께
기록하며 seed만으로 다른 Stim 버전·환경까지 비트 재현성을 보장하지 않는다.

기존 편향 잡음 기록은 위 네 묶음이 통과한 뒤 선택적으로 추가한다. 측정 순서와 detector
정의가 같다면 동일한 RTL을 사용한다. 첫 milestone에 편향 잡음 모델의 확장이나 재학습을
끼워 넣지 않는다.

## 4. 첫 RTL 명세

### 4.1 구조

합성 가능한 Verilog 모듈 하나로 시작한다. 33비트 register에 측정을 순서대로 저장하고,
다음 detector에 필요한 기록이 모두 저장되면 고정 XOR 식을 계산한다. 출력은 Stim ID
순서인 D0부터 D23까지 한 비트씩 보낸다. 입력 수집과 이미 계산 가능한 detector 출력은
같은 shot 안에서 겹칠 수 있다.

```text
직렬 측정 입력 → 33비트 기록 register → 고정 XOR 선택 → 출력 register → detector 입력 소비자
                     ↑                     ↑
               수락한 측정 수        다음 detector ID / 참조 가용 조건
```

전체 기록을 저장해도 작은 회로이므로 첫 버전에서는 circular buffer, 이전 round만 저장하는
최적화, 다중 shot buffer, AXI/WISHBONE 등을 추가하지 않는다. 최대 5개 측정값을 XOR하는
고정 `case`로 시작하고, 검증된 참조 표와 식이 같은지 검사한다.

### 4.2 포트와 전송 규약

| 신호 | 방향 | 폭 | 의미 |
| --- | --- | --- | --- |
| `clk` | 입력 | 1 | 모든 상태는 posedge에서 변경 |
| `rst` | 입력 | 1 | 동기식 active-high reset, 다른 상태 갱신보다 우선 |
| `in_valid` | 입력 | 1 | 입력 측정 비트가 유효함 |
| `in_ready` | 출력 | 1 | 현재 shot의 다음 측정을 받을 수 있음 |
| `in_bit` | 입력 | 1 | 현재 shot의 다음 측정값 |
| `out_valid` | 출력 | 1 | detector 출력 register가 유효함 |
| `out_ready` | 입력 | 1 | 소비자가 detector를 받을 수 있음 |
| `out_bit` | 출력 | 1 | detection event 값 |
| `out_index` | 출력 | 5 | detector ID 0–23 |
| `out_last` | 출력 | 1 | 현재 유효 출력이 D23임 |

전송은 posedge 직전의 `valid && ready && !rst`가 참일 때만 성립한다.
입력 source는 `in_valid && !in_ready` 동안 `in_valid`와 `in_bit`를 유지한다.
source는 ready와 무관하게 valid를 제시할 수 있어야 하며 ready를 기다리느라
valid를 영원히 올리지 않는 방식으로 구성하지 않는다.

reset 후 첫 수락 비트가 `m[0]`이고, 33번째가 `m[32]`다. 별도 start 포트 없이
고정 길이로 shot을 구분한다. D23의 출력 전송 완료가 shot 종료다.
첫 버전은 동시에 한 shot만 보유한다. D23이 소비되기 전에는 다음 shot의 첫 비트를
수락하지 않는다. 입력 33개를 모두 받으면 `in_ready=0`으로 두고 출력 완료를 기다린다.
마지막 출력 전송 edge에서 상태를 초기화하며 다음 edge부터 새 shot을 수락할 수 있다.

reset이 1이면 전송을 계산하지 않고 `in_ready`는 0으로 둔다. reset posedge에서
출력 valid와 진행 상태를 지운다. 미완료 shot과 아직 소비되지 않은 출력은 폐기하고,
source도 reset 해제 후 새 shot의 `m[0]`부터 다시 보낸다. 이전 shot을 이어 보내지 않는다.

고정 길이 인터페이스에는 누락된 입력 비트나 외부 순서 오류를 스스로 발견하는 기능이 없다.
driver와 파일 reader에서 길이·순서·0/1 값을 검사한다. 실제 계측 장치의 재정렬이나
중간부터 시작한 프레임의 재동기화가 필요하면 후속 버전에서 framing을 확장한다.

### 4.3 진행 상태와 출력 대기

- 측정 수락 수는 0–33을 표현하는 6비트 값으로 관리한다. 입력 전송 때만 해당 위치를 쓰고 증가시킨다.
- 다음 detector ID는 0–24를 표현한다. 24는 24개 출력을 모두 register에 올렸다는 종료 상태다.
- 다음 detector의 `required_count` 이하가 아니라 **이상**의 측정 수가 저장돼야 계산한다.
- 출력 register가 비었거나 기존 출력이 이번 edge에서 소비될 때만 새 결과를 올린다.
- 출력 대기 시 `out_valid`, `out_bit`, `out_index`, `out_last`를 유지한다. 입력은 같은 shot의 33개까지 계속 받을 수 있다.
- 기존 출력을 소비하면서 다음 결과를 채울 수 있어, 준비된 detector는 매 cycle 전송할 수 있다.
- 새 입력 비트는 그 다음 edge의 XOR 계산부터 사용한다. 같은 edge의 입력을 forwarding하는 경로는 두지 않는다.
- shot 종료 시 기록은 지우거나 유효 수를 초기화한다. 어느 방식을 택하든 새 shot에서 아직 쓰지 않은 기록은 참조하지 못해야 한다.

검증용 `out_index`와 `out_last`는 출력 비트와 같은 전송에 속한다. 출력 값이 0인
detector도 생략하지 않는다. 출력이 준비되는 순서와 ID 순서가 다를 때에는 ID 순서를
우선한다. 예를 들어 D1의 입력이 먼저 준비돼도 D0 출력 전에 D1을 내보내지 않는다.
이로 인한 대기 시간도 해당 구조의 지연에 포함한다.

측정 이력의 물리적 초기화 여부, counter 갱신과 출력 refill의 동시 동작은 실제 RTL에서
하나의 명확한 우선순위로 구현한다. 첫 버전에서 RAM 추론이나 면적 최소화보다 기능과
파형 설명을 우선한다.

## 5. 경계 조건을 포함한 참조 표

다음은 Stim 1.15.0의 현재 회로에서 확인한 참조다. 모든 인덱스는 0부터 시작한다.
`필요 측정 수`는 데이터 의존성이 충족되는 시점이며 출력 전송 cycle 자체는 아니다.
모든 `b[j]`는 0이다. 구현 단계에서 추출한 JSON과 이 표가 다르면 회로·버전 차이를
확인하고 명세를 갱신한 뒤 RTL을 변경한다. 출력만 맞추려고 임의로 순서를 바꾸지 않는다.

| detector | XOR하는 측정 인덱스 S_j | 필요 측정 수 |
| --- | --- | --- |
| D0 | 4 | 5 |
| D1 | 1 | 2 |
| D2 | 6 | 7 |
| D3 | 3 | 4 |
| D4 | 0, 8 | 9 |
| D5 | 1, 9 | 10 |
| D6 | 2, 10 | 11 |
| D7 | 3, 11 | 12 |
| D8 | 4, 12 | 13 |
| D9 | 5, 13 | 14 |
| D10 | 6, 14 | 15 |
| D11 | 7, 15 | 16 |
| D12 | 8, 16 | 17 |
| D13 | 9, 17 | 18 |
| D14 | 10, 18 | 19 |
| D15 | 11, 19 | 20 |
| D16 | 12, 20 | 21 |
| D17 | 13, 21 | 22 |
| D18 | 14, 22 | 23 |
| D19 | 15, 23 | 24 |
| D20 | 20, 27, 30 | 31 |
| D21 | 17, 24, 25, 27, 28 | 29 |
| D22 | 22, 28, 29, 31, 32 | 33 |
| D23 | 19, 26, 29 | 30 |

최소한 파형으로 설명할 식은 `D0=m[4]`, `D4=m[0] XOR m[8]`,
`D20=m[20] XOR m[27] XOR m[30]`,
`D22=m[22] XOR m[28] XOR m[29] XOR m[31] XOR m[32]`다.
최종 보고서에서는 해당 입력이 수락된 edge와 출력 valid·전송 edge를 함께 표시한다.

## 6. 파일과 생성물 관리

### 6.1 소스 작업 공간

```text
qec_detector_rtl/
  README.md                         # 현재 작업 안내
  PLAN.md                           # 이 문서
  model/
    generate_vectors.py             # 구현 완료: 참조 추출, 직접 XOR, Stim 정답과 벡터 저장
  rtl/
    measurement_to_detection.v      # 구현 완료: 고정 회로 RTL
  tb/
    tb_measurement_to_detection.sv   # 구현 완료: 기본 입력, handshake, scoreboard
  scripts/
    run_sim.py                      # 구현 완료: Icarus/VCS 실행과 출력 비교
    run_vcs.sh                      # 준비 완료: 서버 Python 설치 없이 VCS 실행
    prepare_server_bundle.py        # 구현 완료: frozen 입력·source 전송 ZIP
    <DC 합성 스크립트>              # P4에서 수업 setup과 library 경로 확인 후 추가

tests/
  test_measurement_to_detection.py   # 구현 완료: 기준 모델 및 경계 회귀 검사
  test_detector_rtl.py               # 구현 완료: 기본 RTL·전송 묶음 검사
```

첫 RTL은 Verilog를 사용한다. testbench만 simulator가 지원하는 SystemVerilog 문법을
사용할 수 있다. Python은 표준 라이브러리, 기존 NumPy·Stim을 사용하며 새 패키지를
추가할 필요가 생기기 전까지 별도 dependency를 도입하지 않는다.

### 6.2 run별 생성물

```text
artifacts/qec_detector_rtl/<run_id>/
  vectors/
    ideal.stim / si1000.stim
    reference.01
    measurement_map.json / detector_map.json
    manifest.json
    <case>.measurements.01 / <case>.expected_detectors.01
  sim/
    simulator 실행 파일, 로그, 실제 detector 기록, cycle 기록
    필요한 대표 파형 또는 실패 파형
    verification.json
  synth/
    실행한 constraints, tool 로그, netlist, timing 및 자원 보고서
    summary.json
  report/
    검증 결과 및 한계를 설명하는 Markdown 보고서와 파형 이미지
```

`run_id` 예시는 `d3_r3_z_v1_20261118_001`이다. 새 실행 폴더를 만들고 동일 경로가
이미 있으면 덮어쓰지 않고 실패시킨다. frozen 벡터를 재사용하는 시뮬레이션은 원본을
읽고 별도 실행 폴더에 결과와 원본 manifest 경로·hash를 남긴다.

`.01`은 첫 비트가 인덱스 0인 ASCII 0/1 문자열로, 측정은 shot당 33자, detector는
shot당 24자다. 파일에는 한 줄에 한 shot을 저장한다. Verilog 문자열 처리나 packed
vector의 방향 때문에 첫 문자가 최상위 비트로 뒤집히지 않도록 파일 reader에서 명시적으로
대응시킨다. `$readmemb`를 사용하면 폭·방향·leading zero를 별도로 검증한다.

`manifest.json`에는 schema version, run ID, 회로 설정, 잡음 설정과 resolved rates,
shot 수, 데이터·프로토콜 seed, Python·NumPy·Stim·도구 버전, 소스 파일 SHA-256,
회로·map·벡터 파일 SHA-256, bit 순서와 detector/observable 구분을 기록한다.
Git commit만으로 소스를 식별하지 않는다. dirty working tree에서는 실제 사용한 파일의
내용도 함께 식별해야 한다. schema를 바꾸면 reader와 testbench도 함께 갱신한다.

## 7. 단계별 실행과 완료 조건

### P0. 환경과 명세 고정

현재 확인한 항목은 `CircuitConfig` 및 builder 사용 가능, Stim 1.15.0,
33개 측정·24개 detector와 위 참조 표, SI1000 주입 후 펼친 무잡음 회로의 일치다.
`C:/Users/82108/anaconda3/envs/fp/python.exe`에서 확인했으며 기본 `python`은
다른 환경을 가리킬 수 있다. 후속 환경 점검에서는 Icarus Verilog 12.0 (devel)의
compile/run, 작은 parity RTL의 32개 입력 비교, reset·출력 정체·파일 입출력·VCD 저장과
의도한 오류의 종료 코드 1을 확인했다. 이는 실제 detector 변환기의 검증 결과가 아니다.

초기 검사에서 compiler의 `0xC0000005` 비정상 종료가 발생했고, 최종 3회 compile은
정상 종료했다. 크래시 원인은 미확정이며 실패 종료 코드를 무시하지 않는다.
GTKWave는 WinError 623 DLL 재할당 오류로 실행하지 못했다. 상세 로그와 후속 처리는
[환경 점검 결과](ENVIRONMENT_CHECK.md)에 보관한다. 이 기록은 Windows 보조 환경의 상태다.

사용자 확인에 따라 주 도구는 VCS·Verdi·Design Compiler로 정한다. 로컬 Lab3 자료에서
sky130 HD, `tt_025C_1v80` 합성 환경과 DC S-2021.06-SP4 리포트를 확인했다.
첫 합성 목표는 8 ns다. 실행 환경은 VPN·MobaXterm으로 접속하는 수업 서버다.
서버 작업 경로와 환경 초기화, 현재 도구 버전·license 사용 가능 여부, 표준셀 DB와
simulation model 경로, constraints와 FSDB 연동은 접속 단계에서 확인한다.
세부 근거는 [Synopsys 수업 환경](ENVIRONMENT_CHECK.md#6-synopsys-수업-환경-반영)에 기록한다.
매 run에 실제 버전과 종료 코드를 남긴다.

**완료 조건:** 회로 설정과 bit 순서 고정, 사용 Python 및 simulator 확인,
합성 타깃 선정 또는 미확정 상태의 명시. 합성 타깃 미확정이어도 P1–P3은 진행한다.

### P1. Python 기준 모델 및 벡터 생성

**2026-11-18 완료:** 아래 작업을 구현하고 검사했다. frozen run은
`d3_r3_z_v1_20261118_001`이며, 저장 파일을 다시 읽어 129,840 detector bits의
Stim·직접 XOR 일치와 source·생성물 hash를 확인했다. 상세 근거는 [P1 결과](P1_RESULTS.md)에 기록한다.

1. `model/generate_vectors.py`에서 기존 builder를 호출한다.
2. 측정·detector map과 기준 패리티를 추출하고 각 입력의 유효성을 검사한다.
3. 네 벡터 묶음을 생성하고 Stim 정답과 직접 XOR 결과를 비교한다.
4. 무잡음 detector 전부 0, 길이 33/24, D0–D23 순서와 두 경계식을 검사한다.
5. `tests/test_measurement_to_detection.py`에 작은 재현 검사를 추가한다.
6. 작은 합성 회로 `X 0; M 0; DETECTOR rec[-1]`로 기대 패리티가 1인 경우도 검사해, 기준 패리티를 무시한 구현을 잡는다.
7. 벡터 파일을 다시 읽고 bit 순서, 줄 수, hash를 확인한 뒤 frozen 입력으로 보관한다.

**완료 조건:** 모든 묶음에서 Stim 대비 mismatch 0, 무잡음 출력 전부 0,
경계·단일 반전·비영 기준 패리티 검사가 통과하고 manifest가 생성됨.
비영 기준 패리티 회로는 Python 추출기의 보조 검사이며 첫 RTL의 대상 회로는 그대로다.

### P2. 고정 회로 RTL과 기본 testbench

**2026-11-18 로컬 준비 완료:** 기본 RTL과 testbench를 구현하고 Icarus에서
5,410개 기록, 129,840 detector bits의 정답 일치를 확인했다. P2 테스트 3개가 통과했다.
정상 실행은 sandbox 밖에서 수행했다. 전송 ZIP과 실행 절차는 준비했으며,
사용자가 아직 서버에 접속하지 않은 상태여서 VCS·Verdi 검증은 미완료다.
근거와 남은 작업은 [P2 결과](P2_RESULTS.md), [서버 실행 안내](SERVER_RUN.md)에 기록한다.

주 simulator는 VCS를 사용한다. Python/Stim 정답과 동일한 frozen 벡터를 읽는 testbench를
구성하고, 수업 환경의 FSDB 연동을 적용해 Verdi에서 파형을 확인한다.
로컬에서 RTL·testbench와 실행 파일을 작성한 뒤, P1 frozen 벡터와 manifest를 포함한
전송 묶음을 준비한다. 수업 서버의 별도 작업 폴더에서 실행하고 결과를 회수한다.
서버에 Stim을 새로 설치하거나 벡터를 재생성하는 것을 요구하지 않는다.

1. 4절의 포트·상태·우선순위에 따라 모듈을 작성한다.
2. 추출한 24개 XOR 식과 `required_count`를 고정한다.
3. source와 sink가 항상 준비된 조건에서 네 벡터 묶음을 순서대로 실행한다.
4. scoreboard는 출력 전송 시에만 값을 수집해 shot ID, detector ID, bit, `out_last`를 비교한다.
5. 한 shot의 수락 입력 33개, 소비 출력 24개와 다음 shot 경계를 확인한다.
6. 불일치·X/Z 출력·조기 종료·추가 출력·timeout은 simulation 실패로 처리한다.

**완료 조건:** 모든 기본 벡터에서 Python·Stim·RTL mismatch 0,
detector 누락·중복·순서 오류 0, 대표 경계 파형을 설명할 수 있음.

### P3. 입출력 규약과 cycle 검증

| 검사 | 입력/출력 조건 | 필수 확인 |
| --- | --- | --- |
| 입력 공백 | 첫 round, round 사이, 마지막 data 측정 직전·중간에 valid 공백 | 수락한 입력만 인덱스가 증가하고 조기 출력하지 않음 |
| 출력 정체 | D0, 중간 detector, D23에서 ready를 여러 cycle 낮춤 | valid·값·ID·last 유지, 전송 후 중복 없음 |
| 연속 shot | source는 다음 shot을 즉시 제시 | 이전 D23 소비 전 수락 없음, 다음 shot의 첫 detector부터 일치 |
| 동시 동작 | 입력 수락과 이전 detector 전송을 같은 edge에 발생 | 각 count와 출력 refill이 한 번씩만 갱신됨 |
| reset | 입력 중, 출력 대기 중, shot 종료 부근에 reset | 미완료 shot 폐기 후 새 shot이 독립적으로 일치 |
| 미완료 입력 | 32개까지만 보내고 대기 | D22·D23 최종 완료나 shot 종료가 발생하지 않음 |
| 임의 지연 | 256 random-bit shots에 고정 seed 입력 공백·출력 정체 | 모든 전송의 값·개수·순서와 복구가 일치 |

미완료 입력에서는 이미 준비된 앞쪽 detector가 출력될 수 있다. 그 자체를 실패로
판정하지 않는다. 입력 33개 없이 전체 shot 완료가 발생하면 실패다.

reset 이후에는 scoreboard도 미완료 shot을 취소한다. 취소된 출력과 정상 완료 shot의
정답을 섞지 않는다. driver는 negedge에서 입력을 바꾸고 posedge의 전송을 관측하는 등
race를 피한다. 출력 register 갱신 후 valid 관측과 edge에서의 전송을 구분한다.
프로토콜 검사에는 0/1 값만 사용하고, 유효 출력의 X/Z는 별도 실패로 검출한다.

무정체 구간과 정체 구간을 다른 결과로 보고한다. bounded test에는 충분한 watchdog을
두며, sink가 ready를 영원히 올리지 않는 경우를 유한 완료 지연의 검증 대상으로 삼지 않는다.
첫 버전의 제어는 측정 bit 값에 의존하지 않으므로 동일한 valid/ready 패턴에서
무잡음·잡음 기록의 전송 일정도 같아야 한다.

**완료 조건:** 모든 directed 검사와 고정 seed 지연 검사 통과,
no-stall cycle 일정과 정체 후 복구 확인, 측정 기준과 단위가 기록됨.

### P4. 타깃을 지정한 합성 및 timing·자원 확인

Design Compiler와 수업의 sky130 HD 표준셀 환경을 재사용한다. 현재 확인한 corner는
`tt_025C_1v80`이며, 실제 실행 호스트의 DB·simulation model·설정을 검증한 뒤 고정한다.
첫 clock 목표는 **8 ns, 125 MHz**다. 기존 10 ns 제안은 이 조건으로 대체한다.
이는 새 블록에서 달성한 주파수가 아니다.

별도 작업 폴더에서 `dc_setup.tcl`의 RTL 목록과 top을 QEC 블록으로 지정한다.
합성 전에 clock/reset 처리, 입출력 delay, output load, clock uncertainty와
사용 corner·library를 기록한다. core용 constraints를 그대로 적용하지 않고 QEC 포트의
I/O budget을 검토한다. 가정한 값은 보고서에 밝히며 결과를 본 뒤 조용히 바꾸지 않는다.
경로를 false path로 덮어 timing을 통과시키지 않는다.
Lab3 pp.13–14의 reset false path는 비동기 active-low `rst_ni`에 대한 수업 설정이다.
현재 동기식 active-high `rst`에 그대로 복사하지 않고 reset timing을 검사한다.
clock port도 `clk_i`에서 현재 `clk`에 맞춘다. core의 외부 memory latency를 자동으로 재사용하지 않는다.

검토 항목은 inferred latch, 잘못된 bit 폭·범위, 다중 driver, 제거된 top 출력,
unconstrained path, timing violation, 예상치 못한 RAM 또는 큰 mux, critical path다.
warning은 이유를 분류하고 기능·timing에 영향을 주는 항목은 해결한다.
전체 메모리 초기화나 가변 인덱스 쓰기가 자원·timing에 미친 영향도 실제 보고서로 확인한다.

**완료 조건:** 재실행 가능한 스크립트와 constraints, tool 원본 보고서,
명시한 타깃의 자원·critical path·timing slack, 미해결 warning과 제약의 기록.
목표 clock 미충족이면 기능 통과와 timing 미충족을 구분하고, 수정 후 해당 검사를 다시 한다.

합성 산출물은 `measurement_to_detection.mapped.v`, `.mapped.sdf`, `.sdc`와
timing·area 보고서를 run별로 보관한다. power 보고서는 activity 입력 여부와 추정 조건을
표시한 보조 결과로 남길 수 있다. cell area는 die area와 구분하고, power 리포트를
실측 전력으로 해석하지 않는다.

대응하는 cell Verilog model과 SDF를 VCS에 연결해 같은 frozen 입력·정답으로
gate-level 검증을 반복한다. SDF annotation 성공 여부와 timing violation을 확인하고,
Verdi에서 FSDB 파형의 경계·reset·handshake를 사용자가 확인한다. Lab3 pp.23–24의
`pre-sim`은 mapped netlist·SDF·sky130 cell models를 사용하는 합성 후 gate-level 검증이다.
수업 helper `mk_simlib.py`의 실제 파일과 model 구성을 확인하고 전용 QEC bench로 대응시킨다.
보고서에는 RTL과 합성 후 gate-level 검증을 명확히 구분한다.
이 검증까지 P4의 완료 조건에 포함한다.

`-retime`은 baseline 검증 후 필요할 때 별도 run으로 비교하는 후속 항목이다.
적용 시 값·순서뿐 아니라 reset·handshake와 외부 cycle 동작을 다시 검증한다.

라이브러리 없는 generic synthesis는 구조 확인 자료로만 사용한다. 그것만으로 물리적인
ASIC 면적·timing이 검증됐다고 쓰지 않는다. 도구·타깃이 없으면 P4를 미완료로 남긴다.
합성 timing과 배치·배선 후 timing을 구분한다. 실물 보드 동작이나 ASIC sign-off는
첫 완료 조건에 포함하지 않는다.

### P5. 결과 정리와 재현 확인

하나의 완료 run에 회로·map·벡터 hash, 기능 검사 결과, cycle 측정, 합성 조건과
원본 로그를 연결한다. 최종 Markdown 보고서에는 다음을 기록한다.

1. 회로와 입출력 규약, 첫·마지막 경계 XOR 식.
2. 실제 실행한 명령, 버전, 벡터 수와 검사별 통과/실패/미실행 상태.
3. 대표 파형과 cycle 지연·처리 간격, 정체의 영향.
4. 타깃·제약·자원·critical path와 timing slack.
5. 현재 구조의 한계와 실제 QEC 통합에서 추가로 필요한 부분.

**완료 조건:** 같은 frozen 벡터로 기능 검증을 재실행할 수 있고,
결과의 기준점과 제약을 보고서만으로 확인할 수 있음. P4가 미완료이면 전체 완료로 표시하지 않는다.

## 8. 시간 지표의 정의

입력 전송 edge를 `c_in[i]`, detector j에 필요한 마지막 입력 수락 edge를
`a[j]=max(c_in[i] for i in S_j)`라 한다. `v[j]`는 해당 detector가 출력 register에
처음 나타나 valid가 된 edge 이후 시점이고, `o[j]`는 sink가 실제로 수락한 edge다.
정수 edge 번호와 edge 이후 관측이라는 표기를 로그·파형에 함께 사용한다.

| 지표 | 계산 및 해석 |
| --- | --- |
| detector valid 지연 | `v[j] - a[j]`. 필요한 기록 수집 후 출력 register에 결과가 나타날 때까지 |
| detector 전송 지연 | `o[j] - a[j]`. 순서 대기·기존 출력 정체·해당 출력 수락 대기 포함 |
| shot 전체 지연 | `o[23] - c_in[0]`. 첫 입력 수락부터 마지막 출력 전송까지 |
| 마지막 입력 이후 지연 | `o[23] - c_in[32]`. 마지막 data 측정 수락 이후 남은 처리 |
| 입력 처리 간격 | 같은 shot의 연속 `c_in[i+1] - c_in[i]`. source 공백이 없는 조건을 별도 측정 |
| 출력 처리 간격 | 연속 detector 전송 edge의 차이. 입력 부족과 순서 대기를 구분 |
| shot 시작 간격 | 다음 shot의 `c_in[0] - 현재 shot의 c_in[0]` |
| shot 경계 간격 | 다음 shot의 첫 입력 수락 edge - 현재 D23 전송 edge |

예를 들어 필요한 마지막 측정을 edge 10에 수락하면, 명세상 새 기록은 edge 11의
계산에서 사용할 수 있다. 대기 중인 이전 출력이 없을 때 edge 11 직후 결과가 valid가 되고,
sink가 준비돼 있으면 edge 12에서 전송한다. 이 예의 valid 지연은 1 cycle,
전송 지연은 2 cycles다. 이는 명세를 설명하는 예이며 구현 측정값은 아니다.

순서를 기다리는 D1이나 D21 등은 위 예보다 늦을 수 있다. FIFO 깊이나 buffering으로
이 대기를 숨겼다고 가정하지 않는다. 24개 detector 각각의 최소·최대 지연과 shot 지표를
보고하고, 외부 ready 대기에 의한 시간은 별도로 표시한다.

cycle을 시간으로 바꿀 때는 `시간(ns)=cycle 수 × clock period(ns)`를 사용한다.
시뮬레이션의 clock 설정, synthesis의 목표 clock, timing을 통과한 clock을 각각 구분한다.
125 MHz를 testbench에 썼다는 이유만으로 8 ns 합성 timing 통과를 주장하지 않는다.
기능 검증 실행의 Python wall time도 RTL의 하드웨어 지연으로 사용하지 않는다.

이 인터페이스는 측정 bit를 하나씩 직렬로 받는다. 8비트 round 데이터나 실제 장치의
동시 readout과 바로 같은 조건이 아니다. 가정한 입력률에 대해 같은 shot의 입력 처리와
출력 처리, shot 사이 정지를 함께 계산한다. backpressure를 따를 수 없는 계측 장치에
연결하려면 외부 buffer 또는 입력 병렬화가 필요하며, 현재 모듈만으로 연속 acquisition의
backlog가 없다고 주장하지 않는다. 전체 QEC latency에는 readout·전송·decoder·feedback이
추가된다.

## 9. 예정 실행 명령과 검증 정책

다음 Python 기준 모델 명령은 구현했다. 사용할 Python 환경을 활성화한 뒤
저장소 루트에서 실행한다. 생성에는 아직 사용하지 않은 run ID를 지정한다.

```powershell
python -m qec_detector_rtl.model.generate_vectors --out-root artifacts/qec_detector_rtl/d3_r3_z_v1_20261118_002
python -m unittest discover -s tests -p test_measurement_to_detection.py -v
python -m py_compile qec_detector_rtl/model/generate_vectors.py tests/test_measurement_to_detection.py
```

아래 RTL runner와 전송 묶음 생성기는 구현했다. 로컬 보조 실행은 Icarus,
수업 서버 실행은 VCS를 명시한다. 서버 명령과 FSDB 절차는 [서버 실행 안내](SERVER_RUN.md)에 있다.

```powershell
python -m qec_detector_rtl.scripts.run_sim --vectors artifacts/qec_detector_rtl/d3_r3_z_v1_20261118_001/vectors --out-root artifacts/qec_detector_rtl/p2_local_20261118_006 --simulator iverilog
python -m qec_detector_rtl.scripts.prepare_server_bundle --vectors artifacts/qec_detector_rtl/d3_r3_z_v1_20261118_001/vectors --out-root artifacts/qec_detector_rtl/p2_server_bundle_20261118_002
```

고정 회로 실습이므로 처음부터 distance·basis·round를 CLI 옵션으로 일반화하지 않는다.
벡터 생성기가 네 묶음과 고정 설정을 manifest에 남긴다. 주 simulator는 VCS를 사용하고
compile/run 실패를 비정상 종료 코드로 전달한다. runner가 검증 실패를 정상 완료처럼
표시하지 않게 한다. Icarus는 로컬 보조 검사로만 사용할 수 있으며 결과에 simulator를
명시한다. VCS 실행·FSDB 옵션은 준비했으며 실제 수업 환경에서 검증해야 한다.
DC 합성 명령은 P4에서 경로·setup 확인 후 추가한다.

코드가 추가되면 해당 파일의 `py_compile`, 작은 Python 회귀 검사와 RTL simulation을
수행한다. 기존 공유 코드까지 수정하면 그 경로의 회귀 검사도 실행한다. 코드 수정 후에는
저장소 규칙에 따라 `graphify update .`를 실행한다. 이 계획 문서와 폴더 준비만으로
전체 실험 test suite나 합성을 실행했다고 기록하지 않는다.

## 10. 최종 완료 체크리스트

- [x] P1의 회로·Python/Stim 버전·측정 순서·detector 순서·hash가 고정됐다.
- [x] 첫 round 4개, 중간 16개, 마지막 경계 4개의 참조가 확인됐다.
- [x] Python 직접 XOR와 Stim 정답이 모든 계획 벡터에서 일치한다.
- [x] 로컬 Icarus 기본 bench에서 RTL 값·ID·last·출력 수가 정답과 일치한다.
- [ ] 수업 서버의 VCS 검증과 Verdi 대표 파형 확인을 완료했다.
- [ ] 공백·정체·연속 shot·reset·동시 전송 검사가 통과한다.
- [ ] no-stall 지연, 입력 간격, shot 시작·경계 간격과 정체의 영향을 기록했다.
- [ ] 타깃·클록·I/O·corner 등의 합성 조건과 원본 자원·timing 보고서가 있다.
- [ ] 기능·목표 timing 통과 여부와 미해결 항목을 구분했다.
- [ ] frozen 입력과 실제 명령으로 검증을 재실행할 수 있다.
- [ ] 최종 보고서의 결론이 블록 검증 범위를 넘지 않는다.

P1과 P2 로컬 준비는 완료했다. 다음 작업은 전송 묶음의 서버 VCS·Verdi 검증이며,
이후 P3으로 넘어간다. 한 단계씩 결과를 확인한다. 전체 첫 목표를 완료한 후에는
실제로 확인한 병목에 따라 round 병렬 입력,
이전 round 기록만 저장하는 구조, 다중 shot buffering 또는 decoder 연결 중 한 가지를
다음 목표로 고른다. 현재 단계에서 이 확장들을 모두 구현하지 않는다.
