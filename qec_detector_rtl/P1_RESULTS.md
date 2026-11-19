# P1 결과: Python 기준 모델과 검증 벡터

- 완료일: 2026-11-18, Asia/Seoul
- 판정: **P1 통과. RTL 구현은 다음 단계다.**
- 회로: `stim_rotated`, `d=3`, 3라운드, Z memory
- 환경: Python 3.10.20, NumPy 2.2.6, Stim 1.15.0, Anaconda `fp`
- frozen run: `artifacts/qec_detector_rtl/d3_r3_z_v1_20261118_001/`

## 1. 구현한 내용

[기준 모델과 벡터 생성기](model/generate_vectors.py)는 기존 `CircuitConfig`와
`build_memory_circuit`를 사용한다. 이상 회로를 펼쳐 측정 순서와 detector 참조를 추출하고,
각 detector의 측정 인덱스, 좌표, 기준 패리티와 필요한 측정 수를 저장한다.
측정 33개, detector 24개, observable 1개를 확인하며 observable은 출력에 넣지 않는다.

직접 변환은 참조한 측정값들의 XOR에 기준 패리티를 XOR한다. Stim 변환 결과를 복사해
기준 모델로 사용하지 않는다. 회로가 무잡음에서 결정적인 detector를 갖는지도 확인한다.
지원 범위는 단일 큐비트 측정과 measurement-record 참조로 한정하며, MPP와 같은
미지원 측정은 오류로 처리한다.

같은 측정 배열을 Stim 변환기에 전달해 정답을 만들고 직접 XOR와 비교한다.
SI1000 회로는 기존 `apply_si1000_noise`로 생성했다. 주입 전후의 측정·detector map이
같고, 잡음을 제거해 펼친 회로가 이상 회로와 같은지도 확인했다.

벡터 파일은 ASCII `.01` 형식으로 저장한다. 왼쪽 첫 문자가 인덱스 0이다.
파일을 다시 읽어 길이와 값이 보존되는지 확인하며, 기존 run 폴더가 있으면 덮어쓰지 않는다.
manifest는 마지막에 작성하고 완료 상태, 버전, 설정, seed, source·생성물 SHA-256을 남긴다.

## 2. Stim 대비 비교 결과

| 묶음 | 기록 수 | 비교한 detector bits | 불일치 | detection event 합계 |
| --- | --- | --- | --- | --- |
| 무잡음 | 256 | 6,144 | 0 | 0 |
| 기준 기록 + 33개 단일 비트 반전 | 34 | 816 | 0 | 52 |
| 임의 비트 기록 | 1,024 | 24,576 | 0 | 12,384 |
| SI1000, p=0.001 | 4,096 | 98,304 | 0 | 2,466 |
| 합계 | 5,410 | 129,840 | 0 | 14,902 |

SI1000의 p는 기본 scale이며 실제 채널 확률은 manifest의 `noise_resolved_rates`에
기록했다. 단일 반전과 임의 비트 기록은 디지털 연결 검사용이다. 이들의 event 합계나
전체 합계를 물리적 오류율 또는 logical error rate로 해석하지 않는다.

저장한 회로와 벡터를 다시 읽어 Stim 변환과 직접 XOR 비교를 재실행했다.
소스 파일 7개와 생성물 13개의 hash를 확인했고, 계획 문서의 24개 참조 행과
`required_count`도 추출 결과와 일치했다. 기준 패리티는 현재 회로에서 모두 0이다.

## 3. 회귀 검사

[테스트](../tests/test_measurement_to_detection.py)의 5개 검사가 모두 통과했다.

1. 첫·마지막 경계식, `REPEAT`를 펼친 좌표, 측정 group과 단일 비트 연결을 검사한다.
2. 기준 패리티가 1인 작은 회로와 동일 참조 두 번의 XOR 상쇄를 검사한다.
3. 무잡음에서 결정적이지 않은 detector와 미지원 측정을 거부하는지 검사한다.
4. 배열 폭·차원·0/1 값과 잘못된 `.01` 파일을 검사한다.
5. 네 묶음 전체의 생성, hash·파일 왕복과 기존 run 덮어쓰기 방지를 검사한다.

실제로 실행한 명령은 다음과 같다. 사용한 Python은
`C:/Users/82108/anaconda3/envs/fp/python.exe`이며 저장소 루트에서 실행했다.

```powershell
python -m py_compile qec_detector_rtl/model/generate_vectors.py tests/test_measurement_to_detection.py
python -m unittest discover -s tests -p test_measurement_to_detection.py -v
python -m qec_detector_rtl.model.generate_vectors --out-root artifacts/qec_detector_rtl/d3_r3_z_v1_20261118_001
```

생성 명령을 다시 실행하려면 새 run ID를 지정한다. 위 `001`은 검증된 frozen 입력으로
보관하며 동일 경로에 재생성하지 않는다. 기존 공유 코드는 수정하지 않았으므로 이번에는
새 모델의 검사와 저장 파일 검증을 수행했다.

## 4. 근거 파일

- [Manifest](../artifacts/qec_detector_rtl/d3_r3_z_v1_20261118_001/vectors/manifest.json)
- [측정 map](../artifacts/qec_detector_rtl/d3_r3_z_v1_20261118_001/vectors/measurement_map.json)
- [Detector map](../artifacts/qec_detector_rtl/d3_r3_z_v1_20261118_001/vectors/detector_map.json)
- [저장 파일 재검증 결과](../artifacts/qec_detector_rtl/d3_r3_z_v1_20261118_001/report/p1_verification.json)

Manifest SHA-256:

```text
dc0e6cc61c1a823fa639c08a18e7177644d27196175cdcd3ca9b2661999a31eb
```

## 5. 다음 단계와 남은 범위

다음 P2는 frozen detector map을 기준으로 고정 Verilog 모듈과 기본 testbench를
작성하는 단계다. 입력 33개와 출력 24개, 첫·마지막 경계, ID 순서와 handshake를 검증한다.

이번 단계에서 RTL, cycle latency, 처리 간격, 합성 timing·area는 검증하지 않았다.
[환경 점검](ENVIRONMENT_CHECK.md)에서 발견한 compiler 비정상 종료와 GTKWave 실행
문제도 그대로 남아 있다. P1의 Python 검사 통과를 이 문제의 해결로 해석하지 않는다.
