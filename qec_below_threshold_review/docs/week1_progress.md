# 1주차 진행 기록

- 기준일: 2026-08-24
- 범위: 공식 자료 확보, 핵심 용어 확인, $d=3$ X/Z baseline 및 correlated-noise smoke test
- 상태: 핵심 실행 경로 확인 완료

## 1. 공식 자료 확보

프로젝트 `paper/` 폴더에 다음 자료를 저장하고 PDF를 검증했다.

| 자료 | 파일 | 페이지 |
|---|---|---:|
| Nature 본문 | `quantum_error_correction_below_surface_code_threshold.pdf` | 8 |
| Supplementary Information | `supplementary_information.pdf` | 27 |
| 2026 저자 정정 | `author_correction_2026.pdf` | 1 |

공식 출처:

- Article: <https://doi.org/10.1038/s41586-024-08449-y>
- Author Correction: <https://doi.org/10.1038/s41586-026-10559-8>

## 2. 저자 정정 확인

2026년 저자 정정은 Fig. 3a의 표기만 수정한다.

- x축: `Surface code distance, d` → `Repetition code distance, d`
- 범례: `d=3 (ref. 17)` / `d=5` → `Ref. 17` / `This work`
- 논문의 핵심 수치와 결론은 변경되지 않았다.

포스터에서 Fig. 3a를 설명할 경우 반드시 repetition-code distance라고 표기한다.

## 3. Supplementary Information 1차 확인

### 코드와 회로

- Supplementary Fig. S15는 모든 stabilizer가 ZXXZ를 측정한다고 명시한다.
- “X stabilizer”와 “Z stabilizer”는 서로 다른 CZ ordering과 logical basis를 구분하는 convention이다.
- 프로젝트 simulation은 Stim rotated CSS memory이므로 Willow ZXXZ 회로의 직접 재현으로 표현하지 않는다.

### Decoder

Supplementary Information은 다섯 decoder 계열을 구분한다.

1. Ref. 16의 neural-network decoder
2. Ensemble size 51의 Libra matching synthesis
3. Ensemble size 101의 Harmony decoder
4. Correlated matching decoder
5. Real-time decoder

Neural-network decoder는 synthetic pretraining 후 fitted DEM과 experimental data를 사용해 두 단계로 fine-tuning한다. 장시간 memory experiment를 위해 gated recurrence가 추가되었다.

### 실시간 복호

- 평균 software latency: $63\,\mu\mathrm{s}$
- Input latency: $T_{\mathrm{input}}<10\,\mu\mathrm{s}$
- Output latency와 실제 control feedback은 실험에 포함되지 않음
- Block size: surface code $d=3,5$는 10 cycles, repetition code $d=29$는 90 cycles
- Constant-sized graph buffer: 128 blocks

## 4. 실행 환경 확인

| Dependency | Version |
|---|---:|
| Python | 현재 저장소 실행 환경 |
| Stim | 1.15.0 |
| PyMatching | 2.3.1 |
| Sinter | Stim 1.15.0 배포본 |
| NumPy | 2.2.6 |

## 5. $d=3$ X/Z smoke test

실행 조건:

- Distance: 3
- Rounds: 3
- Basis: X, Z
- Synthetic noise parameter: $p=0.003$
- Shots: 2,000 per basis/family
- Seed: 20260824
- Families: `stage_a_si1000`, `stage_c_corr`

생성 위치:

```text
artifacts/qec_below_threshold_review/smoke_d3/
```

PyMatching 결과:

| Logical axis | Family | Frame error rate | Estimated $\varepsilon$ per cycle |
|---|---|---:|---:|
| $X$ flip | Stage A | 0.0215 | 0.007272 |
| $X$ flip | Stage C | 0.0280 | 0.009513 |
| $Z$ flip | Stage A | 0.0220 | 0.007444 |
| $Z$ flip | Stage C | 0.0250 | 0.008476 |

이 수치는 2,000-shot 실행 경로 검증용이며 연구 결과로 해석하지 않는다. Stage A와 Stage C는 noise 구성과 detection probability가 일치하지 않으므로 통제 비교가 아니다.

## 6. Correlated ZZ-only 경로 확인

기존 `stage_c_corr` 기본 설정은 다음 두 component를 모두 포함한다.

- `CORRELATED_ERROR`: correlated ZZ
- `PAULI_CHANNEL_2`: swap-like component

프로젝트의 목표는 correlated ZZ만 비교하는 것이므로 기본 Stage C를 그대로 사용하면 안 된다. `enable_swap_component=False`로 별도 smoke test한 결과는 다음과 같다.

| Basis | Detector shape | Observable shape | Correlated ZZ instructions | Swap-like instructions |
|---|---:|---:|---:|---:|
| X | `(128, 24)` | `(128, 1)` | 72 | 0 |
| Z | `(128, 24)` | `(128, 1)` | 72 | 0 |

따라서 기존 noise builder를 수정하지 않고 config override만으로 ZZ-only circuit을 만들 수 있다.

## 7. 현재 결론

- Stim/PyMatching X/Z baseline 경로가 정상 작동한다.
- Correlated ZZ-only circuit과 DEM 생성이 정상 작동한다.
- 본 실험에서는 `stage_a_si1000`과 기본 `stage_c_corr`를 직접 비교하지 않는다.
- Independent/correlated 비교는 동일한 local-noise baseline을 사용하는 Stage B와 ZZ-only Stage C로 구성해야 한다.
- `variant="xzzx"`는 실제 Willow schedule이 아니므로 사용하지 않는다.

## 8. 다음 작업

1. Stage B와 ZZ-only Stage C를 같은 parameter로 생성하는 최소 project runner 작성
2. 공통 CSV schema 확정
3. Adaptive sampling 중단 조건 확정
4. $p$ pilot sweep 범위와 detection-probability matching 절차 확정
5. 세 팀원 이름과 실제 담당자 입력
