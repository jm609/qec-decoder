# Willow 공개 실험 데이터 재디코딩 계획

> 상태: **Phase 0–4 완료**\
> 데이터 출처: Google Quantum AI, Zenodo DOI [`10.5281/zenodo.13273331`](https://doi.org/10.5281/zenodo.13273331)

## 1. 목적

공개된 Willow 105Q surface-code memory 데이터를 직접 읽고 제공된 detector error model과 PyMatching으로 재디코딩하여 다음을 확인한다.

1. 공개 데이터에서 distance $d=3,5,7$의 logical error versus QEC rounds를 계산할 수 있는가?
2. 논문과 같은 cycle당 logical error $\epsilon_d$ fitting 절차를 적용할 수 있는가?
3. Matching decoder에서 suppression factor $\Lambda$를 재현할 수 있는가?
4. 제공된 decoder 예측, 직접 PyMatching 결과와 논문의 neural-network decoder 결과가 어떻게 다른가?

이 단계는 기존 synthetic simulation을 폐기하지 않는다. Synthetic 결과는 통제된 threshold·correlation 분석으로 유지하고, 공개 데이터 분석은 실제 hardware 결과와 논문 fitting을 검증하는 별도 축으로 추가한다.

## 2. 목표 표현

### 목표로 하는 것

> **공개 Willow syndrome과 제공 DEM을 사용한 matching-decoder logical error 및 $\Lambda$ 재현**

### 목표로 하지 않는 것

- PyMatching으로 neural-network decoder의 $\Lambda=2.14\pm0.02$를 정확히 재현
- AlphaQubit 또는 Willow neural decoder 구현
- 전체 112.5GB 데이터의 즉시 처리
- Repetition-code error floor의 1차 단계 재현

논문의 $\Lambda=2.14\pm0.02$는 neural-network decoder 결과이므로 PyMatching 결과의 정답값으로 사용하지 않는다. 동일한 decoder 계열과 fitting 조건의 공개 결과를 비교 기준으로 사용한다.

## 3. 1차 데이터 범위

| 항목 | 선택 |
|---|---|
| Archive | `google_105Q_surface_code_d3_d5_d7.zip` |
| Download size | 5.7GB |
| MD5 | `21fa6ad35b395d838ebcdbc92e364a12` |
| Code distance | $d=3,5,7$ |
| Logical basis | X/Z |
| 포함 자료 | Stim circuit, detector events, actual observable flips, DEM, decoder predictions, metadata |
| 제외 | 72Q surface-code set 1/2, 65GB repetition-code archive |

공식 archive 전체를 source code 폴더에 넣지 않고 다음 artifact 경로를 사용한다.

```text
artifacts/qec_below_threshold_review/willow_data/
```

## 4. 단계별 작업

### Phase 0 — Archive와 README 검증

**완료.** Archive 크기와 MD5가 공식값과 일치했고 9,959 entries, 420 configurations의 schema를 확인했다.

- 저장공간 확인
- 공식 archive 다운로드
- MD5 검증
- README와 metadata schema 확인
- archive 내부 distance, patch, basis, round 목록 생성
- 전체 압축 해제 대신 ZIP에서 필요한 파일만 읽는 방식 사용

완료 기준: README의 예제 명령과 필요한 파일 관계를 문서화한다.

### Phase 1 — 단일 configuration pilot

**완료.** `d3_at_q4_5/X/r01`에서 measurement 변환과 direct/provided prediction이 byte-for-byte 일치했고 둘 다 306/50,000 failures였다.

- 작은 $d=3$ X-memory configuration 하나 선택
- `metadata.json` 파싱
- `detection_events.b8`와 `obs_flips_actual.b8` shape 검증
- 제공된 `obs_flips_predicted.b8`로 logical failure 계산
- README 예제 또는 제공 결과와 일치 확인

완료 기준: shot 수, detector 수, actual/predicted bit 수와 logical failure가 모순 없이 계산된다.

### Phase 2 — PyMatching 직접 재디코딩

**완료.** 각 `circuit_noisy_si1000.stim`에서 DEM을 다시 생성해 364 main-sweep configurations을 batch decode했다.

- Noisy SI1000 circuit에서 DEM 생성
- PyMatching graph 생성
- 동일 detection event를 batch decode
- 예측 observable과 actual observable 비교
- 제공된 matching-decoder prediction과 직접 결과 비교

완료 기준: 일치율과 logical error 차이를 수치로 보고하고, 차이가 있으면 decoder option·prior·correlation 처리 차이를 설명한다.

### Phase 3 — 전체 $d=3,5,7$ scaling

**완료.** 제공 decoder 5개 경로와 direct PyMatching 경로의 $\epsilon_d$, $\Lambda$를 CSV/JSON으로 저장했다. Public Libra $\Lambda=2.029$, direct PyMatching+SI1000 $\Lambda=1.525$였다.

- X/Z, patch, round별 logical failure 집계
- Main sweep `r10`, `r30`–`r250` 사용; RL calibration `r13` 제외
- $p_L(r)$에서 cycle당 $\epsilon_d$ fitting
- Patch별 및 aggregate $\epsilon_d$ 계산
- $\Lambda$와 회귀 불확실성의 간이 전파 계산

완료 기준: 재현한 $\epsilon_d$, $\Lambda$, 비교 decoder와 fitting 조건이 하나의 CSV와 figure에 기록된다.

### Phase 4 — 기존 프로젝트와 통합

**완료.** 공개 Willow 결과를 보고서와 포스터의 주 결과로, synthetic 결과를 보조 검증으로 재배치했다.

- Synthetic baseline과 실제-data scaling을 구분해 비교
- Correlated-ZZ stress test는 decoder-prior 해석용 확장 결과로 유지
- `content_master.md`, 최종 보고서와 포스터 원고 갱신

## 5. 최소 구현 원칙

- 기존 Stim 1.15.0과 PyMatching 2.3.1 사용
- README의 예제와 기존 저장소 유틸리티 우선 재사용
- 원본 archive와 선택 추출 pilot 데이터는 `artifacts/`에 보관
- 전체 archive는 압축 상태로 처리해 중복 저장을 피함
- 새 코드는 archive inspection과 decoding에 필요한 최소 script만 추가
- 원본 데이터는 수정하지 않고 derived CSV/JSON을 별도 저장

## 6. 완료 산출물

```text
qec_below_threshold_review/
├── analyze_willow_data.py
├── plot_willow_reproduction.py
└── docs/
    └── willow_data_reproduction_progress.md

artifacts/qec_below_threshold_review/willow_data/
├── google_105Q_surface_code_d3_d5_d7.zip
├── pilot/
├── pilot_decode.json
└── <decoder pathway>/
    ├── logical_error_by_configuration.csv
    ├── epsilon_by_patch_basis.csv
    ├── epsilon_by_distance.csv
    └── summary.json

artifacts/qec_below_threshold_review/
├── willow_reproduction_comparison.csv
└── willow_reproduction_comparison.png
```

## 7. 해석 규칙

- “실제 Willow 데이터를 사용했다”와 “Willow neural decoder를 재현했다”를 구분한다.
- 제공 prediction을 재집계한 결과와 직접 PyMatching으로 다시 얻은 결과를 구분한다.
- Decoder, prior, fitting round, patch aggregation이 다르면 $\Lambda$를 직접 동일시하지 않는다.
- 논문 수치와 다를 경우 실패로 숨기지 않고 원인을 분석한다.
- Surface-code 분석이 완료되기 전에는 repetition-code archive를 받지 않는다.

## 8. 중단·확장 기준

다음 조건이면 전체 decoding 전에 원인을 먼저 해결한다.

- MD5 불일치
- README의 expected result와 pilot 불일치
- `.b8` bit ordering 또는 padding 해석 불명확
- 제공 DEM을 PyMatching에서 읽을 수 없음
- 추출 후 저장공간 부족 위험

Surface-code $d=3,5,7$ 재현이 완료된 뒤에만 72Q real-time 또는 repetition-code 데이터 확장을 검토한다.
