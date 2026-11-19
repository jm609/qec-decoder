# QEC Below-Threshold 프로젝트 상세 역할분담 계획

## 1. 문서 목적

이 문서는 3인 팀이 다음 프로젝트를 수행하기 위한 역할, 산출물, 협업 방식과 완료 기준을 정한다.

> **Stim의 rotated surface-code memory에서 below-threshold scaling을 정성적으로 검증하고, detection probability를 맞춘 independent noise와 spatially correlated ZZ noise에서 logical error suppression factor가 어떻게 달라지는지 분석한다.**

구현은 Willow processor의 정확한 재현이 아니다. Willow의 ZXXZ schedule, calibrated device noise, Google neural-network decoder와 실시간 streaming system은 구현 범위에 포함하지 않는다.

---

## 2. 역할 배정 요약

| 구분 | 팀원 1 | 팀원 2 | 팀원 3 |
|---|---|---|---|
| 주 역할 | 논문·이론·문서 통합 | Baseline simulation | Correlated noise·decoder 분석 |
| 핵심 질문 | 논문이 무엇을 증명했는가? | 단순 모델에서 below-threshold가 나타나는가? | Correlation과 prior mismatch가 성능을 얼마나 낮추는가? |
| 주요 입력 | 논문 본문, Supplementary Information | 공통 실험 규격 | Baseline 회로, CSV schema |
| 주요 산출물 | 배경·논문 리뷰·한계 | Baseline CSV·threshold graph·$\Lambda$ | Correlated CSV·prior 비교·통계 그래프 |
| 포스터 | 왼쪽 | 가운데 | 오른쪽 |
| 발표 | 배경과 논문 성과 | 구현과 baseline | 확장 결과와 결론 |

실제 배정:

- 팀원 1: ____________________
- 팀원 2: ____________________
- 팀원 3: ____________________

---

## 3. 공통 연구 규격

역할별 작업을 시작하기 전에 세 팀원이 다음 사항을 함께 확정한다.

### 3.1 공통 실험 조건

- Circuit: Stim `surface_code:rotated_memory_x/z`
- Code distance: $d=3,5,7$
- Basis: $X$, $Z$ 모두 실행
- QEC rounds: 기본값 `rounds=distance`
- Decoder: PyMatching MWPM
- Baseline noise: circuit-level independent noise
- Extension noise: CZ 인접 edge의 correlated ZZ surrogate
- Sampling: 목표 logical failure 수 또는 최대 shots 도달 시 종료
- Uncertainty: 95% binomial confidence interval
- 결과 저장 위치: `artifacts/qec_below_threshold_review/`

### 3.2 공통 평가 지표

- Shot logical failure probability $p_L$
- Logical error per cycle

$$
\varepsilon_d=
\frac{1}{2}
\left[1-(1-2p_L)^{1/r}\right]
$$

- Pairwise error suppression factor

$$
\Lambda_{3\rightarrow5}=\frac{\varepsilon_3}{\varepsilon_5},
\qquad
\Lambda_{5\rightarrow7}=\frac{\varepsilon_5}{\varepsilon_7}
$$

- $\ln\varepsilon_d$ 대 $d$ 회귀로 구한 단일 $\Lambda$
- Mean detection probability $p_{\mathrm{det}}$
- Optional: offline batch decode throughput

Simulation의 $p_{\mathrm{det}}$는 모든 detector bit의 평균인 `detections.mean()`으로 통일한다. 이는 논문의 bulk detection probability와 동일한 지표가 아니므로 수치를 직접 비교하지 않는다.

### 3.3 공통 CSV column

모든 실험 결과는 최소한 다음 column을 사용한다.

```text
basis,distance,rounds,noise_model,decoder_prior,p,p_corr,
shots,failures,p_L,epsilon,p_det,seed
```

팀원 2가 schema를 먼저 확정하고, 팀원 3은 같은 schema를 그대로 사용한다.

---

## 4. 팀원 1 — 논문 리뷰·이론·문서 통합

### 4.1 목표

원 논문의 주장과 프로젝트 simulation의 범위를 정확하게 구분하고, 결과를 과장 없이 설명할 수 있는 이론·문서 기반을 만든다.

### 4.2 세부 담당 업무

#### 논문 분석

- 논문 본문 공동 1회독 주도
- Supplementary Information 확보 및 관련 절 검토
- Surface code, stabilizer, syndrome, detector 설명
- Code distance와 threshold 의미 정리
- Logical error per cycle과 $\Lambda$ 정의 정리
- Below-threshold와 breakeven의 차이 설명
- Willow의 $d=3,5,7$ 결과와 주요 수치 검증
- Leakage, drift, rare correlated burst 정리
- Real-time throughput과 decoder latency 구분
- 원문 ZXXZ 표기와 XZZX 참고문헌 convention 확인

#### 결과 해석 지원

- 팀원 2·3이 만든 그래프의 물리적 해석 검토
- Willow 수치와 simulation 수치를 직접 비교하지 않았는지 확인
- 관측 결과, 저자의 해석, 프로젝트의 추론을 구분
- 통계적으로 불확실한 결과의 단정적 표현 방지

#### 문서 작성

- 최종 보고서 Introduction
- Surface Code Background
- Review of the Willow Experiment
- Real-Time Decoding
- Discussion의 논문 한계와 프로젝트 한계
- 참고문헌 및 그림 출처 정리
- 팀원별 초안의 용어와 기호 통일

### 4.3 산출물

- 논문 핵심 수치 표 1개
- 핵심 개념과 수식 설명 원고
- 논문 Fig. 1–4의 한 문단 요약
- 포스터 왼쪽 영역 초안
- 최종 보고서 통합본
- 발표 도입부 대본

### 4.4 완료 기준

- $p_L$, $\varepsilon_d$, $\Lambda$, threshold, breakeven을 서로 구분해 설명할 수 있음
- Willow 실험과 프로젝트 simulation의 차이가 명시됨
- 모든 논문 수치에 출처 또는 그림 번호가 연결됨
- ZXXZ/XZZX 표기가 확인되거나 미확인 사항으로 명시됨
- 포스터와 보고서에서 동일한 용어와 기호를 사용함

### 4.5 담당하지 않는 항목

- Baseline simulation 구현의 단독 소유
- Correlated decoder 구현의 단독 소유
- 확인되지 않은 Willow hardware 세부사항 추정

---

## 5. 팀원 2 — Baseline Surface-Code Simulation

### 5.1 목표

Stim과 PyMatching으로 $d=3,5,7$의 below-threshold scaling을 정성적으로 검증할 수 있는 재현 가능한 baseline을 만든다.

### 5.2 세부 담당 업무

#### 환경과 smoke test

- `stim`, `pymatching`, `sinter`, `numpy` 실행 환경 확인
- $d=3$, X/Z basis의 소규모 smoke test
- Detector와 observable shape 검증
- 고정 seed에서 결과 재실행 가능 여부 확인

#### Baseline 실험

- `circuits.py`의 rotated memory 생성 기능 우선 재사용
- $d=3,5,7$과 X/Z basis 실행
- 기본 `rounds=distance` 적용
- Independent circuit-level noise 적용
- 초기 $p$ pilot sweep 실행
- Crossing 주변의 $p$ 구간 세분화
- 목표 logical failure 수 또는 최대 shots 기반 sampling
- Zero-failure 결과의 upper bound 처리

#### 지표 계산

- $p_L$ 계산
- $\varepsilon_d$ 계산
- 95% confidence interval 계산
- Pairwise $\Lambda$ 계산
- Regression 기반 $\Lambda$ 계산
- $p_{\mathrm{det}}$ 계산

#### 데이터 인계

- 공통 CSV schema 확정
- Baseline 회로와 DEM 저장
- 실행 parameter와 dependency version 기록
- 팀원 3에게 회로, DEM, CSV와 plotting 기준 전달

### 5.3 산출물

- Baseline experiment CSV
- Basis별 및 평균 결과표
- Physical noise parameter 대 $\varepsilon_d$ 그래프
- $d=3,5,7$ threshold-crossing 그래프
- Pairwise/regression $\Lambda$ 표
- 실행 명령과 환경 정보
- 포스터 가운데 영역 초안
- 발표 방법론 대본

### 5.4 완료 기준

- $d=3,5,7$과 X/Z basis가 모두 실행됨
- 같은 설정과 seed로 결과를 다시 만들 수 있음
- Threshold 아래에서 distance 증가에 따른 오류 감소가 확인됨
- Error bar와 total shots가 모든 결과에 표시됨
- $p_L$와 cycle당 $\varepsilon_d$가 구분되어 저장됨
- 팀원 3이 별도 변환 없이 같은 CSV schema를 사용할 수 있음

### 5.5 담당하지 않는 항목

- Willow의 정확한 ZXXZ schedule 구현
- Google neural-network decoder 재현
- 실시간 Sparse Blossom system 구현
- 포스터 전체 디자인의 단독 소유

---

## 6. 팀원 3 — Correlated Noise·Decoder-Prior 분석

### 6.1 목표

비슷한 detection probability에서도 spatial correlation과 decoder-prior mismatch가 logical error suppression을 악화시키는지 평가한다.

### 6.2 세부 담당 업무

#### Correlated noise 구성

- `noise_willowcore.py`의 correlated ZZ surrogate 우선 재사용
- CZ가 작동하는 인접 edge에 $P(Z_iZ_j)=p_{\mathrm{corr}}$ 적용
- Correlated instruction 수와 적용 edge 검증
- `decompose_errors=True`의 graphlike approximation 한계 기록

#### 비교 조건

1. Independent data + independent-noise decoder prior
2. Correlated data + independent-noise decoder prior
3. Correlated data + correlation-aware decoder weight

조건 A와 B는 $p_{\mathrm{det}}$ confidence interval이 겹치도록 pilot parameter를 조정한다. 조건 B와 C는 동일한 detector sample을 사용하여 decoder prior만 비교한다.

#### 분석

- Independent/correlated $p_L$ 비교
- Independent/correlated $\varepsilon_d$ 비교
- $\Lambda$ 감소 여부 확인
- Mismatched/matched decoder의 성능 차이 계산
- X/Z basis별 민감도 비교
- 필요할 경우 offline batch throughput 측정

#### 통계와 시각화

- 팀원 2와 동일한 confidence interval 사용
- Error bar가 포함된 비교 그래프 생성
- Parameter, shots, failures를 결과표에 표시
- 통계적으로 차이를 구분하기 어려운 경우 그대로 보고

### 6.3 산출물

- Correlated-noise experiment CSV
- Detection-probability matching 결과표
- Independent 대 correlated noise 비교 그래프
- Mismatched 대 matched decoder 비교 그래프
- $\Lambda$ 비교표
- 포스터 오른쪽 영역 초안
- 발표 결과·결론 대본

### 6.4 완료 기준

- 세 비교 조건의 noise와 decoder prior가 명확하게 기록됨
- 조건 B와 C가 동일한 sample을 사용함
- 조건 A와 B의 $p_{\mathrm{det}}$ matching 품질이 보고됨
- Correlation 효과와 decoder mismatch 효과가 분리되어 해석됨
- Batch throughput을 측정했다면 논문의 $63\,\mu\mathrm{s}$와 직접 비교하지 않음

### 6.5 담당하지 않는 항목

- Temporal rare burst error floor 재현
- Leakage/DQLR의 완전한 물리 모델
- Z-biased noise를 포함한 추가 sweep
- FPGA 또는 streaming decoder 구현

---

## 7. 주차별 작업과 인계 일정

### 1주차 — 범위 확정과 smoke test

#### 공동

- 논문 공동 1회독
- 연구 질문과 용어 확정
- 공통 실험 조건 확정
- 저장 위치와 파일명 규칙 확정

#### 팀원별

- 팀원 1: 논문 핵심 주장과 수치표 초안
- 팀원 2: $d=3$, X/Z baseline smoke test
- 팀원 3: correlated ZZ instruction smoke test

#### 주차 종료 조건

- 작은 sample에서 세 작업 경로가 모두 실행됨
- Willow 재현이 아니라 정성적 검증이라는 표현에 합의함

### 2주차 — 실험 protocol 고정

- 팀원 1: Background와 논문 리뷰 초안
- 팀원 2: CSV schema, pilot $p$ sweep, sampling 기준 확정
- 팀원 3: $p_{\mathrm{corr}}$ pilot과 decoder-prior 구성

주차 종료 시 팀원 2가 schema와 baseline sample을 팀원 3에게 전달한다. 이후 column 변경은 세 팀원이 합의해야 한다.

### 3주차 — Baseline 본 실험

- 팀원 2: $d=3,5,7$, X/Z baseline 본 실험
- 팀원 3: Baseline 결과를 사용한 correlated parameter calibration
- 팀원 1: 그래프 해석 기준과 논문 비교 문장 작성

주차 종료 조건:

- Threshold curve 초안
- Pairwise/regression $\Lambda$ 초안
- Confidence interval 포함 결과표

### 4주차 — Correlated-noise 확장

- 팀원 3: 세 비교 조건 본 실험
- 팀원 2: 추가 sampling과 결과 계산 지원
- 팀원 1: 결과 해석과 한계 검토

주차 종료 조건:

- Independent/correlated 비교 그래프
- Decoder-prior 비교 그래프
- Detection-probability matching 결과표

### 5주차 — 보고서와 포스터 통합

- 팀원 1: 최종 보고서 초안 통합
- 팀원 2: Method와 baseline 결과 작성
- 팀원 3: Extension, Discussion, Limitation 작성
- 공동: 포스터 1차 완성 및 내부 검토

### 6주차 — 최종 검증과 발표

- 모든 그래프의 수치와 CSV 대조
- 논문 수치와 simulation 수치의 표현 검토
- 포스터 최종 수정
- 발표 대본 통합
- 9~10분 발표 리허설
- 예상 질문과 답변 정리

---

## 8. 공동 의사결정 항목

다음 항목은 한 명이 단독으로 바꾸지 않는다.

- 연구 질문과 핵심 결론
- Circuit distance, rounds, basis
- Noise parameter 범위
- Sampling 중단 조건
- Detection probability matching 기준
- CSV schema
- 그래프 축과 단위
- 결과 제외 기준
- 논문과 simulation의 비교 표현

변경이 필요하면 이유와 변경 전후 값을 간단히 기록한다.

---

## 9. 작업 인계 규칙

각 산출물에는 다음 정보를 포함한다.

- 실행 명령 또는 생성 방법
- 입력 파일
- 출력 파일
- Parameter와 seed
- Dependency version
- 완료 여부
- 알려진 한계

실험 데이터는 메신저 파일로 따로 전달하지 않고 `artifacts/qec_below_threshold_review/` 아래의 합의된 경로를 사용한다. 그래프는 원본 CSV로 다시 생성할 수 있어야 한다.

---

## 10. 포스터 및 발표 분담

### 포스터 왼쪽 — 팀원 1

- QEC와 surface code 배경
- Threshold와 $\Lambda$
- Willow 논문의 핵심 결과
- 논문의 correlated-error 문제

### 포스터 가운데 — 팀원 2

- Stim–PyMatching pipeline
- $d=3,5,7$, X/Z basis
- Baseline noise와 sampling 방법
- Threshold curve와 baseline $\Lambda$

### 포스터 오른쪽 — 팀원 3

- Correlated ZZ noise
- Detection-probability matching
- Decoder-prior comparison
- 핵심 결론과 한계

### 발표 시간

- 팀원 1: 약 3분 — 배경과 논문
- 팀원 2: 약 3분 — 구현과 baseline
- 팀원 3: 약 3분 — 확장 결과와 결론
- 공동 질의응답 준비

---

## 11. 지연 시 최소 완료 범위

일정이 부족하면 다음 순서로 범위를 줄인다.

1. Offline batch throughput 삭제
2. Correlation-aware decoder weight 비교 삭제
3. Correlated noise는 대표 parameter 한 개만 실행
4. Baseline $d=3,5,7$, X/Z와 $\Lambda$ 계산은 반드시 유지

Temporal burst, leakage, DQLR, neural-network decoder와 실시간 decoder는 처음부터 구현 범위에 포함하지 않는다.

---

## 12. 최종 완료 체크리스트

### 논문 리뷰

- [ ] 논문과 Supplementary Information 검토
- [ ] 핵심 수치와 그림 출처 확인
- [ ] ZXXZ/XZZX convention 확인 또는 미확인 표시
- [ ] 논문의 주장과 프로젝트 simulation 범위 구분

### Baseline

- [ ] $d=3,5,7$ 완료
- [ ] X/Z basis 완료
- [ ] Confidence interval 포함
- [ ] Threshold curve 생성
- [ ] Pairwise/regression $\Lambda$ 계산

### Correlated analysis

- [ ] Correlated ZZ noise 적용 검증
- [ ] Detection probability matching
- [ ] Mismatched decoder 결과
- [ ] Matched decoder 결과 또는 범위 축소 사유 기록
- [ ] Correlation과 decoder mismatch 효과 분리

### 최종 산출물

- [ ] CSV와 metadata 정리
- [ ] 그래프 원본과 최종 이미지 정리
- [ ] 보고서 완성
- [ ] 포스터 완성
- [ ] 발표 대본 완성
- [ ] 리허설과 예상 질문 준비

---

## 13. 프로젝트 핵심 메시지

> **Surface code의 확장 가능성은 평균 물리 오류율이 threshold 아래인지뿐 아니라, 오류의 공간적 correlation과 decoder prior가 실제 noise를 얼마나 잘 반영하는지에도 좌우된다.**
