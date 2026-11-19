# QEC Below Surface Code Threshold 팀 프로젝트 계획

## 1. 프로젝트 개요

### 대상 논문
**Quantum error correction below the surface code threshold**\
Google Quantum AI and Collaborators, *Nature* (2025)

### 프로젝트 형태
- 3인 팀
- 논문 리뷰
- 단순화된 surface-code 시뮬레이션/구현
- 포스터 제작
- 최종 발표

### 권장 프로젝트 방향

단순히 논문을 3등분하여 각자 요약하기보다, 논문의 핵심 주장을 단순화된 simulation과 공개 Willow 실험 데이터 재분석으로 연결한다. Synthetic 구현은 threshold와 correlation을 통제해 설명하고, Zenodo 공개 데이터는 실제 hardware syndrome에서 matching-decoder scaling을 재현하는 데 사용한다.

논문의 핵심 축은 다음과 같다.

1. 코드 거리를 증가시켰을 때 논리 오류율이 감소하는 **below-threshold scaling**
2. Leakage, CZ correlation, rare burst 등 **실제 하드웨어의 오류 특성**
3. 정확도뿐 아니라 처리 속도까지 만족해야 하는 **decoder 및 real-time decoding**

---

## 2. 권장 프로젝트 주제

### 메인 주제

> **단순화된 Surface Code 시뮬레이션에서 Below-Threshold Scaling 검증과 상관 노이즈 영향 분석**

영문 제목 예시:

> **A Simulation Study of Below-Threshold Surface-Code Scaling under Correlated Noise**

이 주제는 다음 세 가지를 동시에 만족한다.

- 원 논문의 핵심 결과를 리뷰할 수 있음
- 핵심 경향을 단순화된 회로에서 직접 시뮬레이션할 수 있음
- 단순 재현을 넘어 correlated noise에 대한 확장 분석이 가능함

---

## 3. 전체 프로젝트 구성

프로젝트를 크게 세 부분으로 구성한다.

### Part A. 논문 리뷰

논문에서 달성한 주요 결과를 설명한다.

- Surface code threshold의 의미
- Code distance $d$
- Logical error per cycle $\varepsilon_d$
- Error suppression factor $\Lambda$
- Beyond-breakeven logical memory
- Leakage removal
- Correlated error floor
- Real-time decoding

### Part B. 핵심 현상 정성적 검증

실제 Willow processor 전체를 재현하지 않는다. Stim의 rotated planar surface-code memory와 PyMatching을 이용하여 일반적인 threshold 현상을 정성적으로 검증한다. 이 구현은 Willow의 ZXXZ schedule, calibrated device noise 및 Google decoder의 수치 재현이 아니다.

- $d=3,5,7$ surface code 생성
- $X$와 $Z$ memory basis를 모두 실행
- 기본 QEC rounds는 code distance와 같게 설정
- 여러 physical error rate $p$에서 syndrome sampling
- MWPM decoder 적용
- Shot logical failure probability와 cycle당 logical error rate를 구분하여 계산
- Threshold crossing 관찰
- Error suppression factor 계산

$$
\Lambda =
\frac{\varepsilon_d}{\varepsilon_{d+2}}
$$

### Part C. 확장 분석

논문에서 다룬 실제 noise 요소 중 **인접 큐비트의 spatial correlated ZZ error** 하나만 선택한다. Temporal rare burst와 leakage의 완전한 물리 모델은 구현하지 않고 Discussion에서 한계로 다룬다.

다음 세 조건을 비교한다.

1. Independent noise data + independent-noise decoder prior
2. Correlated ZZ noise data + independent-noise decoder prior
3. Correlated ZZ noise data + correlation-aware decoder weight

Independent와 correlated 조건의 평균 detection probability를 비슷하게 맞춘 뒤 logical error suppression을 비교한다. 이를 통해 noise correlation 자체의 영향과 decoder-prior mismatch의 영향을 구분한다.

### Part D. 공개 Willow 데이터 재디코딩

Zenodo DOI `10.5281/zenodo.13273331`의 `google_105Q_surface_code_d3_d5_d7.zip`을 사용한다.

- 실제 $d=3,5,7$ X/Z memory syndrome과 observable flip 확인
- 제공 decoder prediction으로 logical error 재집계
- 제공 DEM과 PyMatching으로 직접 재디코딩
- Logical error versus rounds fitting
- Matching-decoder $\epsilon_d$와 $\Lambda$ 재현
- 논문의 neural-network decoder 결과와 decoder별 차이 분석

세부 절차와 완료 기준은 [`willow_data_reproduction_plan.md`](willow_data_reproduction_plan.md)를 따른다. PyMatching으로 neural-decoder $\Lambda=2.14$를 그대로 재현한다고 주장하지 않는다.

---

## 4. 3명 역할 분담

완전히 독립적인 세 작업으로 분리하면 최종 결과의 일관성이 떨어질 수 있으므로, 각자 주 담당 영역을 두되 구현 결과는 공동 분석하는 방식이 좋다.

### 팀원 1 — 이론 및 논문 리뷰 담당

#### 주요 담당

- Surface code 기본 원리
- Stabilizer와 syndrome
- Code distance의 의미
- Threshold 원리
- Logical error scaling
- 논문의 주요 실험 결과
- 선행 연구와 본 논문의 차이

#### 핵심 수식

$$
\varepsilon_d
\propto
\left(
\frac{p}{p_{\mathrm{thr}}}
\right)^{(d+1)/2}
$$

$$
\Lambda =
\frac{\varepsilon_d}{\varepsilon_{d+2}}
$$

#### 주요 산출물

- 최종 보고서의 Introduction
- Surface code background
- 논문 Fig. 1 중심 결과 해석
- 포스터 Background
- 발표 초반부

### 팀원 2 — Surface Code 시뮬레이션 담당

#### 주요 담당

- Stim surface code circuit 생성
- $d=3,5,7$ 코드 구성
- $X/Z$ memory basis 실행
- 기본 `rounds=distance` 적용
- Circuit-level depolarizing noise 적용
- Detector sampling
- PyMatching decoder 연결
- Logical error rate 계산
- Threshold curve 생성
- 통계적 불확실성 계산

#### 기본 실험 예시

초기 pilot physical error parameter:

$$
p=
0.0005,\,
0.001,\,
0.002,\,
0.004,\,
0.006,\,
0.008,\,
0.01,\,
0.015,\,
0.02
$$

각 $p$에서 $d=3,5,7$의 logical error rate를 측정한다. 본 실험에서는 pilot 결과를 보고 crossing 주변의 $p$를 더 촘촘히 선택한다. 여기서 $p$는 Willow의 측정 physical error rate가 아니라 Stim noise instruction에 사용하는 합성 parameter이다.

#### 주요 산출물

- Logical error rate vs physical error rate
- Code distance별 비교
- Threshold crossing 추정
- Pairwise 및 regression 기반 $\Lambda$ 계산
- 95% confidence interval
- 포스터 Method / Baseline Result

### 팀원 3 — Noise 확장 및 Decoder 분석 담당

#### 주요 담당

- 인접 큐비트 correlated ZZ noise 적용
- Decoder prior가 일치하는 경우와 불일치하는 경우 비교
- Detection probability matching
- Offline batch decoder throughput 측정(선택)
- Leakage와 rare burst에 대한 논문 분석
- 결과 통계 및 시각화

#### 확장 실험

**Spatial correlated noise**

$$
P(Z_iZ_j)=p_{\mathrm{corr}}
$$

Correlated error는 CZ가 작동하는 인접 edge에 적용한다. PyMatching graph는 independent-noise prior와 correlated-noise prior로 각각 구성한다. `decompose_errors=True`가 고차 error를 graphlike component로 근사한다는 한계를 보고서에 명시한다.

#### 주요 산출물

- Independent vs correlated noise 비교
- $\Lambda$ 변화
- Decoder-prior mismatch와 matched-weight 비교
- Offline batch throughput(선택)
- 포스터 Discussion / Extension / Limitation

---

## 5. 구현 범위 설정

### 최소 구현

시간이 부족할 경우 다음까지만 구현해도 충분하다.

1. Stim으로 rotated surface code memory 생성
2. $d=3,5,7$ 사용
3. $X/Z$ basis와 `rounds=distance` 사용
4. Depolarizing noise 적용
5. PyMatching decoding
6. Logical error rate와 confidence interval 측정
7. Threshold curve와 $\Lambda$ 계산

이 정도만 성공해도 논문의 핵심 주장인

> 물리 오류율이 threshold 아래일 경우, code distance를 증가시키면 logical error rate가 감소한다.

를 재현할 수 있다.

### 권장 구현

최소 구현에 다음을 추가한다.

- Independent noise와 correlated noise 비교
- 동일한 detection probability에서 logical error 차이 분석
- Mismatched decoder prior와 correlation-aware weight 비교
- Offline batch decoder throughput 측정(선택)

이 경우 다음과 같은 연구 질문을 설정할 수 있다.

> **동일한 평균 물리 오류 수준에서도 오류의 correlation 구조가 surface code의 오류 억제 성능을 얼마나 변화시키는가?**

이 질문은 논문의 leakage 및 correlated error 분석과 직접 연결된다.

### 구현 범위를 넘기기 쉬운 항목

다음은 이번 프로젝트에서는 권장하지 않는다.

- Willow hardware 전체 정밀 재현
- Google neural-network decoder 전체 재현
- Sparse Blossom 병렬 실시간 시스템 완전 구현
- DQLR의 완전한 물리 모델링
- Rare temporal burst의 error floor 재현
- Biased noise까지 포함한 다중 noise sweep
- $d=27$ 이상의 대규모 시뮬레이션
- FPGA 실시간 decoder 구현

### 기존 코드 활용과 산출물

이 저장소에는 이미 Stim/PyMatching dependency와 rotated memory, correlated ZZ surrogate, PyMatching evaluation 코드가 있다. 다음 파일을 우선 재사용한다.

- `circuits.py`: `rotated_memory_x/z` 생성
- `noise_willowcore.py`: 인접 edge의 correlated ZZ surrogate
- `sample_dataset.py`: detector/observable sampling과 metadata 저장
- `decoders/baseline_pymatching.py`: PyMatching 평가와 cycle당 오류율 추정

`variant="xzzx"`는 현재 실제 Willow schedule이 아니라 Stim rotated scaffold를 재사용하므로 Willow 재현으로 표현하지 않는다. 프로젝트 결과는 기존 졸업프로젝트 artifact와 분리하여 `artifacts/qec_below_threshold_review/`에 저장한다. 기존 코드를 재사용한 범위는 최종 보고서에 명시한다.

---

## 6. 구현 파이프라인

```text
Stim surface-code circuit 생성
        ↓
Noise model 적용
        ↓
Detector / Observable sampling
        ↓
Detector Error Model 추출
        ↓
PyMatching graph 생성
        ↓
MWPM decoding
        ↓
Predicted logical observable 계산
        ↓
Logical failure 판정
        ↓
Logical error rate 및 Λ 계산
```

---

## 7. 기본 구현 코드 구조

```python
import stim
import pymatching
import numpy as np

distance = 5
basis = "z"
rounds = distance
shots = 100_000
physical_error = 0.003

circuit = stim.Circuit.generated(
    f"surface_code:rotated_memory_{basis}",
    distance=distance,
    rounds=rounds,
    after_clifford_depolarization=physical_error,
    before_measure_flip_probability=physical_error,
    after_reset_flip_probability=physical_error,
)

dem = circuit.detector_error_model(
    decompose_errors=True
)

matching = pymatching.Matching.from_detector_error_model(dem)

sampler = circuit.compile_detector_sampler()

detections, observables = sampler.sample(
    shots=shots,
    separate_observables=True,
)

predictions = matching.decode_batch(detections)

logical_failures = np.any(
    predictions != observables,
    axis=1
)

shot_logical_error_probability = np.mean(logical_failures)
logical_error_per_cycle = 0.5 * (
    1 - (1 - 2 * shot_logical_error_probability) ** (1 / rounds)
)

print(shot_logical_error_probability, logical_error_per_cycle)
```

실제 실험에서는 다음 파라미터를 반복적으로 변경하면 된다.

- `distance`
- `basis`
- `physical_error`
- `rounds`
- `shots`

위 코드는 smoke test용 최소 예시이다. 본 실험에서는 고정 shots만 사용하지 않고, 목표 logical failure 수에 도달하거나 최대 shots에 도달할 때까지 sampling한다. Stim과 함께 설치된 Sinter를 사용하면 이 적응형 수집을 직접 구현할 필요가 없다.

실험 결과는 CSV로 저장하여 plotting과 통계 분석에 사용한다. 최소 column은 `basis`, `distance`, `rounds`, `noise_model`, `decoder_prior`, `p`, `p_corr`, `shots`, `failures`, `p_L`, `epsilon`, `p_det`, `seed`이다.

---

## 8. 핵심 실험 설계

### 실험 1 — Below-Threshold Scaling 정성적 검증

#### 독립변수

- Code distance: $d=3,5,7$
- Physical error rate: $p$
- QEC rounds

#### 종속변수

- Shot logical error probability $p_L$
- Logical error per cycle
- Error suppression factor $\Lambda$
- Mean detection probability $p_{\mathrm{det}}$

Simulation의 $p_{\mathrm{det}}$는 우선 `detections.mean()`으로 정의하여 모든 조건에 동일하게 적용한다. 이는 원 논문의 weight-4 stabilizer만 사용한 bulk detection probability와 완전히 같은 지표가 아니므로 직접적인 수치 비교는 하지 않는다.

#### 기대 결과

Threshold 아래의 낮은 $p$에서는

$$
\varepsilon_7
<
\varepsilon_5
<
\varepsilon_3
$$

가 나타난다.

반대로 높은 $p$에서는 큰 code distance가 더 이상 이점을 보이지 않는다.

$d=3,5,7$의 곡선이 교차하는 지점을 이용하여 사용한 noise model의 threshold 근처를 추정할 수 있다.

각 조건은 $X/Z$ basis에서 별도로 실행하고, basis별 결과와 평균을 모두 보고한다. 오류 막대는 shot failure에 대한 95% binomial confidence interval로 표시한다. 낮은 $p$에서 failure가 0회이면 오류율을 0으로 확정하지 않고 upper bound를 보고하거나 shots를 추가한다.

### 실험 2 — Error Suppression Factor

$$
\Lambda_{3\rightarrow5}
=
\frac{\varepsilon_3}{\varepsilon_5}
$$

$$
\Lambda_{5\rightarrow7}
=
\frac{\varepsilon_5}{\varepsilon_7}
$$

원 논문의 실제 수치

$$
\Lambda = 2.14\pm0.02
$$

를 정확히 재현할 필요는 없다.

실제 Willow hardware, noise model, decoder가 모두 다르기 때문이다.

중요한 것은 다음 경향이다.

- $\Lambda>1$인 영역이 존재하는가?
- Physical error rate가 낮아질수록 $\Lambda$가 증가하는가?
- Correlated noise가 추가되면 $\Lambda$가 감소하는가?

Pairwise ratio $\Lambda_{3\rightarrow5}$와 $\Lambda_{5\rightarrow7}$를 모두 제시하고, 논문과의 형식 비교를 위해 $\ln \varepsilon_d$를 $d$에 대해 선형 회귀한 단일 $\Lambda$도 함께 계산한다.

### 실험 3 — Correlated Noise의 영향

**조건 A: Independent data + independent decoder prior**

$$
P(Z_i)=p
$$

**조건 B: Correlated data + independent decoder prior**

$$
P(Z_i Z_j)=p_{\mathrm{corr}}
$$

**조건 C: Correlated data + correlation-aware decoder weight**

조건 A와 B의 평균 detection probability를 허용 오차 안에서 맞춘 뒤 logical error rate를 비교한다. 조건 B와 C는 동일한 detector sample을 사용하여 decoder-prior mismatch의 효과만 비교한다.

Detection probability matching은 pilot 결과에서 independent-noise parameter를 조정해 correlated 조건의 $p_{\mathrm{det}}$ confidence interval과 겹치게 하는 최소 절차로 수행한다.

핵심 해석은 다음과 같다.

Detection probability가 비슷하더라도 correlated noise가 존재하면 logical error rate가 더 높을 수 있다.

원 논문에서는 DQLR을 적용했을 때 detection probability는 약 12% 감소한 반면, error suppression factor $\Lambda$는 약 35% 증가하였다.

### 실험 4 — Offline Batch Decoder Throughput (선택)

```python
import time

start = time.perf_counter()

predictions = matching.decode_batch(detections)

batch_decode_seconds = time.perf_counter() - start

shots_per_second = shots / batch_decode_seconds
```

비교 항목:

- Code distance
- Detector 수
- Batch decode seconds
- Shots 또는 detector-events per second
- Logical error rate

이 측정은 I/O, streaming, syndrome 변환, graph construction 및 quantum feedback을 제외하므로 latency가 아니다. 원 논문의 $63\,\mu\mathrm{s}$와 직접 비교하지 않고 로컬 환경의 offline batch throughput 참고값으로만 사용한다.

---

## 9. 논문 리뷰 및 재현 보고서 구성

1. **Introduction**
   - Quantum Error Correction의 필요성
   - Physical qubit과 logical qubit
   - Surface code의 장점
   - Threshold의 의미
   - 본 리뷰 및 구현 목표

2. **Surface Code Background**
   - Data qubit
   - Measure qubit
   - Stabilizer measurement
   - Syndrome
   - Detector
   - Logical operator
   - Code distance
   - Logical error

3. **Below-Threshold Principle**
   - Threshold scaling
   - Error suppression factor
   - Below threshold / above threshold
   - Breakeven의 의미

4. **Review of the Willow Experiment**
   - 72-qubit / 105-qubit Willow
   - Distance-5 / distance-7
   - 원문 ZXXZ 표기와 XZZX 참고문헌의 convention
   - Neural-network decoder
   - Ensembled matching synthesis
   - 주요 결과

5. **Realistic Noise Sources**
   - CZ error
   - Idle error
   - Measurement error
   - Leakage
   - Stray interaction
   - Drift
   - Rare correlated burst

6. **Real-Time Decoding**
   - 실시간 decoding 필요성
   - Syndrome throughput
   - Sparse Blossom
   - Streaming graph
   - Decoder latency
   - Offline decoder와의 trade-off

7. **Reproduction Method**
   - Stim
   - PyMatching
   - Noise model
   - 실험 조건
   - 평가 지표
   - Willow 재현이 아닌 단순화된 model임을 명시

8. **Results**
   - Threshold curve
   - Logical error rate
   - $\Lambda$
   - Correlated noise 분석
   - Offline decoder throughput(선택)

9. **Discussion**
   - 논문과 구현 결과 비교
   - 동일한 수치가 나오지 않는 이유
   - 실제 hardware noise와 simulation model의 차이
   - Decoder의 한계
   - 향후 확장 방향

10. **Conclusion**

---

## 10. 포스터 구성

### 왼쪽 — Background and Paper Review

- 문제 정의
- Surface code 개념
- Threshold scaling 식
- Willow 논문의 주요 성과
- Distance-7 구조

### 가운데 — Method and Implementation

- Stim–PyMatching pipeline
- $d=3,5,7$
- Noise model
- Decoder
- 평가 지표
- 구현 flowchart

### 오른쪽 — Results and Discussion

- Logical error vs physical error 그래프
- $\Lambda$ 비교
- Independent vs correlated noise
- Offline decoder throughput(선택)
- 주요 결론
- Limitation
- Future work

---

## 11. 포스터에 넣을 핵심 그림

### 그림 1 — Surface Code 구조

표시할 내용:

- Data qubit
- Measure qubit
- X stabilizer
- Z stabilizer
- Logical operator

### 그림 2 — Simulation Pipeline

```text
Circuit
  ↓
Noise
  ↓
Syndrome
  ↓
Decoder
  ↓
Logical Error
```

### 그림 3 — 핵심 실험 결과

- x축: Physical error rate $p$
- y축: Logical error rate
- curve: $d=3,5,7$

### 그림 4 — 확장 분석

다음 중 하나 또는 두 개를 선택한다.

- Independent vs correlated noise
- $\Lambda$ vs physical error rate
- Decoder runtime vs code distance

---

## 12. 발표 분담

### 발표자 1

- 연구 배경
- Surface code
- Threshold
- $\Lambda$
- Willow 논문의 핵심 성과

### 발표자 2

- 구현 환경
- Stim
- PyMatching
- 실험 조건
- Baseline 결과

### 발표자 3

- Correlated noise 결과
- Decoder runtime
- 논문과의 비교
- 한계
- 향후 연구
- 결론

약 9~10분 발표라면 각자 3분 정도 배정하는 것이 적절하다.

---

## 13. 권장 일정

### 1주차
- 논문 공동 1회독
- Supplementary Information 확보 및 핵심 방법 확인
- 주요 용어 통일
- 역할 분담
- 개발 환경 구축
- $d=3$, X/Z basis smoke test

### 2주차
- 팀원 1: 이론 및 논문 요약
- 팀원 2: Stim–PyMatching baseline과 CSV schema 확정
- 팀원 3: Correlated ZZ 적용 smoke test
- 실험 범위와 sampling 중단 조건 확정

### 3주차
- $d=3,5,7$ 실험
- Logical error rate 계산
- Threshold curve 생성
- Confidence interval과 basis별 결과 검증

### 4주차
- Correlated ZZ noise 추가
- Mismatched/matched decoder prior 비교
- Offline batch throughput 측정(시간이 남을 경우)
- 논문 결과와 비교

### 5주차
- 논문 리뷰 및 재현 보고서 초안
- 그래프 정리
- 포스터 초안

### 6주차
- 문서 통합
- 포스터 최종 수정
- 발표 대본
- 예상 질문 대비

### 공개 데이터 재현 추가 단계

**완료(2026-08-24).** 105Q archive 검증, 단일 pilot, 제공 prediction 집계, 전체 direct PyMatching, $d=3,5,7$ fitting과 문서 통합을 마쳤다. 핵심 결과는 Public Libra $\Lambda=2.029$, direct PyMatching+SI1000 $\Lambda=1.525$다.

---

## 14. 최종 권장 연구 목표

> **Synthetic Stim model에서 below-threshold scaling과 correlated-noise 효과를 검증하고, 공개 Willow $d=3,5,7$ 실험 데이터를 PyMatching으로 재디코딩해 matching-decoder logical scaling을 재현한다.**

이 목표는 다음 장점이 있다.

- 원 논문의 핵심을 충실하게 반영
- 실제 구현 결과 확보 가능
- 단순 요약을 넘어선 확장 분석 가능
- 3인 역할 분담이 명확함
- 포스터에서 보여줄 그래프가 분명함

반대로 **Willow 논문 전체를 재현하는 것**은 여전히 목표가 아니다. Hardware data는 공개돼 있지만 Google의 neural decoder와 전체 calibration pipeline은 제공되지 않으므로 decoder별 재현 범위를 명확히 제한한다.

따라서 가장 현실적인 방향은 다음과 같다.

> **단순화된 simulation의 통제 실험과 공개 Willow 데이터의 matching-decoder 재분석을 결합하고, correlated noise와 decoder-prior mismatch가 logical error suppression에 미치는 영향을 설명한다.**

---

## 15. 프로젝트 핵심 메시지

```text
Physical qubit errors
        ↓
Surface code
        ↓
Syndrome extraction
        ↓
Decoder
        ↓
Logical error rate
        ↓
Below-threshold scaling
        ↓
Correlated noise가 scaling을 제한할 수 있음
```

최종적으로 다음 메시지를 중심으로 정리하면 좋다.

> **Surface code의 확장 가능성은 physical error rate가 threshold 아래인지뿐 아니라, 실제 noise의 correlation과 decoder가 이를 얼마나 잘 반영하는지에도 크게 의존한다.**

---

## 16. 사후 가설 평가

이 문서는 최초 계획과 기대 결과를 보존한다. 실제 simulation이 초기 목표·가설과 얼마나 일치했는지는 [`project_status.md`](project_status.md)의 **초기 목표·가설과 결과의 일치도**와 [`results_discussion_draft.md`](results_discussion_draft.md)의 **초기 가설에 대한 평가**에 정리했다.
