# Quantum Error Correction Below the Surface Code Threshold

## 논문 리뷰, 공개 데이터 재분석 및 단순화된 시뮬레이션 보고서 초안

## 초록

Google Quantum AI의 Willow 실험은 distance가 증가할수록 논리 오류가 감소하는 below-threshold surface-code memory를 실제 초전도 processor에서 시연했다. 본 프로젝트는 공개 105Q archive의 364 configurations에서 matching prediction을 재집계하고 Stim–PyMatching 직접 decode를 수행했다. Public Libra의 $\Lambda=2.029$는 논문 matching 기준 2.04와 가까웠으며, 동일 syndrome의 direct PyMatching+SI1000은 $\Lambda=1.525$였다. 모든 경로가 $\Lambda>1$이지만 decoder 경로가 scaling을 크게 바꿨다. 별도 synthetic baseline의 threshold crossing은 X 0.008852, Z 0.008089였고, detection-matched correlated-ZZ stress test에서는 X memory의 $\Lambda$가 independent 1.880에서 mismatched 1.374로 감소해 matched prior가 1.400으로 일부 회복했다. Z에서는 두 prior가 같았다. 실제-data 재계산과 synthetic 분석은 logical 성능이 hardware error뿐 아니라 correlation의 방향·세기, logical basis와 decoder prior의 적합성에 함께 의존한다는 결론을 지지한다.

## 1. 서론

Quantum error correction(QEC)의 목적은 불완전한 physical qubit 여러 개로 더 안정적인 logical qubit을 구성하는 것이다. Surface code에서는 stabilizer 측정을 반복해 data qubit을 직접 측정하지 않고 오류 정보를 얻으며, decoder가 syndrome으로부터 logical correction을 추론한다.

Code distance를 늘렸을 때 logical error가 감소하려면 장치가 threshold 아래에서 동작해야 한다. 이를 나타내는 대표 지표가 suppression factor

$$
\Lambda=\frac{\epsilon_d}{\epsilon_{d+2}}
$$

이다. 여기서 $\epsilon_d$는 distance $d$의 cycle당 logical error rate다. $\Lambda>1$이면 distance 증가가 logical error를 억제하고, $\Lambda\le1$이면 확장의 이점이 사라진다.

본 프로젝트는 다음 질문을 다룬다.

1. Willow 논문은 어떤 근거로 below-threshold QEC를 주장하는가?
2. 공개 matching prediction에서 논문의 $\Lambda$를 다시 계산할 수 있는가?
3. 동일 syndrome의 direct PyMatching은 어떤 scaling을 보이는가?
4. 단순화된 simulation에서 correlated noise와 decoder-prior mismatch는 logical scaling을 어떻게 바꾸는가?

## 2. 대상 논문의 핵심 내용

### 2.1 Below-threshold scaling과 beyond breakeven

Willow 논문은 105-qubit processor에서 101개 qubit을 사용하는 distance-7 ZXXZ surface-code memory를 구현했다. Neural-network decoder 기준 suppression factor는 $\Lambda=2.14\pm0.02$였고, distance-7 logical error는 $\epsilon_7=(1.43\pm0.03)\times10^{-3}$ per cycle이었다. 또한 logical memory lifetime은 $291\pm6\,\mu s$로, 최상 구성 physical qubit보다 $2.4\pm0.3$배 길었다. 이는 code 규모 증가에 따른 오류 억제와 beyond-breakeven memory를 함께 보여준다.

### 2.2 평균 오류율을 넘어선 error structure

논문은 leakage 제거 기법인 DQLR이 detection probability를 약 12% 낮추는 데 비해 $\Lambda$를 약 35% 높인다고 보고했다. 평균 syndrome activity의 작은 변화가 더 큰 logical-scaling 변화로 이어졌다는 점은 오류의 총량뿐 아니라 구조와 상관관계가 중요함을 뜻한다. Repetition-code 실험에서는 rare correlated burst로 인한 약 $10^{-10}$ per-cycle 수준의 error floor도 관찰했다.

### 2.3 Decoder 구성과 AlphaQubit 구분

논문은 목적에 따라 여러 decoder를 사용했다.

| Decoder | 역할 |
|---|---|
| Neural-network decoder | 가장 높은 offline accuracy, $\Lambda=2.14\pm0.02$ |
| Ensembled matching synthesis | 고정밀 offline matching ensemble |
| Correlated matching | Noise-model 및 error-budget 분석 |
| Real-time Sparse Blossom | 지속적인 syndrome stream을 처리하는 MWPM |

Willow 원문은 고정밀 decoder를 `neural-network decoder`라고 부르고 Bausch et al.의 AlphaQubit 연구를 인용한다. 별도 AlphaQubit 논문은 recurrent-transformer architecture를 AlphaQubit이라고 명명한다. 따라서 본 보고서에서는 이를 “Willow 원문 표기의 neural-network decoder; 관련 선행 architecture는 AlphaQubit 논문에 제시됨”으로 구분한다. 본 simulation의 PyMatching은 AlphaQubit을 구현하거나 재현한 것이 아니다.

### 2.4 Real-time decoding

72-qubit Willow의 distance-5 memory에는 real-time Sparse Blossom decoder가 통합됐다. 평균 software latency는 $63\pm17\,\mu s$, QEC cycle time은 $1.1\,\mu s$였고 최대 $10^6$ cycle을 처리했다. 이 latency는 마지막 syndrome 도착 후 correction을 반환하는 시간이며 완전한 end-to-end quantum feedback latency와 같지 않다. 또한 throughput은 syndrome 생성 속도를 지속해서 따라가는 능력이고 latency와 별개의 지표다.

## 3. 분석 방법

### 3.1 범위

공개 데이터 분석은 Willow actual/predicted bitstream에서 matching decoder의 지표를 수치적으로 재계산한다. 별도 synthetic simulation은 Stim rotated surface-code memory와 PyMatching MWPM으로 threshold 현상과 decoder-prior mismatch를 정성적으로 검증한다. Neural decoder·AlphaQubit과 real-time feedback은 실행하지 않았으며, synthetic 모델에는 Willow의 ZXXZ schedule, calibrated noise, leakage와 DQLR dynamics, analogue readout을 구현하지 않았다.

### 3.2 공개 Willow 데이터 재분석

Zenodo의 `google_105Q_surface_code_d3_d5_d7.zip`을 공식 크기와 MD5로 검증했다. $d=3$ 9 patches, $d=5$ 4 patches, $d=7$ 1 patch의 X/Z memory에서 `r13` calibration point를 제외한 364 main-sweep configurations을 사용했다. Actual observable과 decoder prediction의 XOR로 $p_L$을 계산하고, code·basis별로 $\ln(1-2p_L)$을 cycle 수에 회귀해 $\epsilon_d$를 구했다. 같은 distance의 code와 basis를 평균한 뒤 $\ln\epsilon_d$–distance fit으로 $\Lambda$를 계산했다. 제공 prediction 집계와 함께 noisy SI1000 circuit에서 DEM을 생성해 같은 detection events를 PyMatching으로 직접 decode했다. Direct 경로에서 $p_L\ge0.5$인 late-round point 4개는 log fit에서 제외하고 raw 결과에는 보존했다.

### 3.3 Baseline experiment

- Stage B local independent noise
- Distance $d=3,5,7$, X/Z memory
- `rounds=distance`
- $p=0.0015,0.003,0.006,0.007,0.008,0.009,0.010,0.015,0.020$
- 각 조건 20,000 fixed shots
- PyMatching decoding

Shot logical failure probability $p_L$은 다음 식으로 cycle당 logical error estimate로 변환했다.

$$
\epsilon=\frac12\left[1-(1-2p_L)^{1/r}\right].
$$

$\ln\epsilon_d$를 distance에 대해 선형 적합해 $\Lambda$를 계산하고, 인접한 $p$에서 $\Lambda=1$ crossing을 선형 보간해 pilot threshold를 구했다.

### 3.4 Correlated-ZZ experiment

Correlated extension은 Stage C $p=0.003$에서 인접 CZ edge에 ZZ-only surrogate를 추가했다. 세 조건을 비교했다.

1. Independent noise와 independent prior
2. Correlated noise와 independent, 즉 mismatched prior
3. 같은 correlated syndrome과 correlation-aware, 즉 matched prior

주 stress test는 `zz_factor=1.0`이며 각 조건은 50,000 shots다. Independent 조건의 local-noise parameter를 basis와 distance별로 조정해 모든 최종 비교에서 $|\Delta p_{det}|<0.0005$를 만족시켰다. Matched와 mismatched decoder는 같은 syndrome sample을 사용하므로 한 decoder만 실패한 shot을 이용해 exact paired test를 수행했다.

### 3.5 통계 분석

- 개별 logical error: Wilson 95% interval
- Threshold와 $\Lambda$: failure count의 binomial model에 대한 20,000회 parametric bootstrap
- Decoder comparison: discordant shot에 대한 two-sided exact paired test
- 다중 비교: Holm family-wise error correction
- Sensitivity: $d=3$, `zz_factor=0.05,0.25,0.5,1.0`, 각 50,000 shots

## 4. 결과

### 4.1 공개 Willow decoder scaling

| Decoder 경로 | $\epsilon_3$ | $\epsilon_5$ | $\epsilon_7$ | $\Lambda$ |
|---|---:|---:|---:|---:|
| Paper neural decoder | 0.00650 | 0.00303 | 0.00143 | 2.14 |
| Paper Libra, ensemble 101 | 0.00712 | 0.00349 | 0.00171 | 2.04 |
| Public Libra RL, ensemble 51 재집계 | 0.006802 | 0.003372 | 0.001652 | **2.029** |
| Direct PyMatching + SI1000 | 0.010438 | 0.007102 | 0.004485 | **1.525** |

Public Libra 재집계값은 논문 matching 기준과 가깝다. Direct PyMatching도 distance 증가에 따른 suppression을 보였지만 오류율이 더 높고 $\Lambda$가 낮았다. 본 프로젝트의 회귀 잔차 기반 간이 95% 구간은 각각 1.961–2.103과 1.476–1.578이며 논문의 drift-aware uncertainty와 동일하지 않다. Paper neural/Libra 수치는 reference로 표시한 것이며 neural decoder를 직접 실행한 결과가 아니다.

Direct/Public-Libra $\epsilon$ 비는 $d=3,5,7$에서 각각 1.53, 2.11, 2.71배였다. 즉 decoder 경로의 손실이 distance와 함께 커졌고, 그 결과 direct $\Lambda$는 Public Libra보다 24.8% 낮았다.

### 4.2 Baseline below-threshold scaling

낮은 $p$에서는 $\epsilon_7<\epsilon_5<\epsilon_3$였고 높은 $p$에서는 순서가 뒤집혔다.

| 항목 | X memory | Z memory |
|---|---:|---:|
| Pilot threshold | 0.008852 | 0.008089 |
| Bootstrap 95% interval | 0.008581–0.009188 | 0.007793–0.008526 |
| $p=0.003$의 $\Lambda$ | 2.551 [2.338, 2.801] | 2.364 [2.171, 2.589] |
| $p=0.006$의 $\Lambda$ | 1.300 [1.257, 1.345] | 1.225 [1.184, 1.267] |
| $p=0.009$의 $\Lambda$ | 0.990 [0.966, 1.015] | 0.963 [0.939, 0.987] |

이는 synthetic model에서 below-threshold scaling과 crossing을 정성적으로 재현한 결과다. Threshold 수치는 Stim에 입력한 합성 parameter에만 해당하며 실제 hardware threshold가 아니다.

### 4.3 Detection-matched correlated scaling

| Basis | Independent $\Lambda$ | Mismatched $\Lambda$ | Matched $\Lambda$ |
|---|---:|---:|---:|
| X | 1.880 [1.805, 1.958] | 1.374 [1.333, 1.417] | 1.400 [1.357, 1.444] |
| Z | 1.831 [1.754, 1.913] | 2.202 [2.090, 2.324] | 2.202 [2.090, 2.324] |

X memory에서는 correlated ZZ가 logical-error suppression을 약화시켰고 matched prior가 손실을 일부 줄였다. Z memory에서는 두 correlated decoder가 같은 결과를 냈으며, detection-matched independent 조건보다 $\Lambda$가 높았다. 이는 correlation의 logical 영향이 basis에 비대칭적임을 보여준다.

### 4.4 Paired decoder comparison

| Basis | Distance | Mismatched only | Matched only | Holm-adjusted $p$ |
|---|---:|---:|---:|---:|
| X | 3 | 128 | 82 | 0.00733 |
| X | 5 | 368 | 249 | $9.46\times10^{-6}$ |
| X | 7 | 439 | 303 | $4.03\times10^{-6}$ |
| Z | 3 | 1 | 1 | 1.0 |
| Z | 5 | 10 | 10 | 1.0 |
| Z | 7 | 5 | 5 | 1.0 |

X의 세 distance에서는 matched prior의 개선이 Holm correction 뒤에도 유의했다. Z에서는 net gain이 없었다. Marginal $\Lambda$ interval의 겹침과 달리 paired test가 decoder 차이를 검출할 수 있었던 이유는 두 decoder가 같은 sample을 사용했기 때문이다.

### 4.5 Correlation-strength sensitivity

$d=3$의 X memory에서 `zz_factor=0.05,0.25,0.5,1.0`에 대한 matched-prior net gain은 각각 2, 15, 19, 45 shots였다. 여덟 개 X/Z-factor comparison을 보정하면 `zz_factor=1.0`만 유의했다($p_{adj}=0.0181$). Z의 net gain은 모든 factor에서 0이었다. 따라서 현재 데이터는 강한 stress test에서의 decoder-prior 효과를 지지하지만 약한 correlation으로 일반화하지 못한다.

## 5. 논의

### 5.1 초기 가설과의 일치

Baseline 목표는 충족됐다. 낮은 $p$에서 larger distance가 logical error를 줄였고, 높은 $p$에서는 ordering이 뒤집혔으며, $\Lambda=1$ crossing도 관찰했다.

Correlated-noise 가설은 부분적으로 수정됐다. “같은 $p_{det}$의 correlated noise가 더 해롭다”는 예상은 X memory에서는 지지됐지만 Z memory에서는 지지되지 않았다. “Matched prior가 더 낫다”는 예상도 강한 ZZ stress test의 X memory에서만 지지됐다. 더 정확한 결론은 다음과 같다.

> 평균 detection probability가 같더라도 logical error suppression은 noise correlation의 연산자 방향, logical basis와 decoder prior의 적합성에 따라 달라진다.

### 5.2 원 논문과의 연결

공개 prediction 재집계는 Willow actual bitstream에서 matching 계열 $\Lambda$를 수치적으로 다시 계산한 결과다. Public Libra 2.029와 논문 Libra 2.04의 근접성은 fitting과 bitstream 해석을 검증한다. 반면 direct PyMatching 1.525와 synthetic 실험은 decoder prior와 correlation structure가 scaling을 바꾼다는 메커니즘 비교를 제공한다. Synthetic 결과는 gate schedule, noise calibration과 cycle 수가 달라 Willow 수치와 직접 비교하지 않는다.

### 5.3 시사점

- 평균 physical error 또는 $p_{det}$만으로 logical performance를 충분히 예측할 수 없다.
- X/Z 결과를 평균하면 directional correlation의 영향을 숨길 수 있다.
- Noise-aware decoding은 관련 correlation이 logical 판정을 실제로 바꾸는 경우에 효과가 나타난다.
- Decoder 개선이 correlated error의 피해를 모두 제거하지 못하므로 hardware mitigation과 병행해야 한다.

## 6. 한계

1. Public Libra는 archive의 ensemble 51 prediction이며 논문 Table S1의 ensemble 101과 다르다.
2. 공개 데이터의 간이 불확실성은 논문의 device-drift uncertainty를 완전히 재현하지 않는다.
3. Neural decoder·AlphaQubit은 직접 실행하지 않았다.
4. `zz_factor=1.0`은 실제 장치 보정값이 아닌 강한 synthetic surrogate다.
5. Detection matching의 independent $p$는 distance별로 달라 엄밀한 fixed-$p$ scaling이 아니다.
6. Synthetic bootstrap은 shot noise만 반영하고 graphlike decomposition은 correlated fault를 근사한다.
7. Sensitivity는 $d=3$에 한정되며 logical gate, leakage dynamics와 real-time feedback 성능은 다루지 않는다.

## 7. 결론

본 프로젝트는 Willow 공개 prediction에서 matching 계열 below-threshold scaling을 재계산하고, 동일 syndrome의 direct PyMatching 결과와 비교했다. Public Libra는 $\Lambda=2.029$, direct PyMatching+SI1000은 1.525로 모두 below threshold였지만 suppression의 크기는 달랐다. 단순화된 simulation에서도 synthetic threshold와 basis-dependent correlation 효과를 확인했다. 따라서 surface-code 확장성을 평가할 때 physical error의 양뿐 아니라 correlation 구조, logical basis와 decoder prior를 함께 분석해야 한다.

## 8. 재현 자료

- Experiment runner: `qec_below_threshold_review/run_experiment.py`
- Baseline analysis: `qec_below_threshold_review/plot_baseline.py`
- Correlated analysis: `qec_below_threshold_review/plot_correlated.py`
- Statistical analysis: `qec_below_threshold_review/analyze_statistics.py`
- Willow-data analysis: `qec_below_threshold_review/analyze_willow_data.py`
- Willow comparison plot: `qec_below_threshold_review/plot_willow_reproduction.py`
- Raw and derived artifacts: `artifacts/qec_below_threshold_review/`
- 확정 내용 기준: `qec_below_threshold_review/docs/content_master.md`

## 9. 참고문헌

1. Google Quantum AI and Collaborators, “Quantum error correction below the surface code threshold,” *Nature* 638, 920–926 (2025). <https://doi.org/10.1038/s41586-024-08449-y>
2. J. Bausch et al., “Learning high-accuracy error decoding for quantum processors,” *Nature* 635, 834–840 (2024). <https://doi.org/10.1038/s41586-024-08148-8>
3. Google Quantum AI and Collaborators, Supplementary Information and Author Correction for Ref. 1.
4. Google Quantum AI, “Data for ‘Quantum error correction below the surface code threshold’,” Zenodo. <https://doi.org/10.5281/zenodo.13273331>
