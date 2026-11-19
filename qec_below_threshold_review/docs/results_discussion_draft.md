# Results and Discussion 초안

## 1. 분석 범위

본 프로젝트는 두 분석 층을 구분한다. 첫째, Google Quantum AI가 공개한 Willow 105Q actual/predicted bitstream에서 논문의 fitting 절차로 matching-decoder $\epsilon_d$와 $\Lambda$를 수치적으로 다시 계산한다. 둘째, Stim rotated surface-code memory와 synthetic noise를 이용해 threshold와 correlation 효과를 통제된 조건에서 정성적으로 검증한다.

1. 물리 noise parameter가 threshold 아래일 때 code distance 증가에 따라 cycle당 논리 오류율이 감소하는가?
2. 평균 detection probability가 비슷해도 spatially correlated ZZ noise와 decoder-prior mismatch가 logical scaling을 바꾸는가?

원 논문의 neural decoder·AlphaQubit과 streaming decoder는 실행하지 않았다. Synthetic 실험에는 ZXXZ schedule, calibrated device noise와 leakage dynamics를 포함하지 않았다.

## 2. 공개 Willow 105Q 데이터 재현

Zenodo archive의 364 main-sweep configurations에 대해 code·basis별로 $\ln(1-2p_L)$을 QEC cycle에 선형 회귀하고, distance별 $\epsilon_d$를 평균한 뒤 $\ln\epsilon_d$–distance fit으로 $\Lambda$를 계산했다. `r13`은 RL prior calibration에 사용된 point이므로 제외했다.

| Decoder 경로 | $\epsilon_3$ | $\epsilon_5$ | $\epsilon_7$ | $\Lambda$ |
|---|---:|---:|---:|---:|
| Paper neural decoder | 0.00650 | 0.00303 | 0.00143 | 2.14 |
| Paper Libra, ensemble 101 | 0.00712 | 0.00349 | 0.00171 | 2.04 |
| Public Libra RL, ensemble 51 재집계 | 0.006802 | 0.003372 | 0.001652 | **2.029** |
| Direct PyMatching + SI1000 | 0.010438 | 0.007102 | 0.004485 | **1.525** |

Public Libra의 $\Lambda=2.029$는 논문 matching 기준 2.04와 가깝다. Ensemble size와 uncertainty 절차가 달라 완전 일치를 기대하지 않지만, 공개 prediction에서 핵심 scaling을 독립적으로 다시 얻었다. 동일한 detection events를 noisy SI1000 circuit에서 만든 DEM으로 직접 PyMatching decode하면 모든 distance에서 오류율이 높고 $\Lambda$가 1.525로 낮았다. 두 경로 모두 $\Lambda>1$이므로 below-threshold ordering은 유지되지만, decoder와 prior가 suppression의 크기를 크게 바꾼다.

Direct 경로의 세 $d=3$ code·basis에서 late-round point 4개가 $p_L\ge0.5$여서 log-fidelity fit에서 제외됐다. 제외 전 값은 raw CSV에 유지했다.

본 프로젝트의 간이 95% 구간은 Public Libra 1.961–2.103, direct PyMatching 1.476–1.578이다. 이는 code·basis 회귀 잔차를 전파한 값이며, 장치 drift의 excess residual까지 반영한 논문의 uncertainty와 직접 비교하지 않는다. 또한 공개 neural prediction은 제공되지 않아 neural $\Lambda=2.14$나 AlphaQubit을 재현한 결과가 아니다.

## 3. Baseline below-threshold scaling

Stage B local independent noise에서 $d=3,5,7$, X/Z memory를 각각 20,000 shots로 평가했다. 각 실험은 `rounds=distance`를 사용했기 때문에 distance별 shot failure probability를 직접 비교하지 않고 다음 식으로 cycle당 논리 오류율을 계산했다.

$$
\epsilon_d=\frac{1}{2}\left[1-(1-2p_L)^{1/d}\right].
$$

낮은 noise 영역에서는 $\epsilon_7<\epsilon_5<\epsilon_3$가 나타났다. 반대로 높은 noise 영역에서는 distance가 증가할수록 $\epsilon_d$가 증가하여 threshold crossing을 확인했다.

$\ln\epsilon_d$를 distance에 대해 선형 적합해 계산한 $\Lambda$는 다음과 같다.

| $p$ | X $\Lambda$ | Z $\Lambda$ | 판정 |
|---:|---:|---:|---|
| 0.003 | 2.551 | 2.364 | below threshold |
| 0.006 | 1.300 | 1.225 | below threshold |
| 0.007 | 1.129 | 1.093 | below threshold |
| 0.008 | 1.056 | 1.004 | crossing 근처 |
| 0.009 | 0.990 | 0.963 | threshold 이상 |
| 0.010 | 0.931 | 0.915 | threshold 이상 |

두 인접 점 사이를 선형 보간한 pilot threshold는 X memory에서 0.008852, Z memory에서 0.008089였다. 20,000회 parametric binomial bootstrap의 95% interval은 각각 0.008581–0.009188과 0.007793–0.008526이었다. 이 값은 Stim noise instruction에 입력한 합성 parameter의 threshold이며 실제 Willow processor의 physical error threshold가 아니다.

이 결과는 프로젝트의 첫 번째 목표인 below-threshold scaling을 정성적으로 재현한다. 특히 $p=0.003$에서 $\Lambda>2$가 얻어져 code distance가 2 증가할 때 cycle당 논리 오류율이 절반 이하로 감소했다.

## 4. Correlated-ZZ stress test

Correlated 조건은 baseline threshold보다 충분히 낮은 $p=0.003$에서 수행했다. 인접 CZ edge에 ZZ surrogate를 추가하고 `zz_factor=1.0`을 사용했다. 이 값은 decoder-prior 차이를 관찰하기 위한 강한 stress-test parameter이며 실제 하드웨어 보정값이 아니다.

Stage B의 independent parameter를 basis와 distance별로 조정하여 모든 비교에서

$$
|p_{det}^{corr}-p_{det}^{ind}|<0.0005
$$

를 만족시켰다. 각 최종 조건은 50,000 fixed shots로 평가했다.

### 4.1 X memory

| Distance | Independent $\epsilon$ | Corr. mismatched $\epsilon$ | Corr. matched $\epsilon$ |
|---:|---:|---:|---:|
| 3 | 0.010144 | 0.013445 | 0.013121 |
| 5 | 0.004733 | 0.009732 | 0.009218 |
| 7 | 0.002869 | 0.007117 | 0.006694 |

X memory에서 correlated ZZ는 모든 distance의 logical error를 높이고 scaling을 약화시켰다. Fitted suppression factor는 independent condition의 1.880에서 mismatched prior의 1.374로 감소했다. Correlated DEM을 사용한 matched prior에서는 1.400으로 일부 회복했지만 independent scaling에는 미치지 못했다.

Matched decoder의 오류율 감소는 mismatched decoder 대비 distance 3, 5, 7에서 각각 약 2.4%, 5.3%, 5.9%였다.

### 4.2 Z memory

| Distance | Independent $\epsilon$ | Corr. mismatched $\epsilon$ | Corr. matched $\epsilon$ |
|---:|---:|---:|---:|
| 3 | 0.008669 | 0.007073 | 0.007073 |
| 5 | 0.004459 | 0.003082 | 0.003082 |
| 7 | 0.002585 | 0.001458 | 0.001458 |

Z memory에서는 두 correlated decoder의 결과가 같았고 $\Lambda=2.202$를 얻었다. 이는 ZZ error가 logical basis에 따라 비대칭적으로 작용한 결과다. Detection probability를 맞추기 위해 independent condition의 local-noise parameter를 높였지만, 추가한 Z-type correlation은 Z memory의 logical observable에는 상대적으로 덜 해로웠다.

따라서 이 결과는 correlated noise가 일반적으로 성능을 개선한다는 뜻이 아니다. 같은 평균 detection probability라도 error type과 logical basis의 관계에 따라 logical damage가 달라진다는 뜻이다.

### 4.3 Suppression factor의 통계적 불확실성

Distance별 failure count를 binomial model로 다시 표본화한 20,000회 parametric bootstrap 결과는 다음과 같다.

| Basis | Independent $\Lambda$ | Mismatched $\Lambda$ | Matched $\Lambda$ |
|---|---:|---:|---:|
| X | 1.880 [1.805, 1.958] | 1.374 [1.333, 1.417] | 1.400 [1.357, 1.444] |
| Z | 1.831 [1.754, 1.913] | 2.202 [2.090, 2.324] | 2.202 [2.090, 2.324] |

대괄호는 bootstrap 95% interval이다. X의 mismatched와 matched interval이 겹치므로 decoder 간 차이는 동일 sample을 사용한 paired test로 판단한다.

## 5. Paired decoder-prior comparison

Mismatched와 matched decoder는 같은 correlated syndrome sample을 해독했다. 이에 따라 독립적인 두 오류율의 차이가 아니라, 두 decoder가 서로 다른 결과를 낸 shot만 이용한 paired comparison이 가능하다.

| Basis | Distance | Mismatched only | Matched only | Raw exact $p$ | Holm-adjusted $p$ |
|---|---:|---:|---:|---:|---:|
| X | 3 | 128 | 82 | 0.00183 | 0.00733 |
| X | 5 | 368 | 249 | $1.89\times10^{-6}$ | $9.46\times10^{-6}$ |
| X | 7 | 439 | 303 | $6.72\times10^{-7}$ | $4.03\times10^{-6}$ |
| Z | 3 | 1 | 1 | 1.0 | 1.0 |
| Z | 5 | 10 | 10 | 1.0 | 1.0 |
| Z | 7 | 5 | 5 | 1.0 | 1.0 |

X memory에서는 세 distance 모두 matched prior가 더 많은 오류를 고쳤고, 여섯 개 X/Z-distance 비교에 Holm correction을 적용한 뒤에도 모두 유의했다. Z memory에서는 차이가 없었다. 이는 강한 synthetic stress test의 X basis에서 noise correlation을 decoder graph에 반영하는 것이 유효했음을 보여주지만, 다른 correlation strength나 noise model로 일반화하지는 않는다.

### 5.1 Correlation-strength sensitivity

$d=3$, 50,000 shots에서 `zz_factor=0.05,0.25,0.5,1.0`을 비교했다. X memory의 matched-prior net gain은 각각 2, 15, 19, 45 shots였다. 여덟 개 X/Z-factor 비교에 Holm correction을 적용하면 `zz_factor=1.0`만 유의했다($p_{adj}=0.0181$). Z memory의 net gain은 모든 factor에서 0이었다.

따라서 decoder-prior 개선은 현재 데이터상 **강한 correlated-ZZ stress test에서 확인된 결과**이며, 약한 correlation에서도 유지된다고 말할 수 없다.

## 6. 원 논문과의 비교

원 논문은 Willow hardware와 neural-network decoder에서 $\Lambda=2.14\pm0.02$, $\epsilon_7=(1.43\pm0.03)\times10^{-3}$를 보고했다. 본 baseline도 낮은 $p$에서 $\Lambda>2$를 보였지만 수치가 비슷하다는 사실 자체에는 재현의 의미가 없다.

직접 비교할 수 없는 이유는 다음과 같다.

- 본 구현은 Stim의 rotated memory scaffold이며 Willow의 ZXXZ gate schedule이 아니다.
- noise parameter $p$는 calibrated component error가 아닌 합성 scale parameter다.
- 본 구현은 leakage, DQLR, drift와 rare burst를 포함하지 않는다.
- PyMatching MWPM은 Willow 논문의 neural-network decoder, AlphaQubit 또는 ensemble decoder를 구현한 것이 아니다.
- 원 논문은 여러 하위 격자와 최대 250-cycle fitting을 사용하지만, 본 구현은 `rounds=distance` memory shot을 사용한다.

그럼에도 두 연구는 정성적으로 같은 두 결론을 지지한다.

첫째, threshold 아래에서는 code distance 증가가 logical error를 억제한다. 둘째, detection probability 같은 평균적인 syndrome 지표만으로 logical 성능을 예측할 수 없다. 원 논문에서는 DQLR이 detection probability를 약 12% 줄이면서 $\Lambda$를 약 35% 높였고, 본 stress test에서는 $p_{det}$를 맞춘 뒤에도 correlation structure와 logical basis에 따라 $\Lambda$가 크게 달라졌다.

## 7. 해석과 시사점

### 7.1 Threshold는 noise strength만의 숫자가 아니다

같은 $p_{det}$에서도 independent와 correlated model의 logical error가 달랐다. 따라서 실제 QEC 성능은 오류 발생량뿐 아니라 오류의 공간·시간 구조에 의존한다.

### 7.2 Decoder prior mismatch는 distance와 함께 중요해질 수 있다

X memory에서 matched decoder의 net gain은 $d=3,5,7$에 대해 46, 119, 136 shots로 증가했다. 더 큰 code에서는 syndrome graph가 커지고 가능한 correlated pattern도 많아지므로 noise-aware weight의 가치가 커질 가능성이 있다. 현재 세 distance만으로 일반적인 scaling law를 주장하지는 않는다.

### 7.3 Basis별 결과를 분리해야 한다

X에서는 correlated ZZ가 성능을 악화시켰지만 Z에서는 같은 효과가 나타나지 않았다. X/Z 결과를 단순 평균하면 noise 방향성에 대한 핵심 정보를 잃을 수 있다.

### 7.4 Hardware mitigation과 decoder 개선은 상호 보완적이다

Matched prior는 X-memory 손실을 일부 줄였지만 independent condition의 성능을 회복하지 못했다. Correlated error는 decoder만으로 모두 해결하기보다 hardware-level suppression과 함께 다뤄야 한다.

## 8. 초기 가설에 대한 평가

초기 연구 목표는 “단순화된 rotated surface-code simulation에서 below-threshold scaling을 확인하고, detection probability를 맞춘 independent와 correlated noise에서 logical suppression과 decoder-prior mismatch를 분석한다”는 것이었다.

### 8.1 지지된 가설

- 낮은 $p$에서 distance가 커질수록 $\epsilon_d$가 감소했다.
- 높은 $p$에서 curve ordering이 뒤집혀 threshold crossing이 나타났다.
- $p$가 낮아질수록 fitted $\Lambda$가 증가했다.
- 평균 $p_{det}$가 비슷해도 error structure에 따라 logical error와 $\Lambda$가 달랐다.
- X memory에서는 matched decoder가 mismatched decoder보다 일관되게 낮은 오류율을 보였다.

### 8.2 부분적으로만 지지된 가설

초기에는 correlated noise가 같은 $p_{det}$의 independent noise보다 logical error를 전반적으로 높일 것으로 예상했다. 이 예상은 X memory에서는 맞았지만 Z memory에서는 맞지 않았다. Pure ZZ surrogate가 logical basis에 비대칭적으로 작용했기 때문이다.

또한 correlation-aware prior의 개선도 X memory에서만 나타났고 Z memory에서는 두 prior의 성능이 같았다. 따라서 decoder-prior matching의 효과는 보편적인 상수가 아니라 noise channel과 logical basis에 의존한다.

### 8.3 수정된 결론

실험 결과가 지지하는 더 정확한 가설은 다음과 같다.

> **동일한 평균 detection probability에서도 correlated noise의 logical 영향은 correlation 방향과 logical basis에 따라 달라지며, noise-aware decoder의 이점은 해당 correlation이 decoder의 logical 판정을 바꾸는 경우에 나타난다.**

따라서 프로젝트의 핵심 목표는 달성됐지만, 확장 실험의 결론은 “correlation은 항상 더 해롭다”가 아니라 “correlation의 구조와 basis를 함께 분석해야 한다”로 수정해야 한다.

## 9. 한계

1. Public Libra archive prediction은 ensemble 51이며 논문 Table S1의 ensemble 101과 다르다.
2. 공개 데이터의 간이 불확실성은 논문의 drift-aware uncertainty를 완전히 재현하지 않는다.
3. Neural decoder·AlphaQubit은 직접 실행하지 않았다.
4. `zz_factor=1.0`은 의도적으로 큰 surrogate parameter다.
5. Detection matching을 위해 Stage B의 $p$를 distance마다 변경했으므로 이 조건의 $\Lambda$는 엄밀한 fixed-$p$ scaling이 아니다.
6. Parametric bootstrap은 calibration, noise-model, seed와 finite-distance uncertainty를 포함하지 않는다.
7. Correlated DEM은 원래 fault를 graphlike component로 근사하며 sensitivity 분석은 $d=3$에 한정된다.
8. 본 결과는 memory experiment에 한정되며 logical gate 또는 real-time feedback 성능을 나타내지 않는다.

## 10. 결론 초안

Willow 공개 prediction을 재집계해 Public Libra $\Lambda=2.029$를 얻었으며, 이는 논문 matching 기준 2.04와 가깝다. 동일 syndrome의 direct PyMatching+SI1000은 $\Lambda=1.525$로, 모든 경로가 below threshold이면서도 decoder 선택이 scaling을 크게 바꿈을 보였다. 별도 Stim–PyMatching simulation에서는 합성 noise parameter 약 0.008–0.009 아래의 threshold scaling과 basis-dependent correlation 효과를 확인했다. 이 실제-data 재계산과 통제된 synthetic 결과는 surface-code 확장성에 hardware error뿐 아니라 correlation structure와 decoder prior의 적합성이 중요하다는 결론을 함께 지지한다.

## 11. Figure caption 초안

### Figure 1. Willow public-data decoder scaling

Willow 105Q 공개 archive의 main sweep에서 계산한 cycle당 logical error. Public Libra prediction 재집계는 논문 matching 기준과 가까운 $\Lambda=2.029$를 보였고 direct PyMatching+SI1000은 $\Lambda=1.525$였다. Paper neural/Libra 값은 Table S1의 reference이며 직접 재실행값이 아니다.

### Figure 2. Baseline threshold scaling

Stage B local independent noise에서 측정한 cycle당 logical error rate. 각 점은 20,000 shots이며 error bar는 shot logical failure의 Wilson 95% interval을 cycle당 오류율로 변환한 값이다. 낮은 $p$에서는 code distance 증가에 따라 $\epsilon_d$가 감소하지만 높은 $p$에서는 순서가 뒤집힌다. $\ln\epsilon_d$의 distance fit으로 추정한 crossing은 X memory에서 0.00885, Z memory에서 0.00809이다.

### Figure 3. Detection-probability-matched correlated-ZZ scaling

$p=0.003$, `zz_factor=1.0` correlated-ZZ stress test의 cycle당 logical error rate. Independent condition은 basis와 distance별로 $p$를 조정하여 correlated condition과 $|\Delta p_{det}|<0.0005$가 되도록 했다. X memory에서 correlated noise는 $\Lambda$를 낮췄으며 correlation-aware DEM이 mismatch 손실을 일부 완화했다. Z memory에서는 matched와 mismatched prior가 동일한 성능을 보였다. 각 조건은 50,000 shots이다.
