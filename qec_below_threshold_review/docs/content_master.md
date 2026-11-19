# 논문 리뷰 프로젝트 내용 기준본

> 목적: 포스터·보고서·발표 대본을 만들기 전에 **무엇을 말할지** 확정하는 단일 기준 문서\
> 범위: 내용의 정확성, 중요도와 논리 구조만 다룸. 배치·색상·크기 등 포스터 디자인은 다루지 않음.
> 수식의 단계별 도출은 [`paper_equations_explained.md`](paper_equations_explained.md)를 참고한다.

## 0. 먼저 알아둘 핵심 용어

| 용어 | 쉬운 뜻 |
|---|---|
| Physical qubit | 실제 장치에 있는 큐비트. 자체적으로 오류가 잘 생긴다. |
| Logical qubit | 여러 물리 큐비트를 묶어 오류에 더 강하게 만든 가상의 큐비트 |
| Code distance $d$ | 오류를 견디는 코드의 크기. 여기서는 3, 5, 7을 비교한다. |
| Syndrome | 큐비트를 직접 읽지 않고 얻는 **오류의 흔적** |
| Decoder | 오류의 흔적을 보고 최종 논리값을 추정하는 프로그램 |
| Threshold | 이 값보다 장치 오류가 낮아야 코드를 키울수록 성능이 좋아지는 경계 |
| $\epsilon_d$ | distance $d$에서 QEC 한 cycle당 발생하는 논리 오류율 |
| $\Lambda$ | 코드를 $d$에서 $d+2$로 키웠을 때 오류가 몇 배 줄어드는지 나타내는 값. $\Lambda>1$이면 개선된다. |
| Correlated noise | 여러 큐비트에 서로 연관되어 함께 생기는 오류 |
| ZZ-only surrogate | 실제 상관 오류를 단순하게 흉내 낸 모델. 연결된 두 큐비트에 Z 오류가 동시에 생기게 한다. |
| Decoder prior | 디코더가 계산할 때 미리 가정하는 오류의 종류와 발생 가능성 |
| DEM | 어떤 물리 오류가 어떤 검출 신호를 만드는지 적은 오류 설명서(Detector Error Model) |
| Shot | 같은 실험을 반복한 횟수 중 한 번 |

### 세 줄 요약

1. 공개 Willow 데이터에서 code distance가 커질수록 논리 오류가 감소함을 다시 확인했다.
2. 같은 오류 측정값을 사용해도 디코더에 따라 $\Lambda$가 2.029와 1.525로 달라졌다.
3. 별도 모의실험에서는 오류의 평균 개수뿐 아니라 오류가 함께 생기는 방식과 디코더의 가정도 중요했다.

## 1. 프로젝트의 중심 질문

### 논문 리뷰 질문

> Willow 실험에서는 code distance를 키웠을 때 실제로 논리 오류가 줄었는가? 앞으로 코드를 더 키울 때 어떤 오류와 디코더 문제가 남는가?

### 시뮬레이션 질문

> 단순화한 surface-code 모의실험에서도 코드를 키울수록 오류가 줄어드는가? 평균 오류 신호의 양이 비슷해도, 오류가 함께 생기는 방식과 디코더가 가정한 오류 모델에 따라 결과가 달라지는가?

### 공개 데이터 재현 질문

> 공개된 Willow $d=3,5,7$ 오류 신호를 다시 분석해 cycle당 논리 오류율과 $\Lambda$를 계산할 수 있는가? 같은 데이터를 PyMatching으로 직접 처리하면 제공된 결과와 얼마나 다른가?

## 2. 최종 핵심 주장

이 프로젝트에서 최종적으로 주장할 내용은 다음 네 가지다.

1. **공개 Willow 데이터에서 오류 억제를 다시 계산했다.**\
   공개 Libra 결과에서 $\Lambda=2.029$를 얻어 논문의 matching-decoder 값 2.04와 가까운 결과를 확인했다.

2. **단순화한 모의실험에서도 threshold 현상을 확인했다.**\
   가상 오류율이 낮을 때 $d=3\rightarrow5\rightarrow7$로 코드를 키우면 cycle당 논리 오류가 감소했다. 가상 오류율이 높을 때는 큰 코드의 이점이 사라졌다.

3. **평균 오류 신호의 양만으로 최종 피해를 설명할 수 없다.**\
   검출 이벤트 비율 $p_{det}$를 비슷하게 맞춰도, 독립적으로 생긴 오류와 서로 연관되어 생긴 오류의 $\epsilon$과 $\Lambda$가 달랐다.

4. **상관 오류와 디코더 오류 모델의 효과는 X/Z 실험 종류에 따라 달랐다.**\
   X memory에서는 동시에 생기는 ZZ 오류가 억제 성능을 낮췄고, 그 오류를 아는 디코더가 일부 개선했다. Z memory에서는 두 디코더의 결과가 같았다.

다음과 같이 일반화해서는 안 된다.

- “Correlated noise는 항상 logical error를 높인다.”
- “Correlation-aware decoder는 모든 basis에서 항상 더 좋다.”
- “단순화한 모의실험이 Willow 장치의 수치를 재현했다.”

## 3. 전체 이야기 구조

### 3.1 왜 QEC가 필요한가

- 물리 큐비트는 오류가 잦고 실용 알고리즘이 요구하는 정확도에 미치지 못한다.
- 여러 물리 큐비트로 logical qubit을 구성하고 오류의 흔적인 syndrome을 반복 측정한다.
- Decoder는 이 흔적을 보고 원래 논리값을 추정한다.

### 3.2 Threshold가 왜 핵심인가

$$
\epsilon_d\propto
\left(\frac{p}{p_{thr}}\right)^{(d+1)/2},
\qquad
\Lambda=\frac{\epsilon_d}{\epsilon_{d+2}}.
$$

- $\Lambda>1$: distance 증가가 logical error를 억제
- $\Lambda=1$: crossing 또는 threshold 부근
- $\Lambda<1$: 더 큰 code의 이점 소실

### 3.3 Willow 논문의 답

- 실제 초전도 processor에서 $d=3,5,7$ scaling을 관찰했다.
- Neural-network decoder 기준 $\Lambda=2.14\pm0.02$였다.
- $d=7$에서 $\epsilon_7=(1.43\pm0.03)\times10^{-3}$/cycle이었다.
- 101-qubit distance-7 logical memory가 최상 구성 물리 큐비트보다 $2.4\pm0.3$배 긴 수명을 보였다.
- 실시간 decoder를 통합한 distance-5 memory도 below-threshold 성능을 유지했다.
- Leakage, stray interaction과 rare correlated burst는 scaling을 제한할 수 있다.

### 3.4 본 프로젝트의 답

- 공개 Willow prediction을 다시 계산해 matching decoder의 $\Lambda$를 확인했다.
- 별도 Stim/PyMatching 모의실험으로 threshold 현상을 확인했다.
- 실제 장치의 복잡한 상관 오류 대신, 연결된 두 큐비트에 Z 오류가 동시에 생기는 단순 모델(ZZ-only surrogate)을 사용했다.
- 처음의 “상관 오류는 항상 더 해롭다”는 예상은 “오류의 방향, X/Z 실험 종류와 디코더의 가정이 함께 중요하다”로 수정했다.

## 4. 논문에서 반드시 다룰 사실

| 주제 | 확정 내용 | 중요도 |
|---|---|---|
| Code와 hardware | 105-qubit Willow에서 101개 qubit을 사용하는 $d=7$ memory, 72-qubit Willow에서 $d=5$와 real-time decoding | 필수 |
| Stabilizer 표기 | 원문과 SI는 ZXXZ surface code라고 표기 | 필수 |
| Scaling | Neural decoder $\Lambda=2.14\pm0.02$ | 필수 |
| Logical error | $d=7$, $(1.43\pm0.03)\times10^{-3}$/cycle | 필수 |
| Beyond breakeven | $291\pm6\,\mu s$, 최상 물리 큐비트 대비 $2.4\pm0.3$배 | 필수 |
| DQLR | $p_{det}$ 약 12% 감소에 비해 $\Lambda$ 약 35% 증가 | 필수: 평균 error 지표의 한계와 연결 |
| Real-time | $d=5$, 평균 software latency $63\pm17\,\mu s$, cycle time $1.1\,\mu s$, 최대 $10^6$ cycle | 필수 |
| Error floor | Repetition code $d\le29$, 약 $10^{-10}$/cycle floor, rare burst 약 시간당 1회 | 선택: 남은 한계 설명 |
| Extrapolation | $10^{-6}$ logical error에 $d=27$, 1,457 qubit 필요 추정 | 선택: future scale 설명 |
| Author correction | Fig. 3a의 repetition-code 축과 범례 표기 수정, 핵심 수치 변화 없음 | 보고서에는 포함, 짧은 발표에서는 생략 가능 |

## 5. AlphaQubit과 Willow neural decoder의 관계

이 항목은 혼동하기 쉬우므로 다음처럼 구분한다.

### 확인된 사실

- 별도 Nature 논문인 Bausch et al., *Learning high-accuracy error decoding for quantum processors*는 recurrent-transformer decoder를 **AlphaQubit**이라고 명명한다.
- AlphaQubit 논문은 Sycamore의 $d=3,5$ experimental data와 최대 $d=11$ simulation을 다룬다.
- Willow below-threshold 논문은 Bausch et al.을 인용하며 고정밀 결과에 **neural network decoder**라는 표현을 사용한다.
- Willow 본문과 보관된 SI의 검색 가능한 텍스트에는 `AlphaQubit`이라는 고유명칭이 직접 등장하지 않는다.
- Willow SI의 neural decoder는 synthetic pretraining, device-specific DEM 및 experimental-data fine-tuning과 장시간 decoding용 gated recurrence를 사용한다.

### 권장 표현

> **Willow 논문이 AlphaQubit 연구를 인용해 사용하는 neural-network decoder 계열**

또는 더 엄밀하게:

> **Willow 원문 표기인 neural-network decoder; 관련 선행 architecture는 AlphaQubit 논문에 제시됨**

### 피해야 할 표현

- “Willow 논문의 decoder 이름은 AlphaQubit이다.”
- “본 프로젝트에서 AlphaQubit을 구현했다.”
- “PyMatching 결과가 AlphaQubit 성능을 재현한다.”

본 프로젝트는 AlphaQubit을 **논문 리뷰 내용으로 설명**하지만 simulation에는 구현하지 않는다.

## 6. Decoder 관련 필수 내용

논문은 하나의 decoder만 사용하지 않았다.

| Decoder 계열 | 역할 |
|---|---|
| Neural-network decoder | 가장 높은 offline accuracy, $\Lambda=2.14\pm0.02$ |
| Ensembled matching synthesis | 고정밀 offline matching ensemble, $\Lambda=2.04\pm0.02$ |
| Correlated matching | Noise model과 error-budget 분석 등에 사용 |
| Real-time Sparse Blossom | 낮은 latency와 지속 throughput을 위한 streaming MWPM |

정확도와 실시간성은 분리해 설명한다.

- Offline neural decoder: 더 높은 accuracy
- 실시간 디코더: 정확도는 조금 낮지만 오류 신호가 쌓이지 않도록 계속 처리한다.
- $63\,\mu s$는 마지막 오류 신호를 받은 뒤 결과를 내놓는 프로그램 실행 시간이다. 전체 양자 피드백에 걸리는 시간과는 다르다.
- 처리량은 초당 얼마나 많이 처리하는지, 지연시간은 한 결과가 나오는 데 얼마나 걸리는지를 뜻한다.

## 7. 단순화한 모의실험 방법

### 기본 실험: threshold 확인

- Stim rotated surface-code memory
- 각 큐비트에 오류가 서로 독립적으로 생기는 기본 모델(Stage B)
- $d=3,5,7$과 X/Z memory를 각각 비교
- `rounds=distance`
- $p=\{0.0015,0.003,0.006,0.007,0.008,0.009,0.01,0.015,0.02\}$
- 각 조건을 20,000번 반복
- PyMatching MWPM
- 측정 오차 범위를 나타내는 Wilson 95% 구간

### 추가 실험: 함께 생기는 오류의 영향

- Stage C $p=0.003$, `zz_factor=1.0`
- 연결된 두 큐비트에 Z 오류가 동시에 생기는 단순 상관 오류 모델 사용
- 다른 종류의 동시 오류는 제외하여 ZZ 효과에 집중
- 세 조건 비교: 독립 오류, 상관 오류를 모르는 디코더, 상관 오류를 아는 디코더
- 두 디코더에는 완전히 같은 오류 측정값을 입력
- 각 조건을 50,000번 반복
- 세 조건의 평균 검출 이벤트 비율 차이를 0.0005보다 작게 맞춤

### 평가 지표

$$
\epsilon=\frac12\left[1-(1-2p_L)^{1/r}\right].
$$

- $p_L$: 한 번의 memory 실험이 최종적으로 실패할 확률
- $\epsilon$: QEC 한 cycle당 논리 오류율
- $\Lambda$: 코드를 2단계 키웠을 때 논리 오류가 몇 배 줄었는지 나타내는 값
- $p_{det}$: 전체 오류 검출 신호 중 오류가 감지된 비율
- 한쪽 디코더만 실패한 횟수: 같은 입력에서 두 디코더의 차이를 직접 세는 값

## 8. 확정 결과

### 공개 Willow 105Q 데이터 재현

공개 데이터의 주요 364개 실험 조건에 논문과 같은 계산 방법을 적용했다. 여러 cycle에서 측정한 오류율을 이용해 distance별 $\epsilon$을 구한 뒤 $\Lambda$를 계산했다.

| 경로 | $\epsilon_3$ | $\epsilon_5$ | $\epsilon_7$ | $\Lambda$ |
|---|---:|---:|---:|---:|
| Paper neural decoder | 0.00650 | 0.00303 | 0.00143 | 2.14 |
| Paper Libra, ensemble 101 | 0.00712 | 0.00349 | 0.00171 | 2.04 |
| Public Libra RL, ensemble 51 재집계 | 0.006802 | 0.003372 | 0.001652 | **2.029** |
| Direct PyMatching + SI1000 | 0.010438 | 0.007102 | 0.004485 | **1.525** |

- 공개 Libra 재계산값 2.029는 논문의 matching 계열 값 2.04와 가깝다.
- 직접 실행한 PyMatching도 $\Lambda>1$이므로 코드를 키우면 오류가 줄었다. 다만 감소 폭은 Libra보다 작았다.
- PyMatching/Libra 오류율 비는 $d=3,5,7$에서 1.53, 2.11, 2.71배로 커졌다. 코드가 커질수록 디코더 차이가 더 크게 나타났다.
- 직접 PyMatching 결과 중 오류율이 50% 이상인 마지막 cycle 자료 4개는 로그 계산이 불가능해 곡선 맞추기에서만 제외했다. 원본 결과표에는 남겨 두었다.
- 공개 데이터에는 neural decoder 예측이 없으므로 논문의 2.14를 직접 재현한 것은 아니다.

### 기본 threshold 실험

| 항목 | X | Z |
|---|---:|---:|
| Pilot threshold | 0.008852 | 0.008089 |
| Threshold bootstrap 95% interval | 0.008581–0.009188 | 0.007793–0.008526 |
| $p=0.003$의 $\Lambda$ | 2.551 | 2.364 |
| $p=0.006$의 $\Lambda$ | 1.300 | 1.225 |
| $p=0.009$의 $\Lambda$ | 0.990 | 0.963 |

해석:

- 낮은 $p$에서 distance 증가가 error를 억제했다.
- 높은 $p$에서 curve ordering이 뒤집혔다.
- Threshold 값은 synthetic parameter에 대한 값이며 hardware threshold가 아니다.

### 평균 검출 신호를 맞춘 상관 오류 실험

| Basis | Independent $\Lambda$ | Mismatched $\Lambda$ | Matched $\Lambda$ |
|---|---:|---:|---:|
| X | 1.880 | 1.374 | 1.400 |
| Z | 1.831 | 2.202 | 2.202 |

Parametric bootstrap 95% interval:

| Basis | Independent | Mismatched | Matched |
|---|---:|---:|---:|
| X | 1.805–1.958 | 1.333–1.417 | 1.357–1.444 |
| Z | 1.754–1.913 | 2.090–2.324 | 2.090–2.324 |

X decoder paired result:

| Distance | Mismatched only | Matched only | Net gain | Exact paired $p$ |
|---:|---:|---:|---:|---:|
| 3 | 128 | 82 | 46 | 0.00183 |
| 5 | 368 | 249 | 119 | $1.89\times10^{-6}$ |
| 7 | 439 | 303 | 136 | $6.72\times10^{-7}$ |

Z에서는 exclusive failure 수가 양쪽 decoder에서 같아 net gain이 0이었다.

6개 X/Z-distance comparison에 Holm correction을 적용한 뒤 X의 adjusted $p$는 각각 0.00733, $9.46\times10^{-6}$, $4.03\times10^{-6}$였고 Z는 모두 1.0이었다.

### 상관 오류 세기를 바꾼 추가 확인

$d=3$에서 상관 오류의 세기 `zz_factor=0.05,0.25,0.5,1.0`을 각각 50,000번 시험했다. X memory에서 상관 오류를 아는 디코더가 추가로 성공한 횟수는 2, 15, 19, 45회였다. 여러 조건을 동시에 검사한 효과를 보정하면 가장 강한 `zz_factor=1.0`에서만 통계적으로 분명한 차이가 남았다. Z memory에서는 차이가 없었다.

따라서 **상관 오류를 아는 디코더가 더 좋다는 결과는 강한 ZZ 오류를 넣은 시험에서만 확인됐다.** 약한 상관 오류에서도 항상 좋아진다고 말할 수는 없다.

## 9. 초기 가설 평가

| 가설 | 결과 |
|---|---|
| 낮은 $p$에서 larger distance가 logical error를 억제 | 지지 |
| 높은 $p$에서 threshold crossing 발생 | 지지 |
| 낮은 $p$일수록 $\Lambda$ 증가 | 지지 |
| 평균 오류 검출 비율이 같아도 함께 생기는 오류가 더 해로움 | X에서 확인, Z에서는 확인되지 않음 |
| 함께 생기는 오류를 아는 디코더가 더 좋음 | X에서 확인, Z에서는 차이 없음 |
| 평균 오류 검출 비율만으로 최종 논리 성능을 설명할 수 없음 | 확인 |

### 수정 가설

> **평균 오류 검출 비율이 같아도, 오류가 어떤 큐비트에 어떤 방향으로 함께 생기는지, X/Z 중 무엇을 보호하는지, 디코더가 그 오류를 알고 있는지에 따라 최종 오류 억제 성능이 달라진다.**

## 10. 논문과 모의실험을 연결하는 방식

### 논문과 공통으로 말할 수 있는 점

- 둘 다 $d=3,5,7$에서 below-threshold scaling을 평가한다.
- 둘 다 $\epsilon_d$와 $\Lambda$를 핵심 지표로 사용한다.
- 둘 다 오류가 함께 생기는 방식과 디코더의 오류 가정이 중요하다는 메시지를 준다.
- 평균 오류 검출 비율만으로 최종 논리 성능을 충분히 설명할 수 없음을 보여준다.

### 수치를 직접 비교하면 안 되는 점

- Willow의 실제 게이트 순서와 본 모의실험의 단순화한 회로가 다르다.
- Willow에서 측정한 실제 오류와 본 실험에서 넣은 가상 오류가 다르다.
- Willow는 최대 250 cycle을 사용하지만 모의실험은 `rounds=distance`를 사용했다.
- Willow의 neural/ensemble decoder와 본 실험의 PyMatching이 다르다.
- 본 모의실험에는 leakage 제거, 아날로그 측정값과 실제 장치의 시간 변화가 없다.

공개 Willow 분석은 **실제 공개 이진 데이터에서 matching 계열 $\Lambda$를 다시 계산한 결과**다. 별도의 가상 오류 실험은 **왜 이런 차이가 생길 수 있는지 단순하게 확인한 보조 실험**이다.

## 11. 내용 우선순위

### 반드시 포함

1. Threshold와 $\Lambda$의 의미
2. Willow의 $\Lambda=2.14$, $d=7$ error와 beyond-breakeven
3. Public Libra 재집계 $\Lambda=2.029$와 direct PyMatching $\Lambda=1.525$
4. Neural decoder와 matching decoder의 구분
5. 공개 data 재집계와 synthetic simulation의 범위 차이
6. Baseline threshold crossing
7. Detection-matched correlated 세 조건과 X/Z basis-dependent 결과
8. 초기 가설의 부분 수정
9. 구현하지 않은 AlphaQubit, leakage와 real-time feedback

### 공간·시간이 허용되면 포함

- DQLR의 12% 대 35% 결과
- Repetition-code rare-burst error floor
- $d=27$, 1,457-qubit extrapolation
- Author correction
- Paired exact $p$-value 전체 표

### 과감히 제외 가능

- Willow chip의 모든 component error 수치
- Supplementary figure/table의 세부 목록
- 모든 decoder ensemble size
- 개별 calibration CSV와 seed
- Optional offline throughput benchmark

## 12. 정확성 규칙

### 사용해도 되는 표현

- “정성적으로 재현했다.”
- “단순화한 가상 오류 모델에서 관찰했다.”
- “강한 오류를 넣은 시험에서 X/Z에 따라 다른 결과를 확인했다.”
- “AlphaQubit 관련 연구를 인용한 neural-network decoder.”
- “Pilot threshold estimate.”

### 사용하면 안 되는 표현

- “Willow를 재현했다.”
- “실제 quantum hardware threshold는 0.008이다.”
- “Correlated noise는 항상 나쁘다.”
- “우리 decoder가 AlphaQubit보다 낫다/비슷하다.”
- “$p<0.008$이면 모든 surface code가 threshold 아래다.”
- “$63\,\mu s$는 완전한 quantum feedback latency다.”

## 13. 아직 남은 내용 확인 과제

- 논리 메모리 수명과 $\epsilon_d$ 계산 방법을 발표에서 어느 수준까지 설명할지 결정
- Leakage/stray-interaction simulation parameter를 상세히 인용할지 결정
- 최종 발표 시간에 따라 rare burst와 real-time decoder 비중 결정

이 항목들은 핵심 결론을 바꾸지 않으며, 최종 발표 분량을 정할 때 선택한다.

## 14. 기준 문헌

1. Google Quantum AI and Collaborators, *Quantum error correction below the surface code threshold*, Nature 638, 920–926 (2025), DOI `10.1038/s41586-024-08449-y`.
2. Bausch et al., *Learning high-accuracy error decoding for quantum processors*, Nature 635, 834–840 (2024), DOI `10.1038/s41586-024-08148-8` — AlphaQubit 논문.
3. 위 Willow 논문의 Supplementary Information과 2026 Author Correction.
4. Google Quantum AI, *Data for “Quantum error correction below the surface code threshold”*, Zenodo, DOI `10.5281/zenodo.13273331`.
