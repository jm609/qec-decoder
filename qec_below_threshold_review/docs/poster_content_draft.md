# 포스터 원고 및 배치안 — 1차 내용본

> 상태: **공개 Willow 재디코딩 결과까지 반영한 내용 원고 완료, 실제 layout 제작 전 단계**\
> [`content_master.md`](content_master.md)의 확정 내용과 공개 105Q 데이터 분석을 반영했다. 팀 정보와 poster template이 정해지면 공간에 맞춰 선택 항목만 덜어낸다.
>
> A1 세로형 포스터의 공개 데이터 재디코딩·시뮬레이션 파트는 [`poster_simulation_section.md`](poster_simulation_section.md)를 기준으로 사용한다.

## 0. 포스터에서 사용할 쉬운 용어

| 어려운 용어 | 포스터에서 함께 쓸 쉬운 설명 |
|---|---|
| Syndrome | 오류의 흔적을 담은 측정 신호 |
| Decoder | 측정 신호를 보고 최종 논리값을 예측하는 프로그램 |
| Code distance $d$ | 오류를 견디는 코드의 크기 |
| $\Lambda$ | 코드를 키웠을 때 오류가 몇 배 줄어드는지 나타내는 값 |
| Decoder prior | 디코더가 미리 가정한 오류 모델 |
| Correlated ZZ noise | 연결된 두 큐비트에 Z 오류가 함께 생기는 현상 |
| ZZ-only surrogate | 위 현상을 단순하게 흉내 낸 가상 오류 모델 |

포스터 본문에서는 한국어 설명을 먼저 쓰고 필요한 전문용어만 괄호 안에 남긴다.

## 1. 포스터 제목

### 권장 한글 제목

> **Willow 공개 데이터로 확인한 Surface Code의 오류 억제**\
> 코드 크기와 디코더에 따른 논리 오류 변화 분석

### 영문 제목

> **Logical-Error Suppression in Public Willow Surface-Code Data**\
> Effects of Code Distance and Decoder Choice

제목에는 “neural decoder 재현”이라는 표현을 사용하지 않는다. 공개된 matching decoder 결과는 다시 계산했지만 neural decoder 자체는 실행하지 않았기 때문이다.

## 2. 상단 핵심 메시지

제목 바로 아래에 다음 문장을 한 줄 강조 상자로 배치한다.

> **공개 Willow 데이터에서는 코드를 키울수록 논리 오류가 줄었다($\Lambda>1$). 그러나 감소 폭은 디코더에 따라 1.525–2.029로 달랐다.**

## 3. 전체 배치

```text
┌──────────────────────────────────────────────────────────────┐
│ 제목 / 저자 / 소속 / 핵심 메시지                              │
├─────────────────────────────┬────────────────────────────────┤
│ 1. 배경·논문·공개 데이터    │ 2. 보조 시뮬레이션             │
│                             │                                │
│ Surface code와 Λ            │ 연구 질문·방법                 │
│ Willow 논문 핵심 성과       │                                │
│                             │ Figure 2A: threshold           │
│ Figure 1: 공개 Willow       │                                │
│ decoder 비교                │ Figure 2B: correlated ZZ       │
│                             │                                │
│ 논문·공개 데이터 결론       │ simulation 결론·한계           │
├─────────────────────────────┴────────────────────────────────┤
│ References / QR / artifact 안내                              │
└──────────────────────────────────────────────────────────────┘
```

포스터는 A1 세로형 `594 × 841 mm`로 제작한다. 공통 제목 영역 아래를 중앙 세로선으로 나누고 좌우 폭을 동일하게 사용한다. 왼쪽에는 배경·논문 설명·실제 공개 데이터 분석을, 오른쪽에는 단순화한 시뮬레이션을 배치한다. 중앙선 양쪽에는 충분한 안쪽 여백을 두어 두 파트가 섞여 보이지 않게 한다.

## 4. 왼쪽 영역 — 배경과 논문 리뷰

### 4.1 문제 정의

> 실제 큐비트의 오류가 threshold라는 경계보다 낮으면 surface code의 크기 $d$를 키울수록 cycle당 논리 오류율 $\epsilon_d$가 감소한다. 이것이 below-threshold scaling이다.

### 4.2 핵심 식

$$
\epsilon_d\propto
\left(\frac{p}{p_{thr}}\right)^{(d+1)/2},
\qquad
\Lambda=\frac{\epsilon_d}{\epsilon_{d+2}}.
$$

- $\Lambda>1$: 코드를 키우면 논리 오류가 감소
- $\Lambda=1$: threshold 부근
- $\Lambda<1$: 코드를 키워도 이점이 없음

### 4.3 Willow 논문의 주요 성과

네 개의 숫자 카드로 배치한다.

| 항목 | 논문 결과 |
|---|---:|
| 최대 surface-code distance | $d=7$ |
| Neural decoder의 오류 감소 배율 | $\Lambda=2.14\pm0.02$ |
| $d=7$의 cycle당 논리 오류율 | $(1.43\pm0.03)\times10^{-3}$ |
| 가장 좋은 물리 큐비트 대비 논리 메모리 수명 | $2.4\pm0.3$배 |

아래에 짧게 추가한다.

> 논리 메모리가 가장 좋은 물리 큐비트보다 오래 유지되는 결과도 얻었다. 또한 최대 $10^6$ cycle을 실시간으로 처리했다. 다만 에너지가 계산 공간 밖으로 새는 현상(leakage)과 드물게 발생하는 큰 상관 오류는 앞으로 해결할 문제로 남았다.

두 줄 evidence box를 추가한다.

- **Leakage 제거(DQLR):** 평균 오류 신호는 약 12% 감소했지만 $\Lambda$는 약 35% 증가
- **실시간 $d=5$:** 결과 계산 시간 $63\pm17\,\mu s$, QEC cycle $1.1\,\mu s$, 최대 $10^6$ cycle

Decoder 표기는 다음처럼 구분한다.

> Willow의 가장 정확한 결과는 **신경망 디코더(neural-network decoder)**가 만들었다. AlphaQubit은 관련된 별도 연구이며, 본 프로젝트의 PyMatching은 AlphaQubit이 아니다.

### 4.4 논문에서 프로젝트로

> **논문 질문:** Willow에서는 코드를 키울수록 실제 논리 오류가 줄었는가?\
> **공개 데이터 질문:** 공개된 예측값에서 $\Lambda$를 다시 계산할 수 있는가? 같은 오류 신호를 PyMatching으로 직접 처리하면 결과가 어떻게 달라지는가?\
> **모의실험 질문:** 오류가 함께 생기는 방식과 디코더가 가정한 오류 모델은 성능에 어떤 영향을 주는가?

## 5. 분석 방법 원고

공개 Willow 데이터 처리 과정은 왼쪽 영역에, 단순화한 모의실험 과정은 오른쪽 영역에 배치한다.

### 5.1 공개 Willow 데이터 처리 과정

```text
105Q ZIP: 실제 최종값 + 오류 측정 신호 + 회로
            ↓
공개 예측값 재계산 / PyMatching으로 직접 예측
            ↓
cycle 수에 따른 실패율 변화를 계산
            ↓
크기별 ε 계산 → d=3,5,7의 감소 배율 Λ 계산
```

- 칩의 여러 위치 사용: $d=3$ 9곳, $d=5$ 4곳, $d=7$ 1곳; X/Z 별도 분석
- 조정용 13-cycle 자료를 제외한 주요 실험 조건 364개 사용
- ZIP 크기와 MD5 확인, 작은 시험에서 공개 예측과 직접 예측이 완전히 같음을 확인

### 5.2 단순화한 모의실험 과정

```text
단순화한 surface-code memory
            ↓
독립적으로 생기는 오류 / 두 큐비트에 함께 생기는 ZZ 오류
            ↓
오류 측정 신호와 최종 논리값 생성
            ↓
디코더용 오류 설명서(DEM)
            ↓
PyMatching으로 최종값 예측
            ↓
실패율, ε, Λ와 두 디코더의 차이 계산
```

### 5.3 모의실험 공통 조건

| 항목 | 설정 |
|---|---|
| Simulator | Stim 1.15.0 |
| Decoder | PyMatching 2.3.1 |
| Distance | $d=3,5,7$ |
| 보호한 논리값 | X/Z 별도 |
| QEC cycle 수 | Code distance와 같게 설정 |
| 통계적 오차 범위 | Wilson 95%, 20,000회 반복 계산 |
| 여러 조건 동시 비교 보정 | Holm 방법 |

### 5.4 기본 threshold 실험

- $p=0.0015$–$0.02$의 9개 점
- 각 조건을 20,000번 반복
- 총 54개 X/Z–distance–오류율 조건
- $d=3,5,7$에서 $\epsilon$이 줄어드는 정도로 $\Lambda$와 threshold 추정

### 5.5 함께 생기는 ZZ 오류 실험

세 조건을 비교한다.

| 조건 | 실제로 넣은 오류 | 디코더가 가정한 오류 |
|---|---|---|
| A | 서로 독립적인 오류 | 독립 오류 |
| B | 두 큐비트에 함께 생기는 ZZ 오류 | 독립 오류만 안다고 가정 |
| C | 조건 B와 완전히 같은 오류 | 함께 생기는 ZZ 오류도 안다고 가정 |

- Stage C: $p=0.003$, `zz_factor=1.0`
- 각 조건을 50,000번 반복
- 다른 종류의 동시 오류는 제외하고 ZZ 오류에 집중
- 세 조건의 평균 오류 검출 비율 차이를 0.0005보다 작게 맞춤
- B/C에는 같은 오류 측정값을 넣어 디코더가 가정한 오류 모델의 효과만 비교
- 추가 확인: $d=3$에서 ZZ 오류 세기를 네 단계로 바꾸어 각각 50,000번 반복

### 5.6 평가식

$$
\epsilon=\frac12\left[1-(1-2p_L)^{1/r}\right].
$$

> $p_{det}$는 전체 오류 검출 신호 중 오류가 감지된 비율이다. 논문에서 사용한 평균 검출 확률과 계산 범위가 완전히 같지 않으므로 숫자를 직접 비교하지 않는다.

### 5.7 범위 표시 상자

> **분석 범위:** 공개 Willow의 실제값과 matching-decoder 예측값에서 $\Lambda$를 다시 계산했다. 그러나 neural decoder·AlphaQubit은 실행하지 않았다. 별도의 모의실험은 실제 Willow 장치가 아니라 단순화한 가상 오류 모델이다.

## 6. 결과와 해석 원고

Figure 1은 왼쪽 영역에 배치한다. Figure 2A와 Figure 2B는 오른쪽 시뮬레이션 영역에서 위아래로 배치한다.

### 6.1 Figure 1 — 공개 Willow 데이터의 디코더 비교

주 결과로 가장 크게 배치한다.

```text
artifacts/qec_below_threshold_review/willow_reproduction_comparison_poster.png
```

| 처리 방법 | $\epsilon_3$ | $\epsilon_5$ | $\epsilon_7$ | $\Lambda$ |
|---|---:|---:|---:|---:|
| Paper neural | 0.00650 | 0.00303 | 0.00143 | 2.14 |
| Paper Libra 101 | 0.00712 | 0.00349 | 0.00171 | 2.04 |
| Public Libra 51 재집계 | 0.006802 | 0.003372 | 0.001652 | **2.029** |
| Direct PyMatching + SI1000 | 0.010438 | 0.007102 | 0.004485 | **1.525** |

권장 caption:

> **Figure 1.** Willow 105Q 공개 데이터에서 code distance가 커질 때 논리 오류가 어떻게 변하는지 비교했다. 공개 Libra 결과를 다시 계산한 $\Lambda=2.029$는 논문의 2.04와 가까웠다. 같은 오류 측정값을 기본 SI1000 오류 모델과 PyMatching으로 직접 처리하면 $\Lambda=1.525$였다. 두 결과 모두 $\Lambda>1$이지만 Libra에서 오류가 더 빠르게 감소했다. Neural/Libra 논문값은 비교용이며 neural decoder를 직접 실행한 값은 아니다.

강조 숫자: 직접 PyMatching의 오류율은 Libra보다 $d=3$에서 1.53배, $d=7$에서 2.71배 높았다.

### 6.2 Figure 2A — 단순화한 기본 threshold 실험

사용할 파일:

```text
artifacts/qec_below_threshold_review/baseline_distance_pilot_poster.png
```

그림 아래에는 다음 세 줄만 둔다.

- 낮은 가상 오류율: 큰 코드일수록 논리 오류 감소
- Pilot crossing: X 0.008852 [0.008581, 0.009188], Z 0.008089 [0.007793, 0.008526]
- $p=0.003$: X $\Lambda=2.551$, Z $\Lambda=2.364$

대괄호는 같은 계산을 20,000번 반복해 구한 95% 오차 범위다.

권장 caption:

> **Figure 2A.** 오류가 서로 독립적으로 생기는 기본 모의실험이다. 각 조건을 20,000번 반복했다. 가상 오류율이 낮을 때는 코드를 키울수록 논리 오류가 줄었지만, threshold보다 높아지면 큰 코드의 이점이 사라졌다.

### 6.3 Figure 2B — 함께 생기는 ZZ 오류의 영향

사용할 파일:

```text
artifacts/qec_below_threshold_review/correlated_scaling_poster.png
```

그림 아래에 suppression-factor 표를 둔다.

| 보호한 논리값 | 독립 오류 | ZZ 오류를 모르는 디코더 | ZZ 오류를 아는 디코더 |
|---|---:|---:|---:|
| X | 1.880 | 1.374 | 1.400 |
| Z | 1.831 | 2.202 | 2.202 |

권장 caption:

> **Figure 2B.** 평균 오류 검출 비율을 비슷하게 맞춘 뒤, 두 큐비트에 Z 오류가 함께 생기도록 했다. X memory에서는 오류 억제 성능이 낮아졌고, 이 오류를 알고 있는 디코더가 일부 회복했다. Z memory에서는 두 디코더의 결과가 같았다. 각 조건을 50,000번 반복했다.

### 6.4 디코더가 오류 모델을 아는 효과

작은 강조 상자로 배치한다.

> **같은 X-memory 오류 신호에서 ZZ 오류를 아는 디코더가 추가로 성공한 횟수:**\
> $d=3$: 46, $d=5$: 119, $d=7$: 136

아래에 작게 표기한다.

> 여러 조건을 동시에 비교한 효과를 보정한 뒤에도 X memory의 차이는 통계적으로 분명했다. Z memory에서는 두 디코더의 성공 횟수가 같았다.

### 6.5 ZZ 오류 세기를 바꾼 추가 확인

작은 보조 결과 상자로 배치한다.

| `zz_factor` | 0.05 | 0.25 | 0.5 | 1.0 |
|---:|---:|---:|---:|---:|
| X에서 추가 성공한 횟수 ($d=3$) | 2 | 15 | 19 | 45 |
| Holm-adjusted $p$ | 1.0 | 1.0 | 0.808 | 0.0181 |

> **해석:** 여러 조건을 함께 고려하면 가장 강한 `zz_factor=1.0`에서만 디코더 차이가 분명했다. 약한 상관 오류에서도 항상 효과가 있다고 말할 수는 없다.

### 6.6 가설 평가

| 초기 예상 | 결과 |
|---|---|
| 공개 Willow 데이터에서 코드를 키우면 오류가 감소 | 모든 matching decoder에서 확인 |
| 디코더에 따라 오류 감소 폭이 달라짐 | $\Lambda=1.525$–2.029로 확인 |
| 낮은 가상 오류율에서 큰 코드가 유리 | 확인 |
| 높은 가상 오류율에서는 큰 코드의 이점 소실 | 확인 |
| 함께 생기는 ZZ 오류가 성능을 낮춤 | X에서 확인, Z에서는 확인되지 않음 |
| ZZ 오류를 아는 디코더가 더 좋음 | X에서 일부 확인, Z에서는 차이 없음 |

### 6.7 최종 결론

> **공개 Willow 데이터에서 코드를 키울수록 오류가 줄어드는 것을 다시 확인했다. 그러나 같은 오류 측정값에서도 디코더가 어떤 오류 모델을 가정하는지에 따라 오류율과 감소 폭이 크게 달라졌다.**

보조 결론:

- 공개 Libra 결과는 논문값과 가까웠다.
- 직접 PyMatching도 코드를 키우면 오류가 줄었지만 감소 폭은 더 작았다.
- 단순화한 모의실험에서도 threshold 현상을 확인했다.
- 함께 생기는 오류의 영향은 X와 Z를 나누어 봐야 했다.
- 오류 구조를 아는 디코더가 도움이 될 수 있지만 모든 경우에 효과가 있는 것은 아니었다.

## 7. 하단 — 분석의 한계와 참고문헌

### 분석의 한계

포스터에서는 다음 다섯 항목만 사용한다.

1. 공개 Libra는 디코더 51개를 합친 결과이며 논문의 101개 결과와 구성이 다르다.
2. 직접 PyMatching 자료 중 실패율 50% 이상인 마지막 측정점 4개는 로그 계산에서 제외했다.
3. Neural decoder와 AlphaQubit은 직접 실행하지 않았다.
4. 함께 생기는 ZZ 오류 실험은 실제 Willow 장치가 아니라 단순화한 강한 가상 오류다.
5. 모의실험의 오차 범위는 반복 횟수에 따른 흔들림만 반영하며, 추가 확인은 $d=3$에서만 했다.

### 참고문헌

> Google Quantum AI and Collaborators, “Quantum error correction below the surface code threshold,” *Nature* **638**, 920–926 (2025). DOI: 10.1038/s41586-024-08449-y.

> J. Bausch et al., “Learning high-accuracy error decoding for quantum processors,” *Nature* **635**, 834–840 (2024). DOI: 10.1038/s41586-024-08148-8.

> Google Quantum AI, “Data for ‘Quantum error correction below the surface code threshold’,” Zenodo. DOI: 10.5281/zenodo.13273331.

Software 이름은 Method에 표시하고 상세 software citation은 최종 template의 reference 공간을 확인한 뒤 추가한다.

## 8. 시각 디자인 규칙

- 배경: 흰색 또는 매우 옅은 회색
- 본문 글자: 짙은 남색/회색
- Baseline 거리 색상: 기존 figure의 파랑($d=3$), 빨강($d=5$), 초록($d=7$) 유지
- Correlated 조건 색상: 파랑(independent), 빨강(mismatched), 초록(matched) 유지
- Willow 디코더 그림의 색상과 점 모양은 생성된 그림의 네 경로 순서를 유지
- 강조 색은 2–3개만 사용
- 결과 figure의 범례·색을 포스터 본문 표와 동일하게 유지
- 본문 한 문단은 최대 3문장
- 표와 수식은 설명문보다 크게 배치

권장 A1 font size:

| 요소 | 권장 크기 |
|---|---:|
| 제목 | 56–66 pt |
| 저자·소속 | 28–34 pt |
| 단 제목 | 28–32 pt |
| 소제목 | 22–26 pt |
| 본문 | 18–21 pt |
| Caption/Reference | 13–16 pt |

## 9. 실제 포스터 제작 전 필요한 입력

- 학교 또는 학회 poster template
- 최종 인쇄 여백과 중앙선 규격(A1 세로 방향은 확정)
- 팀원 이름, 소속, 지도교수
- 학교/연구실 logo 원본
- QR code에 연결할 공개 저장소 또는 보고서 URL

이 정보가 정해질 때까지는 본 문서를 포스터 내용 기준본으로 사용한다. Template에 공간이 부족하면 Section 11의 선택 항목부터 제거하고 핵심 주장·Willow 주 figure·범위 상자는 유지한다.

## 10. 실제 포스터에 반드시 남길 내용

공간이 부족해도 다음 요소는 삭제하지 않는다.

1. Willow의 neural $\Lambda=2.14\pm0.02$와 $d=7$ logical error
2. Public Libra 재집계 $\Lambda=2.029$와 direct PyMatching $\Lambda=1.525$
3. $\Lambda>1$의 의미와 Willow decoder scaling figure
4. Neural decoder를 직접 재현하지 않았다는 범위 상자
5. 단순화한 기본 threshold 실험과 함께 생기는 오류 실험의 핵심 한 줄
6. X/Z basis-dependent 결과와 강한 stress-test 한계
7. 최종 결론 한 문장

## 11. 공간이 부족하면 줄일 순서

1. Author correction과 repetition-code error floor 설명 생략
2. Sensitivity 표를 “`zz_factor=1.0`만 보정 후 유의” 한 문장으로 축약
3. 보조 모의실험 Figure 2A/2B를 한 개 또는 핵심 수치 카드로 축약
4. 모의실험 처리 과정을 한 줄 화살표로 축약
5. 가설 평가 표를 최종 결론 상자에 통합
