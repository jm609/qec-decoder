# A1 세로형 포스터 — 공개 데이터 재디코딩 및 시뮬레이션 파트 원고

> 상태: **A1 영문 포스터에 바로 옮길 수 있는 프로젝트 분석 파트 기준본**\
> 실제 Willow 공개 데이터 재디코딩과 Stim–PyMatching 보조 시뮬레이션을 하나의 흐름으로 구성한다. 포스터에 직접 넣는 문구는 영어로, 작업 지침과 발표자 메모는 한국어로 작성한다.

## 1. 포스터 안에서의 위치와 역할

A1 세로 규격은 `594 × 841 mm`이다. 논문 설명 파트와 본 프로젝트 분석 파트를 시각적으로 구분하되, 현재 PDF 초안의 칸 크기에는 얽매이지 않는다. 프로젝트 분석 파트에는 세 결과를 위에서 아래로 배치한다.

```text
┌──────────────────────────────────────────────┐
│ 제목 · 저자 · 소속 · 전체 핵심 메시지        │
├──────────────────────┬───────────────────────┤
│ 1. 배경·논문 설명    │ 2. 프로젝트 분석      │
│                      │                       │
│ Surface code와 Λ     │ 연구 질문·방법 요약   │
│ 논문 핵심 성과       │                       │
│ Willow 실험 결과     │ Figure 1              │
│                      │ Public-data re-decoding│
│ 논문 해석·한계       │ Figure 2A             │
│                      │ Baseline threshold    │
│                      │ Figure 2B             │
│                      │ Correlated-ZZ test    │
│                      │ 통합 결론·분석 범위   │
└──────────────────────┴───────────────────────┘
```

공통 제목 영역은 전체 폭을 사용한다. 본문을 중앙선으로 나눌 경우 선은 `1.5–2 pt`의 옅은 회색으로 하고 양쪽에 `12–16 mm`의 안쪽 여백을 둔다. 프로젝트 분석 파트에서는 Figure 1, Figure 2A, Figure 2B를 위아래로 배치하여 축·범례·caption의 가독성을 우선한다. 세 Figure 번호는 전체 포스터 편집 시 최종 순서에 맞게 다시 지정한다.

## 2. 파트 제목과 핵심 메시지

### 권장 제목

> **WHAT CONTROLS LOGICAL-ERROR SUPPRESSION?**

### 권장 부제

> *Willow re-decoding and controlled noise simulations*

### 제목 바로 아래 강조 문장

> **Public Willow data showed decoder-dependent logical scaling, while controlled simulations isolated threshold behavior and the effects of correlated noise and decoder priors.**

### 실제 데이터와 시뮬레이션을 구분하는 표시

각 결과 위에 다음 label을 사용한다.

- Figure 1: **PUBLIC WILLOW DATA**
- Figure 2A·2B: **CONTROLLED SIMULATION — NOT WILLOW HARDWARE DATA**

## 3. 연구 질문

포스터에는 다음 세 질문만 남긴다.

1. **Can the published logical scaling be recovered from the public Willow data?**
2. **Does increasing code distance reduce the logical error below threshold?**
3. **At matched detection probability, do correlated noise and the decoder prior alter logical performance?**

## 4. 방법 요약

### A. 공개 Willow 데이터 재디코딩

포스터에 다음 문장을 넣는다.

> We re-aggregated the provided Libra predictions and directly decoded the measured Willow detection events with PyMatching. Stim generated the detector-error model from the provided SI1000 circuit; it did not generate the experimental samples.

### B. 보조 시뮬레이션

포스터에 다음 문장을 넣는다.

> Using Stim, we generated and sampled rotated surface-code memory circuits at $d=3,5,7$ and decoded them with PyMatching. We analyzed X and Z memories separately and compared an independent-noise baseline with a correlated-ZZ stress test.

### 보조 시뮬레이션 조건표

| Item | Setting |
|---|---|
| Simulator / decoder | Stim 1.15.0 / PyMatching 2.3.1 |
| Code distance | $d=3,5,7$ |
| Memory basis | X and Z analyzed separately |
| Baseline | $p=0.0015\text{–}0.02$; 20,000 shots per point |
| Correlated-ZZ test | $p=0.003$; 50,000 shots per condition |
| Error bars | Wilson 95% confidence intervals |

본문 공간이 부족하면 조건표 대신 다음 한 줄을 사용한다.

> $d=3,5,7$; X/Z memory; baseline 20,000 shots/point; correlated-ZZ test 50,000 shots/condition.

## 5. Figure 1 — Public Willow data re-decoding

사용 파일:

```text
artifacts/qec_below_threshold_review/willow_reproduction_comparison_poster.png
```

### 위치

- 프로젝트 분석 파트의 첫 번째 결과
- 방법 요약 바로 아래에 배치
- 실제 데이터임을 나타내는 `PUBLIC WILLOW DATA` label 사용

### 그림이 보여주는 내용

- 가로축: code distance $d=3,5,7$
- 세로축: cycle당 논리 오류율 $\epsilon$
- Paper neural과 Paper Libra: 논문 reference
- Public Libra: 공개된 51-decoder prediction 재집계
- Direct PyMatching + SI1000: 본 프로젝트에서 직접 실행한 재디코딩

### 핵심 수치

| Pathway | Suppression factor |
|---|---:|
| Paper neural decoder | $\Lambda=2.140$ |
| Paper Libra, 101 ensemble | $\Lambda=2.040$ |
| Public Libra, 51 ensemble | $\Lambda=2.029$ |
| Direct PyMatching + SI1000 | $\Lambda=1.525$ |

### 포스터용 caption

> **Figure 1. Decoder-dependent scaling in public Willow data.** Re-aggregating the provided Libra predictions yielded $\Lambda=2.029$, close to the paper's Libra value of $2.04$. Directly decoding the measured detection events with PyMatching and a SI1000 detector-error model yielded $\Lambda=1.525$. Both pathways had $\Lambda>1$, but direct PyMatching showed weaker suppression.

### 이 그림에서 전달할 한 문장

> **The public data reproduced below-threshold scaling, but the fitted suppression factor depended strongly on the decoder pathway and prior.**

### 반드시 함께 표시할 범위

> **Scope:** Public Libra is a reanalysis of the provided predictions; only the direct PyMatching pathway was executed in this project. The neural and AlphaQubit decoders were not rerun.

## 6. Figure 2A — Baseline threshold simulation

사용 파일:

```text
artifacts/qec_below_threshold_review/baseline_distance_pilot_poster.png
```

### 위치

- 오른쪽 시뮬레이션 영역의 위쪽 절반
- 방법 요약 바로 아래에 배치
- 오른쪽 영역의 가용 폭을 거의 모두 사용
- 그림 너비는 A1 인쇄 기준 약 `245–260 mm` 권장

### 그림이 보여주는 내용

- 위 왼쪽: X memory에서 물리 오류율 $p$에 따른 cycle당 논리 오류율 $\epsilon$
- 위 오른쪽: Z memory에서 같은 비교
- 파랑, 빨강, 초록: 각각 $d=3,5,7$
- 아래: 물리 오류율에 따른 suppression factor $\Lambda$
- 검은 점선 $\Lambda=1$: 큰 코드의 이점이 사라지는 경계
- 세로 점선: X/Z memory에서 추정한 crossing 위치

### 그림 옆 핵심 수치

Low-noise operating point, $p=0.003$:

$$
\Lambda_X=2.551,
\qquad
\Lambda_Z=2.364
$$

Estimated crossing:

$$
p_X\approx0.00885,
\qquad
p_Z\approx0.00809
$$

### 포스터용 caption

> **Figure 2A. Baseline threshold simulation.** Under independent noise, larger code distances yielded lower logical error at low $p$. As $p$ increased, $\Lambda$ approached unity; beyond the crossing, increasing the distance no longer helped. Each point used 20,000 shots; error bars show Wilson 95% confidence intervals.

### 이 그림에서 전달할 한 문장

> **The independent-noise model reproduced the expected below-threshold scaling.**

## 7. Figure 2B — Correlated-ZZ stress test

사용 파일:

```text
artifacts/qec_below_threshold_review/correlated_scaling_poster.png
```

### 위치

- 오른쪽 시뮬레이션 영역에서 Figure 2A 바로 아래
- 오른쪽 영역의 가용 폭을 거의 모두 사용
- 그림 너비는 A1 인쇄 기준 약 `245–260 mm` 권장
- 그림 아래에 suppression-factor 수치와 해석 상자를 배치

### 세 조건

| Color | Sampled noise | Decoder prior |
|---|---|---|
| Blue | Independent noise | Independent prior |
| Red | Correlated ZZ noise | Mismatched independent prior |
| Green | Same correlated samples as red | Matched correlated-ZZ prior |

> Red and green use the same detector samples, isolating the effect of the decoder prior. The independent and correlated conditions were calibrated to differ in mean detection probability by less than `0.0005`.

### 그림 아래 핵심 수치

| Memory | Independent | Mismatched prior | Matched prior |
|---|---:|---:|---:|
| X | $\Lambda=1.880$ | $\Lambda=1.374$ | $\Lambda=1.400$ |
| Z | $\Lambda=1.831$ | $\Lambda=2.202$ | $\Lambda=2.202$ |

### 포스터용 caption

> **Figure 2B. Correlated-ZZ stress test.** After matching the mean detection probability, correlated ZZ noise reduced the X-memory suppression factor relative to independent noise. A matched correlated prior recovered a small part of the loss. For Z memory, the matched and mismatched priors produced identical results. Each condition used 50,000 shots.

### 이 그림에서 전달할 한 문장

> **Matching the average detection rate did not guarantee equal logical performance; the effect of correlated noise depended on the memory basis.**

## 8. Decoder prior 효과를 보여주는 작은 결과 상자

Figure 2B 아래 공간이 충분할 때만 넣는다.

> **Additional successful decodes from the matched prior on the same X-memory samples**\
> $d=3$: 46, $d=5$: 119, $d=7$: 136

아래 문장을 작게 덧붙인다.

> After Holm correction across comparisons, the prior-dependent difference remained statistically significant for X memory. The two priors produced the same number of successful decodes for Z memory.

공간이 부족하면 성공 횟수는 삭제하고 Figure 2B caption만 유지한다.

## 9. 프로젝트 분석 파트 최종 결론

파트 맨 아래에 다음 결론 상자를 배치한다.

> **Conclusion**\
> Public Willow data showed below-threshold scaling under both decoder pathways, but the fitted suppression factor changed from $\Lambda=2.029$ for Public Libra to $\Lambda=1.525$ for direct PyMatching. Controlled simulations reproduced threshold behavior and showed that similar detection rates can still yield different logical scaling when the noise structure or decoder prior changes.

## 10. 반드시 표시할 분석 범위

결론 옆이나 파트 하단에 작은 회색 상자로 표시한다.

> **Scope and limitations:** Public Libra is a reanalysis of provided predictions, while direct PyMatching is the only decoder pathway executed here. The controlled simulations do not reproduce the full Willow hardware noise, and the correlated-ZZ stress test does not quantitatively explain the `2.029` versus `1.525` gap. Its strongest prior-dependent effect occurred at `zz_factor=1.0` and should not be generalized to all weaker correlations.

## 11. 포스터에 실제로 남길 내용

공간이 부족해도 다음 항목은 삭제하지 않는다.

1. `PUBLIC WILLOW DATA`와 `CONTROLLED SIMULATION` 구분
2. Public Libra $\Lambda=2.029$와 Direct PyMatching $\Lambda=1.525$
3. Public Libra는 재집계, Direct PyMatching만 직접 실행했다는 범위
4. Stim, PyMatching, $d=3,5,7$, X/Z memory
5. Figure 2A의 $\Lambda=1$ 의미와 threshold 결과
6. Figure 2B의 independent/mismatched/matched 조건 구분
7. X memory와 Z memory에서 결과가 달랐다는 점
8. 단순화된 강한 correlated-ZZ stress test라는 한계
9. 최종 결론 상자

## 12. 공간이 부족할 때 줄이는 순서

1. Decoder prior의 추가 성공 횟수 삭제
2. Figure 1의 네 수치 표를 caption에 통합
3. Figure 2A의 crossing 신뢰구간 삭제하고 중앙값만 유지
4. 방법 조건표를 한 줄로 축약
5. Figure 2B suppression-factor 표를 caption에 통합
6. Figure 2A를 아래쪽 $\Lambda$ 대 $p$ 패널 중심으로 단순화

Figure 2A와 Figure 2B를 모두 제거하거나, 실제 Willow 데이터와 같은 Figure로 합쳐 표시하지 않는다.

## 13. 발표자가 알고 있어야 할 해석

- Figure 1은 실제 Willow 측정 데이터 분석이며 시뮬레이션이 아니다.
- Stim은 Direct PyMatching에서 DEM을 생성했지만 실제 Willow sample을 생성하지 않았다.
- Public Libra의 `2.029`는 제공 prediction 재집계이고 Direct PyMatching의 `1.525`만 직접 디코딩한 값이다.
- $\Lambda>1$은 code distance를 2만큼 키웠을 때 논리 오류가 감소했다는 뜻이다.
- Figure 2A는 논문의 정확한 Willow threshold를 측정한 것이 아니라, 단순화한 회로·오류 모델의 crossing을 구한 것이다.
- Figure 2B에서는 detection probability를 비슷하게 맞췄지만 전체 오류 구조까지 같게 만든 것은 아니다.
- X memory에서 matched prior의 개선은 작지만 paired sample 비교에서 확인되었다.
- Z memory에서 correlated 조건이 independent 조건보다 좋아 보이는 결과도 있으므로, “correlated noise는 항상 성능을 낮춘다”고 말하면 안 된다.
- 가장 안전한 결론은 “상관 오류의 영향과 decoder-prior 효과가 basis-dependent였다”이다.
- 보조 시뮬레이션은 실제 Willow의 `2.029`와 `1.525` 차이를 정량적으로 설명하지 않는다.
