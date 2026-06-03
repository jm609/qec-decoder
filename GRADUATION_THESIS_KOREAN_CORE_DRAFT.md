# 졸업논문 한국어 핵심 초안

이 문서는 `GRADUATION_THESIS_DRAFT.md`의 `Integrated Core Draft V1`을
한국어 제출본에 바로 옮길 수 있도록 정리한 핵심 본문 초안이다. 기술
주장과 수치는 현재 고정된 evidence 문서를 기준으로 작성했다.

## 논문 제목

표면 코드 양자 오류 정정을 위한 전이 정보 기반 신경망 사전 디코더의
설계 및 성능 분석

영문 제목:

Design and Evaluation of a Transition-Aware Neural Pre-Decoder for Surface-Code
Quantum Error Correction

## 초록

본 연구는 표면 코드 기반 양자 오류 정정에서 기존 PyMatching 디코더의
입력 syndrome을 신경망으로 사전 보정하는 전이 정보 기반 neural
pre-decoder를 설계하고 평가한다. 제안하는 방법은 최종 logical frame을
신경망이 직접 분류하는 방식이 아니라, 36채널 syndrome/noise volume을
입력으로 받아 작은 local detector edit 후보를 선택한 뒤, 수정된 syndrome
또는 원본 syndrome을 PyMatching에 전달하는 구조이다. 따라서 PyMatching은
최종 디코더로 유지되고, 신경망은 matching 이전 단계에서 syndrome 구조를
국소적으로 보정하는 역할을 수행한다.

제안 모델은 3차원 residual convolution trunk, local motif candidate 생성,
patch-head CandidateEditSelector, benefit/harm 기반 candidate ranking,
그리고 candidate-first selected-mode safety policy로 구성된다. selector의
목표는 detector reconstruction 자체가 아니라, local edit을 적용한 뒤
PyMatching의 최종 `logical_class4` 정답률이 원본 syndrome 대비 개선되는지
여부이다. 이 decoder-aware target을 통해 신경망은 최종 matching 성능에
도움이 되는 local edit을 선택하도록 학습된다.

실험은 d3, d5, d7 표면 코드 거리에서 수행했으며, 주요 평가는 held-out
correlated-noise family인 `stage_c_corr`에서 진행했다. d3와 d5에서는
selected-mode predecoder가 raw PyMatching 대비 각각 `+0.006591797`,
`+0.005615234`의 평균 정확도 향상을 보였다. 반면 d7에서는 selected-mode
향상이 `+0.000151536`에 그쳤다. 추가 분석 결과, d7에서는 모든 58개 seed에
positive oracle headroom이 존재했지만 learned selector가 유익한 후보를
안정적으로 선택하지 못했다. 특히 validation-positive candidate seed 중
held-out에서 harmful이 된 경우가 13개, positive가 된 경우가 5개로 나타나,
d7의 핵심 병목은 candidate coverage 부족이 아니라 selector ranking 및
generalization 문제임을 확인했다.

본 연구는 neural pre-decoding이 작은 거리의 표면 코드에서 강한 matching
decoder의 성능을 개선할 수 있음을 보이고, 더 큰 거리로 확장하기 위해
해결해야 할 selector 설계상의 병목을 실험적으로 분석한다.

키워드: 양자 오류 정정, 표면 코드, neural pre-decoder, PyMatching, syndrome
decoding, selector ranking

## 1. 서론

양자컴퓨터는 계산 과정에서 decoherence, gate error, measurement error 등
다양한 물리적 잡음의 영향을 받는다. 이러한 오류는 양자 상태를 빠르게
손상시키므로, 실용적인 양자 계산을 위해서는 양자 오류 정정(quantum error
correction)이 필수적이다. 표면 코드(surface code)는 국소적인 stabilizer
측정과 높은 오류 허용 특성 때문에 가장 널리 연구되는 양자 오류 정정
코드 중 하나이다.

표면 코드에서 디코더는 측정된 syndrome 또는 detector event를 바탕으로
발생한 오류의 logical effect를 추정해야 한다. PyMatching은 minimum-weight
perfect matching 기반의 강력한 고전 디코더로, 표면 코드 디코딩에서 널리
사용된다. 그러나 복잡한 노이즈 환경이나 더 큰 코드 거리에서는 syndrome
구조가 복잡해지고, 일부 샘플에서는 matching 이전의 국소적 보정이 최종
성능을 개선할 가능성이 있다.

본 연구는 PyMatching을 대체하는 end-to-end neural decoder가 아니라,
PyMatching 앞단에서 local syndrome edit을 선택하는 neural pre-decoder를
연구한다. 핵심 질문은 다음과 같다.

1. 신경망이 선택한 작은 local detector edit이 PyMatching의 최종
   `logical_class4` 정확도를 개선할 수 있는가?
2. 이 개선이 d3, d5, d7로 코드 거리가 증가할 때 어떻게 변하는가?
3. 성능 한계가 발생한다면 그 원인은 candidate 부족인가, 아니면 learned
   selector의 ranking 및 generalization 문제인가?

본 연구의 기여는 다음과 같다.

1. 직접 구성한 noise family와 surface-code syndrome dataset을 바탕으로
   PyMatching과 결합 가능한 neural pre-decoder 구조를 설계했다.
2. 36채널 syndrome/noise volume, 3D residual trunk, local motif candidate,
   patch-head selector를 결합하여 local detector edit을 선택하는 구조를
   구현했다.
3. held-out `stage_c_corr` 환경에서 d3와 d5가 raw PyMatching 대비
   selected-mode 성능 향상을 보임을 확인했다.
4. d7에서는 oracle headroom이 존재함에도 learned selector가 이를 안정적으로
   활용하지 못함을 분석하여, 확장 병목이 candidate coverage가 아니라
   selector ranking/generalization임을 보였다.

## 2. 배경 및 관련 연구

표면 코드는 qubit을 2차원 격자 위에 배치하고 stabilizer 측정을 반복하여
오류 syndrome을 얻는다. 반복 측정 과정에서 발생하는 detector event는
오류가 발생했을 가능성이 있는 시공간적 위치 정보를 제공한다. 디코더는
이 detector event pattern을 이용해 최종 logical frame을 추정한다.

PyMatching은 표면 코드 syndrome을 graph matching 문제로 변환하여 오류
chain을 추정한다. 이 방식은 해석 가능하고 강력하며, 본 연구에서도 최종
디코더로 유지된다. 본 연구의 neural model은 PyMatching을 대체하지 않고,
PyMatching이 입력으로 받는 syndrome을 국소적으로 수정할지 여부를 판단한다.

기존 neural decoder 연구 중 일부는 syndrome에서 logical class 또는
correction field를 직접 예측한다. 그러나 본 프로젝트의 direct neural
baseline인 FLFD-small과 M3D-FLFD는 거리 증가에 따라 성능이 안정적으로
확장되지 않았다. 이러한 결과는 final decoder를 neural classifier로
대체하기보다, neural model을 PyMatching 앞단의 pre-processing 모듈로
사용하는 방향을 지지한다.

NVIDIA의 Ising-Decoding 계열 연구도 neural pre-decoder와 PyMatching을
결합하는 관점에서 관련이 있다. 다만 본 연구는 explicit local motif
candidate 생성과 patch-head candidate ranking을 사용하며, candidate별
benefit/harm, oracle gap, harmful-edit taxonomy를 직접 분석한다는 점에서
구조적 초점이 다르다.

## 3. 데이터셋 및 노이즈 환경

실험 데이터는 표면 코드 회로를 생성하고, 구성된 noise model을 주입하여
얻은 syndrome sample로 구성된다. 주요 noise family는 `stage_a_si1000`,
`stage_b_local`, `stage_c_corr`이며, 최종 주장은 held-out family인
`stage_c_corr`에서의 성능을 기준으로 한다. `stage_a_si1000`과
`stage_b_local`은 training 및 validation 단계에서 selector adoption
판단에 사용되고, `stage_c_corr`는 일반화 성능을 확인하는 held-out
평가 환경으로 사용된다.

각 샘플은 detector event뿐 아니라 detector mask, geometry metadata,
noise-family indicator, distance/round statistics, event fraction,
physical-noise summary 등으로 구성된 36채널 syndrome/noise volume으로
표현된다. d3와 d5 성공 실험에서 input tensor shape은 각각 `[36, 4, 4, 4]`,
`[36, 6, 6, 6]`이다. 채널 의미는 거리와 무관하게 유지되며, spatial 및
temporal volume 크기만 코드 거리에 따라 달라진다.

본 연구에서 noise 환경은 실험 설정의 중요한 부분이지만, 알고리즘적
기여의 중심은 noise model 자체가 아니라 noise-aware 정보를 포함한
입력 표현과 neural pre-decoder 구조, 그리고 selected-mode 분석이다.

## 4. 제안 방법

제안 모델은 transition-aware neural pre-decoder이다. 전체 추론 경로는
다음과 같다.

```text
36-channel syndrome/noise volume
  -> SyndromeEditPreDecoder 3D residual trunk
  -> local motif candidate set
  -> patch-head CandidateEditSelector
  -> selected local detector edit or raw no-edit fallback
  -> PyMatching
  -> logical_class4 prediction
```

`SyndromeEditPreDecoder`는 3D convolution stem과 세 개의 residual 3D
convolution block으로 구성된다. d3/d5 recipe에서는 hidden channel `24`,
dense hidden dimension `64`, dropout `0.1`을 사용하며 parameter count는
`118834`이다. trunk는 detector-level edit logits, shot-level needs-edit
logits, pooled shot feature를 생성한다. 이 중 pooled shot feature는 candidate
selector의 입력으로 사용된다.

candidate generation은 임의의 큰 syndrome edit을 허용하지 않는다. 성공한
d3/d5 recipe는 `local_motif_selector`를 사용하고, local motif class를 최대
`16`개, top-k motif candidate를 `32`개로 제한한다. 또한
`selector-policy-candidate-mode none`을 사용하여 raw threshold/top-k policy
candidate를 제거한다. identity, 즉 no-edit candidate는 항상 포함된다.
이 설계는 raw PyMatching fallback을 별도 baseline이 아니라 method 내부의
안전 경로로 만든다.

`CandidateEditSelector`는 patch-head 구조를 사용한다. 각 candidate에 대해
selector는 pooled shot embedding, candidate edit weight, neural edit
probability summary, local motif 및 pattern feature, detector geometry
summary, local evidence feature, local patch feature, raw PyMatching 대비
benefit/harm transition feature를 입력으로 받는다. selector는 각 candidate에
score를 부여하지만, 가장 높은 score를 가진 candidate가 항상 적용되는 것은
아니다. candidate branch는 selected-mode safety policy를 통과해야 최종
입력 syndrome에 적용된다.

selector target은 `benefit_harm`이다. 어떤 candidate가 유익한지는 detector
mask를 잘 복원했는지가 아니라, 해당 local edit을 적용한 뒤 PyMatching을
수행했을 때 최종 `logical_class4` 정답 여부가 raw no-edit PyMatching보다
개선되는지로 판단한다. 따라서 본 모델은 final decoder의 behavior를
반영하는 decoder-aware selector이다.

selected-mode adoption policy는 candidate-first safety 구조이다. validation
family에서 candidate branch가 충분한 positive evidence를 보이지 못하거나
harm guard, support guard, plateau guard 등의 조건을 통과하지 못하면 최종
selected mode는 raw no-edit PyMatching으로 fallback한다. 이 fallback은
성능을 보수적으로 유지하기 위한 핵심 구성 요소이다.

## 5. 실험 설정

실험은 d3, d5, d7 세 거리에서 수행했다. 주요 비교 대상은 다음과 같다.

| 비교 항목 | 의미 |
| --- | --- |
| raw PyMatching | local edit 없이 원본 syndrome을 PyMatching에 입력 |
| selected predecoder | selected-mode policy가 선택한 local edit 또는 raw no-edit을 PyMatching에 입력 |
| candidate branch | learned candidate edit을 fallback 없이 적용했을 때의 성능 |
| target local-edit oracle | 가능한 local edit target 중 최선의 성능 |

주요 metric은 held-out `stage_c_corr`에서의 `logical_class4` accuracy이며,
selected predecoder가 raw PyMatching 대비 얼마나 개선되는지를 selected delta로
보고한다. 추가로 selected/candidate branch가 개선한 shot 수와 악화한 shot
수를 함께 분석한다.

fair main baseline은 동일한 predecoder target artifact 위에서 계산한 raw
no-edit PyMatching이다. FLFD-small, M3D-FLFD, RectCNN과 같은 direct neural
decoder는 final architecture 선택의 맥락을 설명하기 위한 context baseline으로
사용하며, 최종 head-to-head 비교의 중심으로 사용하지 않는다.

## 6. 실험 결과

held-out `stage_c_corr`에서의 주요 결과는 다음과 같다.

| distance | raw PyMatching | selected predecoder | candidate branch | target local-edit oracle | selected delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| d3 | `0.928710938` | `0.935302734` | `0.935302734` | `0.992187500` | `+0.006591797` |
| d5 | `0.888671875` | `0.894287109` | `0.891845703` | `0.978515625` | `+0.005615234` |
| d7 | `0.873046875` | `0.873198411` | `0.871531519` | `0.984375000` | `+0.000151536` |

d3와 d5에서는 selected predecoder가 raw PyMatching보다 높은 정확도를 보였다.
d3의 selected delta는 `+0.006591797`, d5의 selected delta는 `+0.005615234`이다.
d3는 seed `0..7` 전체에서 양의 selected delta를 보였고, d5는 selected-mode
policy가 약한 seed에서는 raw no-edit으로 fallback하고 유익한 seed에서만 local
selector를 채택하는 보수적 개선 양상을 보였다.

d3/d5 seed `0..7` robustness 결과는 다음과 같다.

| distance | positive/neutral/harmful selected seeds | mean selected delta | selected improved/harmed |
| --- | ---: | ---: | ---: |
| d3 | `8/0/0` | `+0.006591797` | `205/151` |
| d5 | `2/6/0` | `+0.005615234` | `56/10` |

현재 확인된 d3/d5 seed set에서는 harmful selected seed가 없다. d3는 여덟 seed
모두 local selector를 채택하며 양의 성능 향상을 보였고, d5는 두 seed에서
local selector를 채택하고 여섯 seed에서는 raw no-edit으로 fallback했다. d5의
candidate branch는 seed `4`, `6`에서 harmful이었지만 selected-mode safety가
이를 차단했다. 이는 selected-mode safety가 method의 부가 장치가 아니라 성공
결과를 구성하는 핵심 요소임을 보여준다.

seed-level bootstrap 95% confidence interval은 d3가
`[+0.004516602, +0.008544922]`, d5가 `[+0.000000000, +0.013671875]`이다.
따라서 d3 결과는 확장 seed set에서도 명확히 양수이며, d5 결과는 평균은
양수이나 CI 하한이 0에 닿기 때문에 보수적 selected-mode 개선으로 해석해야
한다.

추가로 seed-level exact sign test와 sign-flip mean test를 계산했다. d3는
8개 seed 모두 양의 selected delta를 보이며 one-sided p-value가
`0.003906250`이다. 반면 d5는 nonzero selected delta가 두 seed에만 존재하고
여섯 seed가 raw no-edit으로 fallback하므로 one-sided p-value가
`0.250000000`이다. 따라서 d5는 positive mean과 non-harmful selected behavior를
보였다고 표현하는 것이 적절하며, d3처럼 uniformly positive라고 표현하지
않는다.

반면 d7에서는 selected delta가 `+0.000151536`에 불과하다. candidate branch는
평균적으로 raw PyMatching보다 낮은 `0.871531519`를 보였고, improved/harmed
shot 수에서도 `161/251`로 harmful edit이 더 많았다. selected mode는 대부분
raw no-edit으로 fallback하여 큰 성능 하락을 막았지만, d7에서 robust learned
recovery를 달성했다고 보기는 어렵다.

## 7. d7 한계 분석

d7 결과는 단순 실패가 아니라 현재 방법의 확장 병목을 보여주는 분석 결과로
해석해야 한다. d7의 target local-edit oracle accuracy는 `0.984375000`으로
높고, mean candidate-oracle delta도 `+0.096679688`이다. 또한 분석한 58개
seed 모두 positive oracle headroom을 갖는다. 따라서 d7 한계의 원인은
candidate space에 유익한 local edit이 없기 때문이 아니다.

oracle-gap recovery 관점에서도 같은 결론이 나온다. d7에서 selected mode가
회복한 oracle gap은 평균 `0.14%`에 불과하지만, candidate-oracle recovery
가능성은 평균 `86.84%`로 크다. 즉 후보 공간 안에는 PyMatching 오류를 줄일
수 있는 local edit이 남아 있지만, 현재 selector가 이를 held-out 환경에서
안정적으로 선택하지 못한다.

문제는 learned selector가 그 유익한 후보를 안정적으로 고르지 못한다는 점이다.
d7 58-seed 분석에서 actual candidate outcome은 positive `6`, neutral `35`,
harmful `17`로 나타났다. mean actual candidate delta는 `-0.001515356`으로
음수이며, 이는 selector가 유익한 후보 대신 neutral 또는 harmful edit을
상위로 ranking하는 경우가 많다는 뜻이다.

validation-to-held-out mismatch는 d7 한계를 더 명확히 보여준다.

| validation candidate class | held-out harmful | held-out neutral | held-out positive |
| --- | ---: | ---: | ---: |
| neutral | `4` | `31` | `1` |
| positive | `13` | `4` | `5` |

validation-positive candidate seed 22개 중 held-out에서 harmful이 된 seed는
13개이고, positive가 된 seed는 5개이다. 즉 validation에서는 좋아 보이는
candidate branch가 held-out `stage_c_corr`에서는 harmful이 되는 경우가 더
많다. false-positive ratio는 `13/22 = 59.09%`이다.

harmful-edit taxonomy에서도 같은 결론이 나온다. harmful candidate seed는 총
17개이며, selected-mode guard는 이 `17/17`개를 모두 raw no-edit으로 fallback하여
최종 selected 성능 하락을 막았다. 이 결과는 selected-mode safety가 d7에서
필수적임을 보여주는 동시에, learned selector가 아직 d7에서 충분히 신뢰할 수
없음을 의미한다.

d7 개선을 위한 여러 경로도 검토되었지만 현재 모델 family에서는 채택하지
않았다. scalar adoption-threshold grid는 `183040`개 policy를 확인했지만
preserve/recover/block sentinel gate를 통과한 policy가 `0`개였다. cross-family
hard positive-vs-negative objective는 seed54 false-positive gate를 해결하지
못했다. candidate-compatibility pairwise top-k는 seed54를 block했지만 seed2의
true-positive candidate branch를 `-0.136718750`까지 악화시켰다. 따라서 d7의
다음 개선은 threshold sweep이 아니라 selector ranking/generalization 자체를
새롭게 설계하는 방향이어야 한다.

## 8. 논의

본 연구의 핵심 의의는 neural model을 final decoder로 대체하는 대신, 강한
matching decoder 앞단에서 decoder-aware local edit을 선택하는 구조를
제안하고 분석했다는 점이다. d3와 d5 결과는 local syndrome edit이 PyMatching의
성능을 실제로 개선할 수 있음을 보여준다. 동시에 d7 결과는 local edit
headroom이 존재하더라도 learned selector가 이를 안정적으로 활용하지 못하면
성능 향상으로 이어지지 않음을 보여준다.

candidate-level 분석은 direct logical classifier보다 더 해석 가능한 정보를
제공한다. candidate branch, selected branch, target oracle을 분리해 보고하면
문제가 candidate generation에 있는지, selector ranking에 있는지, adoption
policy에 있는지 구분할 수 있다. 본 연구에서는 d7의 경우 candidate coverage가
아니라 selector ranking/generalization이 핵심 병목임을 확인했다.

연구의 한계도 명확하다. d7에서는 robust learned recovery를 달성하지 못했으며,
결과는 선택된 noise family와 seed 범위 안에서 해석해야 한다. 본 연구에서는
`stage_a_si1000`과 `stage_b_local`을 학습/검증 family로 사용하고,
`stage_c_corr`를 held-out 평가 family로 사용했다. `config.py`에는 Stage D
leakage surrogate와 Stage E DQLR 설정이 정의되어 있지만, 본 논문의 정량
평가에는 포함하지 않았다. 따라서 noise-family 일반화 주장은 Stage A/B에서
학습한 모델을 Stage C에서 평가한 범위로 제한한다. 또한 d3/d5는 seed `0..7`,
bootstrap CI, paired exact test까지 확인했지만, d5는 CI 하한이 0에 닿고
여섯 seed가 fallback하므로 보수적인 positive-mean selected-mode 결과로
제시해야 한다. 현재 졸업논문 범위에서는 d3/d5의 selected-mode improvement와
d7의 controlled scaling limitation을 함께 제시하는 것이 가장 방어 가능한
결론이다.

## 9. 결론

본 연구는 표면 코드 양자 오류 정정에서 PyMatching과 결합하는 전이 정보 기반
신경망 사전 디코더를 설계하고 평가했다. 제안 모델은 36채널 syndrome/noise
volume, 3D residual neural trunk, local motif candidate generation,
patch-head benefit/harm selector, selected-mode safety policy를 결합하여
local detector edit을 선택한 뒤 PyMatching에 전달한다.

실험 결과, 제안한 selected-mode predecoder는 held-out `stage_c_corr` 환경에서
d3와 d5에 대해 raw PyMatching 대비 각각 `+0.006591797`, `+0.005615234`의
정확도 향상을 보였다. d3는 seed `0..7` 전체에서 양의 selected delta를 보인
반면, d5는 두 seed에서 local edit을 채택하고 여섯 seed에서 raw no-edit으로
fallback하는 보수적 개선 구조를 보였다. 이는 neural pre-decoding이 작은
거리의 표면 코드에서 강한 matching decoder의 성능을 개선할 수 있음을
보여주되, d5 claim은 positive mean과 selected-mode safety 중심으로 해석해야
함을 의미한다.

반면 d7에서는 selected-mode 향상이 `+0.000151536`에 그쳤다. 그러나 d7의
target oracle 및 candidate-oracle 분석은 local edit headroom이 충분히 남아
있음을 보여준다. 따라서 d7의 한계는 candidate coverage 부족이 아니라 learned
selector가 validation-positive false positive를 held-out 환경에서 구분하지
못하는 ranking/generalization 문제로 해석된다.

결론적으로, 본 연구는 neural pre-decoder가 PyMatching의 유용한 front-end가
될 수 있음을 d3/d5에서 확인하고, d7 분석을 통해 더 큰 거리로 확장하기 위해
해결해야 할 selector 설계상의 병목을 제시한다.

## 그림 및 표 캡션 초안

그림 파일은 `PREDECODER_FIGURE_PACKAGE.md`와
`artifacts/figures/predecoder/`에 정리되어 있다.

그림 1. 제안하는 neural pre-decoder와 PyMatching 결합 구조. 36채널
syndrome/noise volume이 3D residual trunk와 local motif selector를 거쳐
selected local edit 또는 raw no-edit syndrome으로 변환되고, 최종 logical
frame은 PyMatching이 예측한다.

파일: `artifacts/figures/predecoder/fig1_predecoder_pipeline.svg`

그림 2. SyndromeEditPreDecoder와 patch-head CandidateEditSelector 구조.
3D trunk가 shot-level feature와 detector-level edit logits를 생성하고,
candidate selector가 local patch feature 및 benefit/harm feature를 이용해
candidate edit을 ranking한다.

파일: `artifacts/figures/predecoder/fig2_model_architecture.svg`

그림 3. d3, d5, d7의 held-out `stage_c_corr` accuracy 비교. d3와 d5는
selected predecoder가 raw PyMatching보다 높은 정확도를 보이나, d7은 거의
raw no-edit에 머문다.

파일: `artifacts/figures/predecoder/fig3_main_accuracy_comparison.svg`

그림 4. d7 candidate-oracle gap과 validation false-positive 구조. 모든 d7
seed에 oracle headroom이 존재하지만, learned candidate branch는 positive,
neutral, harmful outcome으로 분산되며 validation-positive false positive가
주요 실패 유형으로 나타난다.

파일: `artifacts/figures/predecoder/fig4_d7_oracle_gap_false_positive.svg`

그림 5. d7 validation delta와 held-out candidate delta의 seed-level scatter.
validation-positive branch 중 held-out에서 harmful이 되는 경우가 많아, d7
병목이 단순 threshold calibration이 아니라 selector ranking/generalization
문제임을 보여준다.

파일: `artifacts/figures/predecoder/fig5_d7_validation_heldout_scatter.svg`

그림 6. seed-level oracle-gap recovery 분포. d3, d5, d7에서 selected mode와
candidate branch가 target local-edit oracle gap 중 어느 정도를 회복하는지
비교한다. 특히 d7은 selected recovery가 `0.14%`에 그치는 반면
candidate-oracle recovery는 `86.84%`로 커서, candidate coverage가 아니라
selector ranking/generalization이 병목임을 보여준다.

파일: `artifacts/figures/predecoder/fig6_oracle_recovery_distribution.svg`

표 1. 실험에 사용한 noise family와 역할. `stage_a_si1000` 및 `stage_b_local`은
training/validation family이고, `stage_c_corr`는 held-out correlated-noise
evaluation family이다.

표 2. d3/d5/d7 held-out `stage_c_corr` main accuracy table. raw PyMatching,
selected predecoder, candidate branch, target local-edit oracle, selected
delta를 비교한다.

표 3. d3/d5 seed `0..7` robustness table. positive/neutral/harmful selected
seed 수와 improved/harmed shot 수를 제시한다.

표 4. d3/d5 paired statistics table. seed-level exact sign test와 sign-flip
mean test를 사용해 d3는 uniformly positive이고, d5는 positive mean이지만
보수적으로 진술해야 함을 정리한다.

표 5. d7 harmful-edit taxonomy. validation-positive false-positive harmful,
validation nonpositive harmful, high-oracle harmful, broad over-edit harmful,
sparse harmful, severe harmful 유형을 구분한다.

표 6. ablation 및 rejected path summary. direct neural classifier, scalar d7
adoption tuning, cross-family hard-negative objective, candidate-compatibility
top-k가 최종 방향으로 채택되지 않은 이유를 정리한다.

## 최종 편집 TODO

- 학교 양식에 맞춰 장 제목, 표/그림 번호, 참고문헌 형식을 조정한다.
- 초록과 결론의 용어를 학과 스타일에 맞게 다듬는다.
- SVG figure를 학교 양식에서 요구하는 이미지 형식으로 변환할지 결정한다.
- `PREDECODER_REPRODUCIBILITY_PACKAGE.md`의 명령과 artifact path를 부록으로
  옮긴다.
- 최종 제출 전 `artifacts/eval/nn/sedp_final_result_consistency_check.json`의
  `pass=true`, `num_failed=0` 상태를 다시 확인한다.
