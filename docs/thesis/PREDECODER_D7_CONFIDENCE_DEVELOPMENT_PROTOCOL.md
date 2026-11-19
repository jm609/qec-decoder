# d7 selector 신뢰도 개발 규약

2026-11-16 · 1단계 완료. 2단계 비교 규약을 새 맵 생성 전에 확정했다. 최종 평가 규약은 논문 완성 계획의 4단계에서 정한다.

## 검증할 기여

| 주장 | 현재 근거와 남은 검증 |
| --- | --- |
| d7에서 잘못된 순위와 채택 거부의 영향을 같은 shot에서 나눠 측정할 수 있다 | 완료한 A에서 네 checkpoint, 두 ZZ 강도를 교차 평가했다. 강한 ZZ에서 1위가 원본 정답을 해친 25,126쌍 모두에 정답을 보존하는 다른 비영 후보가 있었다. 강도 증가의 주 효과는 불확실하다. [잡음 변화 결과](results/PREDECODER_D7_NOISE_SHIFT_RESULTS.md) · [논리 전이 진단](results/PREDECODER_D7_LOGICAL_TRANSITION_DIAGNOSTIC.md) |
| 후보별 detector 정보와 matching 신뢰도가 직접 전환 규칙보다 유용할 수 있다 | 현재 paired head는 D₀ 일반 MWPM의 새 맵에서 G₀ +0.203 pp, G₁ +0.148 pp였다. 보존 후보를 선택한 경로와 원본 유지 경로의 논리 예측이 같았다. 2단계에서 특징 제거와 단순 대조군으로 기여를 구분한다. [강한 ZZ](results/PREDECODER_D7_PAIRED_SELECTOR_INDEPENDENT.md) · [약한 ZZ](results/PREDECODER_D7_PAIRED_SELECTOR_WEAK_ZZ.md) |
| 실제 수정하는 정책에서 prior와 디코더가 이득의 크기를 바꿀 수 있다 | A의 최종 정책은 전부 원본을 유지했다. 3단계에서 활성 정책의 동일 mask 비교와 전체 경로 비교를 수행한다. [논문 완성 계획](PREDECODER_PAPER_COMPLETION_PLAN.md) |

이번 논문의 범위는 d7 Bell-pair Z readout의 두 observable, 두 ZZ 강도, 고정한 모델 패널이다. 아래 비교로 뒷받침되는 범위에서 같은 shot의 수정 선택을 기여로 제시한다. 단순 규칙의 결과가 비슷하면 진단과 선택 규칙의 단순화를 중심으로 서술한다.

| 선행연구 | 확인한 방법 | 우리 실험에 반영할 비교 |
| --- | --- | --- |
| Gidney 외, *Yoked surface codes* | 같은 syndrome에서 논리 결과를 달리 제약한 최소 matching 비용을 비교해 complementary gap을 구한다. 그 gap을 바깥 코드의 decoder에 전달한다. | 원본·수정 syndrome 각각에서 클래스별 비용을 검증한다. 기존 수정 전후 최적 비용 실험과 별개의 특징이다. [Complementary gaps](https://arxiv.org/html/2312.04522) |
| Toshio 외, *Decoder Switching* | 빠른 decoder의 신뢰도로 belief-matching 같은 강한 decoder를 호출한다. | 같은 적용 대상에서 원본 정보만 쓰는 Z 전환을 대조한다. 적용 대상을 고르는 후보 생성 비용도 시간에 포함한다. [논문 원문](https://arxiv.org/html/2510.25222v1) |
| Hoyt 외, *Decoder Model Compatibility* | prior가 틀리면 logical gap이 보정되지 않을 수 있다. minimum matching weight를 syndrome과 모델의 호환성을 나타내는 특징으로 써 postselection을 비교한다. | raw 신뢰도 대조군에 raw 최적 비용을 포함하고, 두 ZZ 강도에서 같은 gap 구간의 관측 오류율을 비교한다. [논문 원문](https://arxiv.org/html/2609.29018v1) |
| Darmawan 및 Manor 외 | 각각 물리 잡음과 tensor-network 디코더 모델의 불일치, 잡음 모델과 Willow 간 디코더 순위 전이를 평가한다. | A를 근거로 사용하고 3단계에서 실제 적용한 동일 수정을 여러 DEM으로 다시 디코딩한다. [Darmawan](https://arxiv.org/html/2403.08706) · [Manor 외](https://arxiv.org/html/2609.04557v1) |
| Knapen 외, *Generalized qLDPC Predecoding* | Arqade는 DEM 오류마다 detector와 observable의 변화를 함께 기록한다. | C를 시작할 경우에만 물리 오류 후보의 observable 보정, oracle 회복률과 후보 수를 비교한다. [논문 원문](https://arxiv.org/html/2605.03180v2) |

위 표는 직접 관련된 기법의 비교다. 기존 기법 전반에 대한 최초성 주장은 두지 않는다.

## 기존 방법과 통제 조건

입력은 d7, 7 rounds, `p=0.0015`, G₀/G₁ `zz_factor=0.05/0.25`의 detector event 벡터 `s`다. 각 로컬 잡음 맵의 G₀ 회로에서 얻은 DEM D₀로 일반 MWPM `D`를 구성한다. 잡음 맵은 G₀/G₁에 공통이며 sampler seed는 다르다. 36채널 입력에서 `reserved_label_context=0`을 유지한다. 전체 패널의 detector-event fraction context는 이전 A 개발 G₀ 값 `0.042736961728050596`으로 고정한다. shot 자체의 event와 기하는 그대로 사용한다.

1. `r = D(s)`를 구한다. 재학습 checkpoint에서 고정한 3D trunk와 selector로 no-edit를 포함한 mask `m`과 utility를 얻는다. 수정 후 예측은 `q(m) = D(s XOR m)`이다. 동일한 원본·후보 예측을 모든 비교군에서 재사용한다.
2. observable 순서는 `[Z flip, X flip]`, 클래스 값은 `2·Z flip + X flip`이다. top-1이 비영 mask이고 `q(top1) XOR r = 2`이며, `r`과 같은 예측을 돌려주는 비영 후보가 있는 shot을 비교 대상으로 삼는다. 보존 후보 중 utility가 가장 큰 행을 택한다. 기존 코드의 동률 규칙은 먼저 생성된 후보 우선이다.
3. head는 두 예측 중 하나가 정답인 개발 사례에서만 학습한다. 양성은 Z 전환으로 원본 오답을 고친 사례, 음성은 Z 전환으로 원본 정답을 해친 사례다. 평가는 이외의 shot도 포함한다. log-odds가 0 이상이면 Z 후보를 선택한다. 그 미만이면 고정 수정 경로는 보존 후보, 원본 유지 허용 경로는 no-edit를 선택한다. 비교 대상 밖에서 각각 top-1과 기존 checkpoint의 margin 경로를 사용한다. 이 margin은 shot 단위 규칙이다.
4. 새 패널은 d7 재학습 seed 0–7 전체다. head가 모델 식별자를 사용하는 비교에서는 seed 0–7 순서의 8차원 one-hot을 공통으로 쓴다. 기존 네 checkpoint용 head는 과거 자료 재현에 사용하고 새 패널의 head는 새 개발 맵에서 다시 학습한다. `fit_head`의 L2 `0.01`, 학습 데이터의 평균, `max(표준편차, 1)`, cutoff 0을 전 비교군에 고정한다. 공통 적용 대상과 후보를 바꾸지 않는다.

기존 구현은 학습용 bundle 함수를 호출할 때 정답 대신 0을 넘긴다. 평가 정답은 선택 후 집계에 사용한다. 아래 점검은 변경한 정답·정답 집계가 기존 경로의 결과에 영향을 주지 않았음을 확인했다. 2단계의 새 head와 특징 경로에서도 동일 검사를 수행한다. [후보 점수 계산](../../tools/revalidate_predecoder.py) · [현재 paired head](../../tools/run_d7_paired_selector.py) · [추론 경로](../../tools/evaluate_d7_paired_selector_weak.py)

## 데이터 사용 이력과 새 seed 예약

| 용도 | 맵과 sampler | 사용 범위 |
| --- | --- | --- |
| 기존 trunk/selector 학습 | 고정 Stage A/B 파일, Stage B local seed 12345. 당시 sampling seed는 두 파일 모두 `null` | 기록된 파일 해시를 검증해 학습 자료를 식별한다 |
| 이전 개발 | 783000–783001, sampler 1783000–1783003 | 기존 네 checkpoint의 head 학습·설계 자료. 512-shot 점검과 신뢰도 구현 점검에 사용 가능 |
| 이전 평가·진단 | 독립 재검증 780700–780707, A 783100–783107, 비용 재순위 784100–784107, 전이 재순위 785100–785107, paired 786100–786107 | 각 당시 평가의 근거. 새 head의 최종 평가에서는 제외 |
| 2단계 개발·교차검증 | **787000–787003**, sampler **1792000–1792007**, G₀부터 G₁ 순서 | 맵·강도당 1,024 shot. 4개 맵을 한 개씩 제외해 G₁ 3개로 학습하고 제외한 맵의 G₁·G₀를 평가한다 |
| 2단계 별도 보정 확인 | **787100–787101**, sampler **1792200–1792203**, G₀부터 G₁ 순서 | 맵·강도당 1,024 shot. 개발 G₁ 4개로 다시 학습한 규칙을 적용해 gap 구간별 오류율과 이득을 기술한다. 특징·cutoff·방법 선택에는 사용하지 않는다 |
| 4단계 최종 평가 | 새 맵·sampler seed 및 표본 수는 4단계 시작 전에 동결 | 위 자료와 겹치지 않게 배정하고 방법을 고정한 뒤 생성한다 |

기존 case 102개의 설정, sampler seed, 기록된 sample 해시와 case 파일 해시를 목록화했다. 관련 Stage A/B 입력과 d7 checkpoint 8개의 해시를 원래 기록과 대조했다. 신규 예약은 이 목록과 독립 재검증 seed와 대조했다. 새 데이터 생성 후에는 `case.json`의 설정 및 sample·metadata·DEM·회로 해시를 등록한 뒤 실행한다. [데이터 등록부](../../artifacts/eval/d7_confidence_development/20261116/data_registry.json)

## 2단계 비교와 중단 기준

클래스 `c`의 비용 `W(s,c)`는 **같은 D₀ 일반 MWPM 그래프**에서 syndrome과 두 observable parity를 만족하는 edge 집합의 최소 비용이다. 네 클래스의 최소 비용과 도달 불가 여부를 구한다. 제약 없는 PyMatching이 낸 클래스의 비용이 네 비용의 최솟값과 일치하는지 확인하고, 동률이면 기존 decoder의 선택을 유지한다. 작은 그래프의 edge 조합 전수 검사와 실제 d7 graph의 소규모 독립 검증을 수행한다. 평행 edge가 PyMatching에 병합된 경우 실제 디코더가 사용하는 가중치와 observable을 기준으로 대조한다. solver가 완료하지 못한 사례를 도달 불가로 처리하지 않는다.

신뢰도 후보는 Z 전환에 대응하는 **`W(s, c XOR 2) − W(s, c)`**다. 원본과 두 후보의 자체 syndrome 및 자체 예측 클래스에서 각각 계산한다. 검증된 도달 불가 Z 대안만 별도 indicator로 표현하고 비용 차이 자리는 0으로 둔다. raw의 최소 비용은 `W(s,r)`다. 각 gap은 최적 edge 집합 사이의 비용 차이로 보고한다. 두 관측량을 동시에 바꾸는 가설까지 포함한 정답 확률로 해석하지 않는다.

| 비교군 | 입력과 판단 |
| --- | --- |
| Raw / 기존 margin | 전 shot에서 raw 및 기존 margin 결과를 산출한다 |
| Paired, detector 제거 | 두 편집 후보의 모델 utility 차이, edit weight, 기하·mask, raw 클래스, 모델 ID. 기존 control 재사용 |
| Paired detector 기준 | 위 입력에 두 후보의 detector patch와 수정 위치의 활성 event 수 추가 |
| Paired detector + gap | 위 입력에 원본·Z 후보·보존 후보의 gap과 도달 불가 indicator 추가 |
| Z 후보 대 no-edit | Z 후보와 no-edit의 utility 차이, Z 후보 자체의 weight·기하·mask·detector patch, raw 클래스, 모델 ID |
| 원본 정보 Z 전환 | 전체·라운드별 detector event 비율, raw 클래스, 모델 ID. 검증에 성공하면 raw 최소 비용과 raw gap·indicator를 넣은 변형도 비교한다 |

여섯 학습 변형을 상한으로 한다. raw와 기존 margin은 학습 없이 평가한다. 원본 정보 head는 비교 대상에 대해 `r`과 `r XOR 2` 중 하나를 예측한다. 마스크가 필요한 고정 수정 진단에서는 같은 후보 행으로 대응시킨다. 원본 정보 head도 공통 적용 대상을 정하는 데 후보 생성이 필요하므로 이 비용을 보고한다. 후보 생성 없이 적용할 범위를 넓히는 실험은 별도로 둔다.

- **기록 지표.** 전 shot의 logical_class4 정확도 차이(pp), 개선·악화 수, 실제 예측 변경 수와 적용률을 맵·checkpoint별로 기록한다. 고정 수정과 원본 유지 허용을 모두 계산하고 방법 선택에는 G₁ 원본 유지 허용의 4개 fold 평균을 쓴다. G₀ 결과는 전이 진단이다. gap 보정 확인은 학습 G₁의 raw gap 사분위 경계를 고정해 별도 보정 맵에 적용한다. 도달 불가와 동률은 별도 집계한다.
- **선택 기준.** 단순 대조군은 기존 margin, Z 후보 대 no-edit, 원본 정보 전환 변형 중 G₁ 평균이 가장 높은 방법이다. 차이가 0.05 pp 이내면 비용이 낮은 방법을 택한다. Paired detector 기준이나 gap 변형은 직접 대조군보다 평균 0.05 pp 이상 높고 4개 맵 중 3개 이상에서 이길 때만 더 복잡한 구성을 채택한다. gap 변형은 paired detector 기준에도 이 조건을 적용한다. 이 기준은 개발 선택 기준이며 통계적 유의성 판정이 아니다. paired의 이득을 확인하지 못해도 기존 paired 기준과 단순 대조 결과를 함께 남긴다.
- **실행 상한.** 기존 경로의 후보는 각 맵·강도·모델에서 한 번만 계산하고 비교군에서 재사용한다. 2개 이하 worker와 각 2개 PyTorch 스레드를 기본으로 한다. 총 192개의 512-shot 배치에 기존 경로 시간으로 직렬 약 33분이 필요하다. 후보 준비에 60분, gap 검증·계산에 30분의 실행 한도를 둔다. gap은 처음에 과거 개발 데이터의 최대 64개 고유 syndrome을 검사해 전체 계산 시간을 추정한다. 한도 안에서 신뢰도를 검증·산출할 수 없으면 gap 변형만 보류하고 나머지 비교를 마친다. 6개 학습 변형 밖의 특징·threshold 탐색은 이 개발 단계에서 종료한다.

2단계 산출물은 클래스 비용 검증 기록, 맵·모델별 ablation 표, 비용과 실패 수, 다음 단계에서 쓸 방법 및 단순 대조군의 선택 기록이다. 결과에 따라 이득이 확인된 범위와 3단계에서 검증할 prior 효과를 명시한다.

## 1단계 실행 점검

Anaconda `fp`, PyTorch CPU 스레드 2개, `MKL_THREADING_LAYER=SEQUENTIAL`에서 이전 개발 맵 783000의 G₁ 첫 512 shot과 checkpoint 2를 사용했다. 첫 실행 후 두 번째 실행에서 raw 3.28 ms, 기존 selector 10.25초, selector 내부 매칭 0.114초였다. 최대 프로세스 working set은 463.6 MiB였고 16,896개 후보를 처리했다. 한 배치에서 관측한 시간과 메모리로 개발 예산을 정했다. 새 gap 계산의 비용은 2단계에서 측정한다.

두 observable 정답과 `logical_flip_fraction`을 바꿨을 때 입력 텐서, 후보 mask, utility, 후보 예측 및 선택 행이 같았다. 저장된 개발 후보와 utility, 기본 PyMatching 예측, top-1·margin 행도 재현했다. [환경·입력 해시·시간 기록](../../artifacts/eval/d7_confidence_development/20261116/stage1_smoke.json)
