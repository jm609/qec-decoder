# 상관 잡음 변화가 국소 syndrome 수정에 미치는 영향

2026-11-15 · 코드·저장 결과·인용 대조 후 보완한 후속 실험계획

실행 코드는 `tools/run_predecoder_noise_shift.py`이며, 본 평가 설정은 개발 맵 점검 후 별도 manifest에 고정한다. 기존 산출물은 역사적 기록으로 유지한다. 수정 입력의 새 Stage C 성능은 [독립 재검증 보고서](PREDECODER_INDEPENDENT_REVALIDATION.md)에 기록했다.

실행 완료. 고정한 표본 수, 교차 평가 결과와 검증 기록은 [d7 잡음 변화 결과 보고서](results/PREDECODER_D7_NOISE_SHIFT_RESULTS.md)를 참고한다.

## 1. 연구 질문과 근거

**같은 회로에서 상관 잡음 강도가 바뀌면, 국소 syndrome 수정의 이득은 후보 생성, 후보 선택, 수정 채택 중 어느 단계에서 줄어드는가?** 같은 shot을 서로 다른 detector error model(DEM)로 디코딩해, shot을 만든 잡음 분포의 변화와 PyMatching에 제공한 잡음 모델의 불일치를 분리한다.

기존 d7 결과에서 학습 모델이 만든 후보의 oracle 개선폭은 **+9.67 pp**였고, 기존 `candidate branch`는 **−0.15 pp**, 최종 채택 정책은 **+0.02 pp**였다. 기존 `candidate branch`에는 shot별 `selector_emit_margin`이 이미 적용되어 있으므로, margin 적용 전 1위 후보의 성능과 구분한다. 검증에서 개선된 22개 실행 중 13개는 held-out에서 악화됐다. 이 기록으로 후보 순위와 채택 조건에 대한 가설을 세운다. [기존 d7 병목 분석](results/PREDECODER_D7_TARGETED_BOTTLENECK_ANALYSIS.md) · [기존 selector 평가](../../decoders/syndrome_edit_predecoder.py#L5173)

저장된 d7 `stage_c_corr`의 첫 1,024 shot에서 같은 matched DEM을 사용한 일반 MWPM과 correlated PyMatching은 각각 894개와 887개를 맞혔다. Correlated PyMatching으로 바꾸면 7개를 고치고 14개를 새로 틀렸다(양측 exact McNemar, **p = 0.189**). 2026-11-15에 저장된 `samples.npz`의 `detector_events`, `observable_flips`와 `detector_error_model.dem`으로 재확인했다(PyMatching 2.3.1, Stim 1.15.0). 이 결과는 두 방식을 현재 d7 데이터에서 실행할 수 있음을 확인한 예비 점검이다. 기준선의 우열은 본 실험에서 다시 평가한다. 데이터 위치는 [기존 manifest](../../artifacts/datasets/dev_class4_d7_2k/manifest.json)의 `stage_c_corr` 항목이다.

## 2. 선행연구와 검증 가설

- Darmawan은 단일 큐비트 국소 잡음과 tensor-network 디코더에서 입력 잡음 모델의 일부를 다르게 지정해 파라미터의 영향을 분석했다. 이번 circuit-level 상관 잡음·MWPM 실험과는 조건이 다르며, prior 불일치를 분리하는 설계의 근거로 참고한다. Manor 등은 여러 잡음 모델과 Willow에서 디코더 순위를 비교했다. 이번 실험에서는 **동일 shot의 국소 수정 전후**에 이 접근을 적용한다. [Darmawan](https://arxiv.org/abs/2403.08706) · [Manor 외](https://arxiv.org/html/2609.04557v1)
- QAdapt는 수백 개의 미학습 복합 잡음 조건과 Willow 데이터에서 학습된 사전 디코더의 전이 성능을 비교했다. 이번 계획의 검증 대상은 고정된 사전 디코더에서 나타나는 후보·순위·채택 손실이다. [QAdapt v2](https://arxiv.org/html/2607.28422v2)
- NVIDIA의 사전 디코더는 회로 오류에 대응하는 보정의 논리 패리티를 최종 결과에 합산한다. 현재 코드의 detector-bit 수정은 입력 변환으로 해석하고, oracle은 지정한 후보 집합에서 가능한 성능으로 해석한다. [NVIDIA 사전 디코더 §IV](https://arxiv.org/html/2604.12841v1#S4) · [현재 타깃 생성 코드](../../tools/build_pymatching_edit_targets.py#L186)

위 문헌은 2026-11-15에 해당 버전의 원문을 확인했다. 서로 다른 회로·지표의 성능 수치를 직접 비교하거나, 본 계획의 신규성을 확정하는 근거로 사용하지 않는다.

| 가설 | 지지하는 관측 | 가설을 약화하는 관측 |
| --- | --- | --- |
| **H1. 잡음 변화가 후보 순위를 약화한다.** | 같은 DEM 계열에서 강제 적용의 순이득이 떨어지고, 후보 oracle과 강제 적용 사이의 차이가 커진다. | 원본 오답 중 정답으로 바꿀 수 있는 shot의 비율부터 떨어지며 순위 차이는 거의 일정하다. |
| **H2. 사전 디코더의 이득 일부는 prior 불일치를 보완한다.** | 같은 평가 shot에서 DEM을 맞추면 원본의 정확도가 오르고, 사전 디코더의 추가 이득이 줄어든다. | DEM을 맞춘 뒤에도 사전 디코더의 추가 이득이 비슷하게 남는다. |
| **H3. 채택 정책이 유익한 수정도 함께 거른다.** | 잡음이 바뀐 조건에서 강제 적용의 순이득이 남고, 최종 정책에서 그 이득이 감소한다. | 강제 적용부터 순이득이 사라지거나, 채택 정책이 피해를 줄이면서 순이득을 보존한다. |

## 3. 회로, 잡음, 데이터 분할

- **회로.** d7, 7 rounds, rotated surface code, 기존 Bell-pair readout을 사용한다. 주 지표는 두 논리 observable이 모두 맞는 shot의 비율인 `logical_class4` 정확도이다. 첫 실험에서 거리를 고정한다.
- **잡음.** 기준 조건 (G_0)은 기존 d7 `stage_c_corr`의 파라미터인 `physical_error_rate=0.0015`, `zz_factor=0.05`, `swap_factor=0.02`를 사용한다. 변화 조건 (G_1)은 `zz_factor=0.25`를 사용한다. 같은 잡음 맵에서는 ZZ 강도만 바꾸고 회로, 기저 잡음, swap-like 항은 동일하게 둔다. 0.25는 시뮬레이션 비교를 위한 계획값이다. 조건별 detector-event 비율과 ZZ 확률을 기록한다. [잡음 모델](../../noise_willowcore.py#L255) · [기존 d7 데이터 manifest](../../artifacts/datasets/dev_class4_d7_2k/manifest.json)
- **잡음 변화의 범위.** `zz_factor`는 오류 확률 자체가 아니다. 코드의 edge별 확률은 `p_corr_zz = zz_factor × p_cz × edge_mult × hotspot_scale`이다. 따라서 G₁은 같은 맵에서 ZZ 항만 5배가 되며 총 오류율도 달라진다. 본 실험은 총 오류율을 고정한 상관 구조 비교가 아니다. local map의 seed·분포·clip, hotspot 위치·배율, swap 활성화 여부를 함께 고정하고, 실제 local multiplier map의 동일성과 ZZ 확률의 5배 관계를 검사한다. 현재 `build_local_rate_maps`는 `willow.local.seed`를 사용하며 `build_correlated_edge_maps`는 `corr.seed`를 사용하지 않으므로, `corr.seed`만 바꿔 독립 맵을 만들지 않는다.
- **데이터 분할.** 기존 d7 기록은 재현 및 가설 수립에 쓴다. 새 맵 2개를 개발용, 새 맵 8개를 최종 평가용으로 예약한다. 맵 하나에서 두 강도에 같은 위치별 잡음 파라미터를 사용하고, 각 강도의 shot은 독립 sampler seed로 생성한다. 모델 seed, 잡음 맵 seed, shot sampler seed를 각각 기록한다. 최종 평가 전에 seed 목록과 실행 설정을 manifest로 고정한다. 여기서 G₀는 새 맵에서 기준 강도를 재현한 조건이다. 기존 모델의 학습 데이터와 동일한 분포라고 가정할 필요는 없다.
- **체크포인트.** 수정 입력으로 재학습한 d7 seed `2`, `11`, `54`, `57`을 고정한다. seed 번호는 기존 비교에서 채택·유해·중립 사례를 참고해 골랐다. 새 가중치에서 같은 분류가 반복된다고 가정하지 않는다. 가중치, motif vocabulary, 후보 top-k, 입력 채널 순서를 보존한다. 재학습 checkpoint에 저장된 margin은 순서대로 `1.25, 2.0, 1.0, 1.0`, motif 수는 `9, 9, 9, 7`이다. 이 차이를 통일하지 않는다. 패널 평균의 해석 범위는 이 네 체크포인트이다. [독립 재검증 규약](../../artifacts/eval/predecoder_revalidation_20261115/protocol.json)

기존 입력은 event 1개, geometry 7개, context 28개의 총 36채널이다. 수정 전 `_build_context_planes`는 `qc_stats.logical_flip_fraction`을 입력에 넣었고, 데이터 생성기는 이를 정답의 평균으로 계산한다. **이 경로는 2026-11-15 감사 후 코드에서 차단했다.** 주 분석에서는 stage·잡음 설명·DEM 플래그·instruction 비율을 해당 맵의 G₀ 설명으로 고정한다. `logical_flip_fraction` 자리에는 수정 코드의 `reserved_label_context=0`을 사용한다. `detector_event_fraction`은 개발 G₀ 두 맵의 전체 shot을 합쳐 계산한 값으로 고정해 모든 평가 맵·강도·디코더에 공통으로 쓴다. 이 값의 출처·분모·채널 순서·고정값을 manifest에 저장한다. 실제 G₁ metadata는 생성·진단용으로 별도 보존한다. event 채널과 syndrome 유래 후보 특징만 shot에 따라 바뀌며 geometry는 동일하다. [context 구성](../../decoders/research_noise_aware_3d.py#L161) · [정답 통계 생성](../../sample_dataset.py#L733)

재현 점검은 기존 저장 metadata와 평가 경로를 사용한다. 본 교차 평가에서는 재학습을 마친 체크포인트와 위의 고정 context를 사용한다. 강도나 DEM에 따라 모델 또는 채택 규칙을 새로 맞추지 않는다.

## 4. 교차 평가 절차

각 맵에서 (D_0)은 (G_0) 회로의 DEM, (D_1)은 (G_1) 회로의 DEM이다. (D_0)은 강도 변화 전의 기준 모델을 고정한 경우를 나타낸다. 이 설계는 기준 맵을 알고 강도 변화가 생기는 상황을 다룬다. 각 조건은 일반 MWPM과 correlated PyMatching으로 평가한다.

여기서 matched는 **원본 shot 생성 회로에 맞는 DEM**이라는 뜻이다. 신경망이 바꾼 syndrome의 분포까지 맞는다는 뜻은 아니다. 따라서 H2는 원본 회로 prior 불일치에 대한 조건부 진단이며, 수정 후 분포에 최적인 디코더를 찾는 실험으로 해석하지 않는다.

| shot을 만든 분포 | (D_0) 적용 | (D_1) 적용 |
| --- | --- | --- |
| (G_0) 기준 강도 | 기준 조건, matched | 강도를 높게 가정 |
| (G_1) 증가한 강도 | 기준 DEM 유지 | 변화 조건, matched |

1. **정합성 확인.** 두 회로의 detector 순서·좌표·측정 참조와 observable 순서를 대조한다. 각 DEM이 같은 shot 배열을 읽고 `[logical_z_flip, logical_x_flip]` 순서의 두 observable을 출력하는지 점검한다. DEM 생성 인자와 분해·근사 설정은 두 강도에서 같게 유지한다. Correlated PyMatching의 구성과 디코딩 양쪽에서 `enable_correlations=True`를 설정하고, 분해된 error의 상관 정보를 보존한다. 기존 `_build_matching`과 decode wrapper는 이 옵션을 전달하지 않으므로 현재 CLI만으로 교차 평가가 구현됐다고 간주하지 않는다. [PyMatching 2.3.1 구현](https://raw.githubusercontent.com/oscarhiggott/PyMatching/v2.3.1/src/pymatching/matching.py)
2. **행동 고정.** 각 shot에서 기존 모델과 D₀ 일반 MWPM으로 후보 집합, selector 점수와 아래 경로별 edit mask를 한 번 구한다. 후보 집합은 no-edit를 포함한다. 동일 shot의 mask는 D₀/D₁ 및 두 디코더에서 재사용한다. G₀/G₁은 독립 shot이므로 서로 다른 mask를 가질 수 있다. family 단위 채택 결정은 체크포인트별로 한 번 정한다. 개발 G₀ 두 맵에서 미리 정한 sampler 순서의 154 shot씩, 총 308 shot을 calibration subset으로 사용한다. 기존 seed-2 저장 평가도 Stage A/B에서 각 154 shot, 합계 308 shot을 사용했다. 나머지 checkpoint의 표본 수도 대조한다. 기존 support/harm guard가 절대 개수 기준이므로 4,096 shot에 기존 cap을 그대로 적용하지 않는다. 맵별 calibration metric은 기존 `_family_system_metric_value`의 `accuracy + 0.001 × macro_f1`을 쓰고 두 맵을 동일 가중한다. 본 평가의 순이득은 이 선택용 metric과 구분해 정확도만으로 계산한다. checkpoint margin과 guard 기준은 고정하고 plateau 판단에 필요한 margin grid만 기존 설정으로 평가한다. G₁과 최종 평가 라벨로 margin·채택 결정을 바꾸지 않는다. 새 G₀에서 보정한 결정이므로 기존 논문의 채택 결정을 그대로 재현하는 경로와 구분한다.
3. **후보별 재디코딩.** 네 셀에서 아래 다섯 경로를 계산한다. 같은 shot에 속한 후보의 수와 실제 matching 호출 수를 기록한다. 평가 입력과 행동을 확정한 뒤 정답을 결합한다. 정답 배열뿐 아니라 정답에서 파생되는 metadata·target score를 바꾸거나 제거해도 입력 tensor·후보·점수·mask·채택 결정이 같아야 한다. 기존 `_build_selector_candidate_bundle`은 특징과 평가용 정답을 함께 만들므로 그대로 호출했다는 이유만으로 이 조건이 충족되지는 않는다. 수정한 syndrome의 디코딩이 실패하면 **그 셀의** 원본 예측으로 되돌리고 후보별·shot별 실패율을 보고한다. 원본 실패는 shot 제외로 처리하지 않고 해당 셀 집계를 중단해 정합성을 확인한다.
4. **재현과 본 평가.** 먼저 기존 데이터에서 no-edit 결과와 기존 평가 경로의 후보·채택 수를 대조한다. 새 개발 맵 2개에서는 강도당 2,048 shot으로 교차 평가와 결과 저장을 점검한다. 이후 새 평가 맵마다 강도당 25,000 shot을 사용한다. 개발 단계에서 후보 재디코딩 비용을 측정하고, 실행 예산에 따라 shot 수를 조정할 경우 최종 평가 전에 근거와 값을 기록한다.

| 경로 | 정의 |
| --- | --- |
| 원본 | 해당 셀의 DEM·디코더로 수정하지 않은 syndrome을 디코딩한다. |
| 1위 강제 적용 | no-edit를 포함한 후보 중 selector utility 최대 후보를 고른다. shot margin과 family gate는 적용하지 않는다. 동점은 no-edit 우선, 나머지는 고정 후보 순서로 처리한다. 1위가 no-edit이면 그대로 둔다. |
| 기존 candidate branch | 동일 후보·점수에 checkpoint의 `selector_emit_margin` 등 기존 shot 규칙을 적용한다. family gate는 적용하지 않는다. |
| 최종 정책 | 기존 candidate branch에 개발 G₀에서 고정한 family gate를 적용한다. 미채택 checkpoint는 모든 셀에서 no-edit를 쓴다. |
| 후보 oracle | 해당 셀에서 하나 이상의 후보가 정답이면 정답으로 센다. 정답은 이 사후 집계에만 사용하며 실행 가능한 정책으로 취급하지 않는다. |

family gate는 checkpoint 전체에 적용되므로, 원본과 최종 정책만 비교하면 전부 no-edit인 패널에서 결과가 항상 0일 수 있다. 이를 채택 규칙의 결과로 보고하고 1위·candidate branch 진단을 함께 남긴다. 실험 중 gate를 풀어 효과를 만들지 않는다.

주 분석은 행동을 고정한 상태에서 **downstream 디코더를 바꿨을 때의 영향**을 측정한다. 변경된 디코더의 전이 정보를 selector에도 다시 넣는 실험은 별도 탐색으로 기록한다. [기존 PyMatching 구성](../../decoders/baseline_pymatching.py#L258) · [후보 집합 구성](../../decoders/syndrome_edit_predecoder.py#L1911)

## 5. 지표와 판정 기준

원본과 각 경로의 차이를 전체 shot 기준으로 계산한다. 순이득은 `100 × (오답→정답 수 − 정답→오답 수) / 전체 shot 수`이며 단위는 pp이다. 모든 경로의 정확도와 개선·악화 개수를 같이 기록한다. oracle 순이득과 함께 **원본 오답 중 후보로 고칠 수 있는 비율**을 보고한다. 두 지표를 같이 보면 원본 자체가 좋아져 oracle의 절대 개선폭이 줄어드는 경우를 구분할 수 있다. 후보 oracle과 1위 강제 적용의 정확도 차이를 순위 손실로 정의한다. `candidate branch − 1위`, `최종 정책 − candidate branch`는 각각 shot margin과 family gate의 순효과이다. 두 값은 양수일 수도 있으므로 일괄적으로 손실이라고 부르지 않는다.

실제 수정률은 `nonzero edit shot 수 / 전체 shot 수`, 수정당 피해율은 `원본 정답을 오답으로 바꾼 수 / nonzero edit shot 수`이다. 채택 실패와 decode fallback 횟수는 따로 센다. 분모가 0인 조건부 지표는 0 대신 `NA`로 표시한다. H1은 순위 손실 증가만으로 지지하지 않고 후보 회복 가능성과 실제 개선·악화 개수도 함께 해석한다.

- **주 비교.** 일반 MWPM에서 `C1 = Δ_1위(G₁,D₀) − Δ_1위(G₀,D₀)`, `C2 = Δ_최종(G₁,D₁) − Δ_최종(G₁,D₀)`를 평가한다. Δ는 각 셀 원본 대비 순이득이다. C1의 음수는 강도 증가에 따른 이득 감소, C2의 음수는 prior를 맞춘 뒤 추가 이득 감소를 뜻한다. C2만으로 H2를 지지하지 않고 원본 정확도의 변화도 확인한다. 주 추정량은 네 checkpoint를 동일 가중한 뒤 8개 맵을 동일 가중한 평균이다. 나머지 셀과 correlated PyMatching 결과는 진단용이다.
- **불확실성.** 주 구간은 8개 맵을 복원 추출하는 paired cluster bootstrap으로 구한다(20,000회, bootstrap seed는 manifest에 고정). 뽑힌 맵의 G₀/G₁, 모든 shot, checkpoint, DEM·디코더 결과를 함께 가져와 C1/C2를 다시 계산한다. 이 주 분석에서는 맵 안 shot을 추가 재표집하지 않는다. 고정 맵에 조건부인 shot bootstrap을 추가하면 보조 구간으로 명시하고, G₀/G₁ shot은 독립 재표집하되 동일 강도의 shot ID는 모든 경로에서 함께 재표집한다. 네 checkpoint는 고정 패널이며 독립 반복 표본처럼 세지 않는다. 맵별·checkpoint별 효과와 맵 하나씩 제외한 민감도를 공개한다.
- **계획 기준.** 주 비교 두 개에는 각각 양측 97.5% 구간을 적용한다. 순이득 차이 **0.25 pp**를 실질적 해석의 계획 기준으로 둔다. 구간 전체가 +0.25 pp 위 또는 −0.25 pp 아래에 있으면 해당 방향의 실질적 차이로 해석한다. 구간 전체가 −0.25~+0.25 pp 안에 들면 실질적 차이가 작다고 해석한다. 그 밖의 경우는 불확실한 결과로 기록한다. 맵 간 변동이 크면 현재 표본으로 판단할 수 있는 정밀도를 함께 제시한다.

97.5%는 두 주 비교에 대한 Bonferroni 기준이며, bootstrap percentile 구간의 명목 수준이다. 독립 맵이 8개뿐이므로 작은 표본에서 구간의 보장이나 충분한 검정력을 주장하지 않는다. 25,000 shot은 실행 계획값이며 0.25 pp를 검출할 수 있다는 뜻이 아니다. 개발 자료로 예상 구간 폭과 비용을 점검한 뒤 맵 수·shot 수·분석법을 확정한다. 본 평가를 연 뒤 유의성에 맞춰 표본을 추가하거나 G₁ 강도를 바꾸지 않는다.

| 본 평가 결과 | 다음 작업 |
| --- | --- |
| 후보 회복 가능성은 유지되고 순위 손실이 증가한다. | 수정 전후 matching 신뢰도를 추가하는 작은 비교 실험을 수행한다. |
| 후보 회복 가능성 자체가 줄어든다. | 해당 shot의 detector 패턴을 분석하고, 이후 DEM 오류 메커니즘 기반 후보와 논리 패리티 추적의 효과를 비교한다. |
| DEM을 맞추자 원본 정확도는 오르고 추가 이득이 줄어든다. | 디코더 prior 불일치에 의존하는 조건과 그 크기를 결과로 정리한다. |
| 강제 적용은 개선되고 최종 정책에서 개선폭이 감소한다. | 개발 데이터에서만 채택 정책을 보정하고, 새 평가 맵으로 효과를 확인한다. |

## 6. 산출물과 완료 기준

새 실행은 `artifacts/eval/noise_shift_d7/<run_id>/`에 보관한다. 설정 manifest에는 코드 버전과 미커밋 diff, 체크포인트·회로·DEM·데이터의 경로와 해시, 모든 seed, 입력 context의 출처와 고정값, margin·gate·calibration shot ID, 패키지 버전을 넣는다. shot ID는 `(map_id, G, sampler_seed, shot_index)`로 식별한다. shot별 후보 mask·utility·순위·margin 통과 여부·family gate·셀별 예측·decode 실패와 정답을 보관한다. 조건별 집계 파일과 **4개 교차 셀의 순이득·oracle·피해율 그림 한 장**을 만든다. 원본과 후보를 중복 디코딩한 호출, batch 호출 횟수, 디코딩한 syndrome 행 수, oracle 분석 비용과 실제 정책 실행 비용은 구분한다. 이 실험만으로 처리 속도 향상을 주장하지 않는다.

완료 조건은 기존 저장 데이터에서 기존 경로를 재현하고, 개발 맵에서 정합성·입력 고정·라벨 비의존성·비용 점검을 통과한 뒤, 최종 평가의 네 교차 셀과 두 디코더 결과를 모두 집계하는 것이다. no-edit 후보의 예측은 각 셀 원본과 같아야 하며, 후보 oracle 정확도는 원본·1위·candidate branch·최종 정책보다 낮을 수 없다. 순이득은 정확도 차이와 개선·악화 개수 식에서 일치해야 한다. 보고서에는 가설별 지지 여부와 불확실성을 기록한다. 결론의 범위는 이 d7 Bell-pair 회로, 두 ZZ 강도, 네 체크포인트이다. 본 계획에서 추가 학습과 실제 Willow 하드웨어 검증은 후속 연구로 둔다.
