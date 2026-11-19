# d7 상관 잡음 변화 교차 평가

2026-11-15 · 재학습 checkpoint seed 2, 11, 54, 57

강한 ZZ 조건에서도 후보 집합에는 원본 PyMatching 오답을 고칠 수 있는 수정이 많이 남았다.
기준 DEM과 일반 MWPM을 고정했을 때 후보 oracle과 1위 적용의 정확도 차이는 18.731 pp에서
20.045 pp로 커졌다. 1위 적용 자체가 강도 증가로 얼마나 악화됐는지는 주 비교 구간에서
확정하지 못했다. 개발 맵에서 정한 최종 정책은 네 모델 모두 원본을 유지했다.
[실험계획](../PREDECODER_NOISE_SHIFT_EXPERIMENT_PLAN.md)

## 실행 조건

d7, 7 rounds, rotated surface code, Bell-pair Z readout을 사용했다. G₀와 G₁의
`zz_factor`는 각각 0.05와 0.25이고, D₀와 D₁은 해당 회로에서 얻은 DEM이다. 같은 맵의
두 회로는 기하·detector·observable 정의, 기저 잡음, local multiplier와 swap 항이
같았다. 모든 edge의 ZZ 확률이 5배임을 확인했다. ZZ 확률 증가에 따라 총 잡음도 변한다.

개발 맵 2개에서는 강도당 2,048 shot을 사용했다. 맵 하나의 구동 시간이 약 527–530초였다.
4개 작업을 병행하는 약 60분 예산을 기준으로, 예약한 독립 맵 8개를 유지하면서 강도당
5,376 shot을 본 평가 전에 고정했다. 두 강도 합계 86,016개 shot이다. 같은 강도와 맵의
shot은 네 모델과 네 디코딩 설정에서 재사용했다. 본 평가 실행 시간은 약 69분이었다.
[동결한 평가 설정](../../../artifacts/eval/noise_shift_d7/20261115_retrained/frozen_evaluation.json)

모델 입력에서 정답 집계 채널은 0으로 유지했다. detector-event fraction 입력은 개발 G₀
4,096 shot에서 얻은 4.274%로 고정했다. 본 평가에서 관측된 비율은 G₀ 4.297%,
G₁ 4.419%였다. 후보와 점수는 D₀ 일반 MWPM을 기준으로 한 번 구하고, 각 shot의 후보와
선택을 네 디코딩 설정에서 재사용했다. 최종 채택은 개발 G₀의 첫 154 shot씩, 합계 308 shot으로
결정했다. 네 checkpoint의 margin은 1.25, 2.0, 1.0, 1.0이었으며, 이 margin에서 선택된
비영 수정은 calibration 전체에서 각각 1, 3, 1, 2건이었다. 기존 guard를 적용한 결과는
네 모델 모두 원본 유지였다.
[개발 기록](../../../artifacts/eval/noise_shift_d7/20261115_retrained/development.json)

## 교차 평가 결과

각 셀은 8개 맵과 네 checkpoint를 동일 가중했다. 원본 정확도 단위는 %, 순이득은 각 셀의 원본
대비 pp이다. 1위는 no-edit를 포함한 utility 최대 후보, margin 후는 shot별 기존 margin을 통과한
경로, 최종은 개발 맵의 채택 결정까지 적용한 경로다.

| 디코더 | 분포 / DEM | 원본 정확도 | 1위 순이득 | margin 후 | 최종 | 후보 oracle 순이득 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 일반 MWPM | G₀ / D₀ | 87.284 | −8.033 | −0.090 | 0.000 | +10.699 |
| 일반 MWPM | G₀ / D₁ | 87.286 | −6.953 | −0.068 | 0.000 | +10.089 |
| 일반 MWPM | G₁ / D₀ | 86.391 | −8.501 | −0.091 | 0.000 | +11.543 |
| 일반 MWPM | G₁ / D₁ | 86.416 | −7.289 | −0.073 | 0.000 | +10.906 |
| Correlated PyMatching | G₀ / D₀ | 87.412 | −7.138 | −0.087 | 0.000 | +10.735 |
| Correlated PyMatching | G₀ / D₁ | 87.421 | −6.177 | −0.058 | 0.000 | +10.095 |
| Correlated PyMatching | G₁ / D₀ | 86.496 | −7.521 | −0.080 | 0.000 | +11.534 |
| Correlated PyMatching | G₁ / D₁ | 86.507 | −6.449 | −0.063 | 0.000 | +10.857 |

| 사전 지정 일반 MWPM 비교 | 차이 (pp) | 양측 97.5% 맵 bootstrap 구간 (pp) | 판정 |
| --- | ---: | --- | --- |
| C1 · D₀ 고정, G₁−G₀의 1위 순이득 | −0.469 | [−0.939, +0.065] | 불확실 |
| C2 · G₁ 고정, D₁−D₀의 최종 정책 순이득 | 0.000 | [0.000, 0.000] | 원본 유지로 구조적으로 0 |

맵을 단위로 20,000회 재표집했다. C1 구간은 실질적 차이 기준 −0.25 pp와 0을 모두 포함한다.
맵 하나씩 제외한 C1 평균 범위는 −0.606~−0.376 pp이다. C2는 수치상 ±0.25 pp 안에 들었으나
보정을 실행한 모델이 없으므로, 수정이 활성화된 정책에서 prior 불일치의 영향을 검정하지 못했다.
[교차 평가 그림 · 활성화한 Anaconda `fp`에서 렌더링](../../../artifacts/eval/noise_shift_d7/20261115_retrained/fp_activated_render/crossed_diagnostics.png)

### 가설별 해석

- **H1 · 순위 손실.** D₀ 일반 MWPM에서 원본 오답 중 후보로 고칠 수 있는 비율은 G₀ 83.93%,
  G₁ 84.70%였다. 회복 가능성 변화는 +0.776 pp이고 진단용 97.5% 구간은 [−0.100, +1.844] pp였다.
  같은 조건에서 후보 oracle과 1위 사이의 정확도 격차는 +1.313 pp 늘었다
  (진단용 구간 [+0.968, +1.613] pp). 후보 회복 범위의 뚜렷한 감소보다 순위 손실 증가가 관측됐다.
  다만 1위 순이득의 강도 효과에 대한 주 판정은 불확실하다.
- **H2 · DEM prior 불일치.** G₁에서 D₁을 쓰면 일반 MWPM 원본 정확도는 +0.026 pp 변했다
  (진단용 구간 [−0.019, +0.072] pp). 1위 적용의 피해는 −8.501 pp에서 −7.289 pp로 줄었고,
  최종 정책의 추가 이득은 두 DEM에서 0이었다. DEM 선택이 같은 수정의 피해를 바꿀 수는 있지만,
  관측된 사전 디코더 이득을 prior 불일치의 보완으로 설명할 근거는 얻지 못했다.
- **H3 · 채택이 거르는 수정.** G₁/D₀ 일반 MWPM에서 margin을 적용하면 수정률이 20.72%에서
  1.19%로, 수정당 피해율이 70.48%에서 53.77%로 줄었다. margin 후 고친 것은 948건,
  새로 틀린 것은 1,104건이었다. 최종 gate는 이 둘을 모두 걸렀다. seed 2의 margin 후 순이득은
  +0.235 pp였지만, 패널 평균은 −0.091 pp였다. 일부 유익한 수정도 사라졌으나, 전체 패널에서
  양의 순이득을 제거했다는 결과는 없었다.

## 검증과 산출물

64개 예측 파일과 256개 디코더 셀에서 저장된 정답·예측을 다시 대조했다. 후보 mask와 순위,
no-edit 예측과 각 셀 원본의 일치, 최종 gate, 개선·악화 수, oracle 상한, 주 비교 구간을
재계산했다. 후보는 shot당 평균 33개였고, 수정 syndrome의 디코딩 실패는 0건이었다.
후보 생성 중 D₀ transition 특징 계산에는 344,064회 batch 호출과 11,354,112개 syndrome 행이
필요했다. 원본·oracle 교차 평가에는 별도로 3,072회 batch 호출과 45,416,448개 행이 필요했다.
실험의 최종 정책은 원본 디코딩만 실행하므로 이 교차 분석 비용을 배포 비용으로 해석하지 않는다.

기존 1,024-shot 기준 점검에서 일반 MWPM과 correlated PyMatching의 정답 수 894/887을 재현했다.
개발 입력의 정답 배열과 label 유래 metadata를 바꿔도 입력·후보·점수·mask가 같았다.
회귀 테스트 24개를 통과했다. 수치 계산은 Anaconda `fp`의 `python.exe`를 직접 호출해
실행했다. 이후 Conda로 `fp`를 활성화해 재확인한 결과, PyTorch와 Matplotlib을 한 프로세스에
로드하면 OpenMP 런타임 충돌이 재현됐다. 활성화한 `fp`의 별도 렌더링 프로세스에서는
`tools/render_predecoder_noise_shift.py`가 같은 집계 JSON으로 그림을 정상 생성했다.
[집계 JSON](../../../artifacts/eval/noise_shift_d7/20261115_retrained/summary.json) ·
[재검증 JSON](../../../artifacts/eval/noise_shift_d7/20261115_retrained/verification.json) ·
[실행 코드와 diff](../../../artifacts/eval/noise_shift_d7/20261115_retrained/run_source.zip)

결과 범위는 이 d7 회로, 두 ZZ 강도, 네 checkpoint와 사전 고정한 context·채택 규칙이다.

후속 실험에서는 별도 평가 맵에서 matching 비용 항을 시험했다. 수정할 shot을 고정했을 때
후보 mask는 일부 달라졌지만 논리 예측은 동일했다.
[후속 재순위 평가](PREDECODER_D7_MATCHING_COST_RERANK_RESULTS.md)
