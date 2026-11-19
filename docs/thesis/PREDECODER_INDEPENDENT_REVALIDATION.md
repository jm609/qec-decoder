# 입력 수정 후 predecoder 독립 재검증

2026-11-15. 정답 집계 채널을 제거한 모델을 다시 학습하고 새 Stage C 맵에서
원본 PyMatching과 비교한다. 실험 설계와 seed는 최종 데이터 생성 전에
[protocol.json](../../artifacts/eval/predecoder_revalidation_20261115/protocol.json)에 저장했다.

## 학습과 독립 평가

- 거리는 d3, d5, d7이며 rounds는 distance와 같다. 과거 Stage A/B의 1,024 shot씩을
  그대로 사용한다. 각 model seed에서 train/validation/internal test는 717/154/153이다.
  입력의 `reserved_label_context`는 0이다. 후보·채택 보정에는 Stage A/B만 사용한다.
- d3·d5는 legacy seed4에 기록된 설정, d7은 support-guard seed2의 설정을 거리별로
  고정한다. 서로 다른 예전 seed의 설정을 섞지 않는다. 거리마다 model seed 0..7을
  재학습한다. d7 seed 11, 54, 57은 잡음 변화 진단용으로 추가하며 주 평가에는 넣지 않는다.
- 모든 거리의 seed 0..7 checkpoint를 저장하고 해시를 고정한 뒤, 거리별로 새 Stage C
  잡음 맵 8개를 생성한다. 각 맵에서 d3·d5는 6,144 shot, d7은 4,096 shot을 샘플링한다.
  회로는 Bell-pair Z readout, `p=0.0015`, `zz_factor=0.05`이다. 동일 거리와 맵의 모든
  모델이 같은 shot과 matching DEM을 사용한다.
- d3 이전 개발 기록에서 paired discordance는 356/8,192였다. 이를 이용한 shot 수준
  근사에서 49,152 shot의 양측 97.5% 반폭은 약 0.21 pp이다. 맵 간 변동은 이 계산에
  포함되지 않으므로 관측된 구간 폭을 그대로 보고한다.
- 주 비교는 d3·d5의 원본 대비 최종 정책 `logical_class4` 순이득이다. 각 맵에서 8개
  model seed를 동일 가중하고, 맵마다 얻은 순이득을 동일 가중한다. 양측 map-level
  t 구간은 d3·d5 각각 97.5%, d7 보조 분석은 95%이다. 모델 seed 반복을 독립 데이터
  표본으로 세지 않는다. d3·d5는 하한이 0보다 크면 개선, 상한이 0보다 작으면 악화,
  그 밖에는 불확실로 보고한다. d7은 효과 크기·채택 여부·구간을 기술한다.

## 실행 기록과 순서

산출물은 `artifacts/eval/predecoder_revalidation_20261115/` 아래에 저장한다. 기존
checkpoint, dataset, 논문 수치 파일과 경로가 겹치지 않는다. `complete.json`은 seed별
학습 결과와 checkpoint 해시 검증 완료를 뜻한다. 최종 평가에서 저장하는
`predictions.npz`에는 동일 shot 순서로 원본 예측과 각 seed의 최종 예측이 들어간다.

```powershell
$python = 'C:/Users/82108/anaconda3/envs/fp/python.exe'
$run = 'artifacts/eval/predecoder_revalidation_20261115'
& $python tools/revalidate_predecoder.py train --out $run --distance 3
& $python tools/revalidate_predecoder.py train --out $run --distance 5
& $python tools/revalidate_predecoder.py train --out $run --distance 7 --seeds 0 1 2 3 4 5 6 7 11 54 57
& $python tools/revalidate_predecoder.py evaluate --out $run
```

생성 코드 점검에 사용한 128-shot 맵의 seed는 최종 평가 목록에서 제외한다. 이 맵에서
기준·증가 ZZ 강도 회로의 이상 회로가 같고 DEM이 다름을 확인했다. 독립 평가 harness의
선택 mask에서 얻은 개선·악화 건수는 같은 입력에 대한 기존 평가 CLI 결과와 일치했다.

추가 검증에서는 수정 입력으로 재학습한 각 거리의 seed 0을 불러와, 최종 평가용과 다른
맵 2개(맵당 64 shot)에서 독립 평가 전체 경로를 실행했다. 저장된 원본·최종 예측의
해시와 정답 배열에서 개선·악화 수를 다시 계산해 집계 결과와 일치함을 확인했다.
[검증 기록](../../artifacts/eval/predecoder_revalidation_20261115/engineering_smoke/stage_c_end_to_end/smoke_verified.json)

d7에서는 재학습한 seed 0과 개발용 맵을 사용해 ZZ 두 강도, DEM 두 종류, PyMatching
두 모드의 조합에서 두 observable을 정상 출력함을 확인했다. 이상 회로와 local 맵 설정은
동일하고 ZZ 계수만 5배였으며, 입력의 정답 집계 채널은 0이었다.
[d7 교차 점검 기록](../../artifacts/eval/predecoder_revalidation_20261115/engineering_smoke/d7_cross/smoke.json)

## 독립 평가 결과

주 평가용 24개 checkpoint의 해시를 고정한 뒤 거리별로 독립 맵 8개를 평가했다.
정확도는 seed와 맵을 각각 동일 가중한 평균이다.

| 거리 | 평가 shot | 원본 정확도 | 최종 정책 정확도 | 순이득 (pp) | 맵 단위 구간 (pp) | 후보 적용 seed |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| d3 | 49,152 | 92.910% | 95.169% | +2.260 | 97.5% [+1.688, +2.832] | 8/8 |
| d5 | 49,152 | 90.513% | 90.535% | +0.022 | 97.5% [+0.004, +0.040] | 1/8 |
| d7 | 32,768 | 87.543% | 87.543% | 0.000 | 기술적 95% [0.000, 0.000] | 0/8 |

- d3는 8개 맵 모두에서 평균 순이득이 양수였고, 사전 기준의 개선 조건을 충족했다.
  같은 모델·shot 쌍을 비교했을 때 개선 14,764건, 악화 5,879건이었다.
- d5도 사전 기준의 개선 조건을 충족했다. 순이득은 +0.022 pp로 작으며, 후보를 적용한
  seed 6의 개선 673건과 악화 587건에서 나왔다. 맵 하나에서는 평균 순이득이 음수였다.
- d7은 Stage A/B 채택 규칙이 8개 모델 모두를 원본 경로로 보냈다. 새 Stage C의 모든
  예측이 원본 PyMatching과 일치했다. `[0, 0]` 구간은 고정 정책이 모든 평가 shot에서
  원본을 반환한 결과이며, 미채택 후보의 효용은 이 평가에서 측정하지 않았다.

이번 독립 평가에서 d3는 기존 기록보다 큰 이득, d5는 훨씬 작은 이득을 보였다.
정답 집계 채널 제거와 재학습, 새 맵 평가를 함께 적용했으므로 수치 차이의 원인을
어느 한 변경으로 특정할 수 없다. 구간은 고정된 모델 8개에 조건부이며 맵 간 변동을
사용한다. 모델을 여러 번 적용한 같은 shot은 독립 표본으로 세지 않았다.

검증 과정에서 24개 평가 맵의 seed·shot 수·데이터 해시, checkpoint 및 예측 파일 해시를
대조했다. 저장된 두 observable의 예측과 정답에서 개선·악화 건수, 평균, 구간을 다시
계산해 각 거리의 결과와 일치함을 확인했다. 디코딩 실패로 인한 fallback은 0건이었다.
[검증 집계 JSON](../../artifacts/eval/predecoder_revalidation_20261115/independent_revalidation_summary.json)

후속 잡음 변화 진단에서는 별도로 예약한 d7 맵과 수정 입력으로 학습한 seed 2/11/54/57을
사용했다. 과거 margin과 채택 결과를 새 checkpoint에 이식하지 않았다.
[d7 잡음 변화 결과 보고서](results/PREDECODER_D7_NOISE_SHIFT_RESULTS.md)
