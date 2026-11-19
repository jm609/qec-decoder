# d7 독립 최종 평가 규약

2026-11-17 · 논문 완성 계획 4단계 · 새 평가 맵을 생성하기 전에 동결한다.

## 비교와 데이터

주 평가 조건은 강한 ZZ G₁(`zz_factor=0.25`), 기준 DEM D₀, 일반 MWPM이다. 기존 checkpoint의 **shot별 margin**을 선택 정책으로 고정한다. 주 비교는 (1) margin 대 같은 셀의 원본 PyMatching, (2) margin 대 **paired detector head**다. 2단계에서 선택 방법과 가장 강한 단순 대조 방법이 같은 margin이었으므로, 두 번째 비교에는 별개의 paired 방법을 둔다. Paired head는 개발 G₁ 맵 4개로 적합한 2단계 refit 계수·정규화와 log-odds 0 기준을 사용한다. 두 방법 모두 과거 checkpoint의 family gate를 사용하지 않는다. [2단계 결과](results/PREDECODER_D7_CONFIDENCE_DEVELOPMENT_RESULTS.md) · [3단계 결과](results/PREDECODER_D7_ACTIVE_PRIOR_RESULTS.md)

평가 대상은 d7, 7 rounds, rotated surface code의 Bell-pair Z readout 회로다. `p=0.0015`, `zz_factor=0.05 / 0.25`, d7 재학습 checkpoint seed 0–7을 고정한다. 후보 생성과 점수 경로는 3단계의 전체 경로를 재사용한다. 각 셀의 DEM과 일반/correlated PyMatching으로 원본·후보 논리 예측과 선택을 계산한다. 입력 context는 각 맵의 G₀ 설정과 기존 고정 detector-event fraction을 사용한다. `reserved_label_context`는 0이다. 선택 경로에서 정답을 0으로 대체하고, 저장한 정답으로 성능과 후보 oracle을 계산한다.

| 항목 | 사전 지정 값 |
| --- | --- |
| 로컬 잡음 맵 | 788100–788123, 총 24개 |
| Sampler seed | 1794100–1794147, 맵마다 G₀ 다음 G₁ |
| G₁ 표본 | 맵당 512 shot, 합계 12,288 shot |
| G₀ 표본 | 맵당 256 shot, 합계 6,144 shot |
| 디코더 셀 | D₀/D₁ × 일반/correlated PyMatching, 각 강도마다 4개 |
| 실행 환경 | Anaconda `fp`, CPU worker 2개, PyTorch 스레드 2개/worker, MKL SEQUENTIAL |

한 잡음 맵의 두 강도는 같은 로컬 구성과 이상 회로를 공유하고 별도 sampler를 사용한다. D₀는 그 맵의 G₀ 생성 회로, D₁은 G₁ 회로에서 얻는다. G₀의 모든 셀과 G₁의 나머지 세 셀은 보조 비교다. 새 맵과 sampler seed를 기존 case 및 학습 기록과 대조하고, seed·checkpoint·head·코드와 라이브러리 버전은 기계 판독 규약에 기록한다.

## 표본 규모와 판정

두 주 비교의 계획 구간 반폭을 0.10 pp로 두었다. 3단계의 G₁ 개발·보정 맵 6개에서 모델별 같은 shot의 정오답 차이를 먼저 평균했다. 관측 맵 평균의 분산과 고정 모델 패널 내 shot 변동을 함께 사용해 범위를 보수적으로 산정했다. 24개 맵 × 512 shot에서 양측 97.5% 예상 반폭은 margin 대 원본 **0.059 pp**, margin 대 paired **0.109 pp**다. 두 번째 비교는 0.10 pp 계획 반폭에 못 미친다. 기존 실행 시간으로 환산하면 평가 약 53분이며, 실제 구간은 새 맵의 변동에 따라 달라진다. G₀는 보조 비교이므로 맵 수를 유지하고 맵당 shot을 줄였다.

각 맵 안에서 여덟 모델을 동일 가중하고, 맵 평균 24개에 같은 가중치를 준다. 주 비교별로 **양측 97.5% Student t 구간**을 계산한다. 두 비교의 전체 오차율을 Bonferroni 방식으로 관리한다. 하한이 0보다 크면 개선, 상한이 0보다 작으면 악화, 그 외에는 불확실로 보고한다. 보조 셀의 95% 구간은 기술 통계로 사용한다. 동일 shot을 여러 checkpoint가 공유하므로 모델–shot 쌍 수를 독립 표본 수로 세지 않는다. 결론은 고정 checkpoint 패널에 조건부다.

각 셀에서 두 observable을 모두 맞힌 shot의 정확도, 원본과 비교군 대비 개선·악화 수, 정확도 차이(pp), logical error rate와 상대 오류 감소율을 집계한다. 후보 oracle은 no-edit를 포함한 고정 후보 집합에서 정답을 사용해 구한 상한이다. 맵·모델별 결과와 저장 예측을 함께 남긴다.

평가 실행 상한은 80분으로 둔다. 입력 해시 불일치, 예상 밖의 매칭 실패 또는 실행 중단이 있으면 해당 기록을 남긴다. 복구 시 같은 규약·맵·sampler seed로 재개하고, 실패 맵을 바꾸거나 관측 결과에 따라 표본을 늘리지 않는다. 주 비교는 전체 예약 맵이 검증된 뒤 계산한다.
