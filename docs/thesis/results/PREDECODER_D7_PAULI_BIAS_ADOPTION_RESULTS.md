# d7 편향 잡음 채택 모델 재학습 결과

기존 후보 생성기를 고정하고, eta=20 자료로 새 채택 모델을 학습했다.
[실험 규약](../PREDECODER_D7_PAULI_BIAS_ADOPTION_PROTOCOL.md)과
[소스·자료 registry](../../../artifacts/eval/d7_pauli_bias_adoption/registry.json)에 실행 조건을 기록했다.
이전 독립 평가 788600–788623 맵은 이번 학습·개발·평가에 포함하지 않았다.

## 학습과 기준 선택

d=7, 7 rounds, Stage C p=0.0015, ZZ 계수 0.05, swap 계수 0.02에서
데이터 큐비트 단일 큐비트 gate·idle 채널의 eta를 20으로 고정했다.
기존 seed 2의 후보 CNN과 motif를 사용해 원본 유지 1개와 비영 후보 32개를 생성했다.
이번에 적합한 모델은 L2=0.01의 4-class logistic adoption head다.
원본 MWPM 클래스 대비 실제 정답의 논리 전환을 학습했다.

특징은 원본 matching 비용·논리 클래스, 라운드별 event 수,
클래스별 상대 matching 비용과 도달 가능 여부,
각 후보 논리 전환의 개수·첫 순위·수정 matching 비용 요약이다.
실제 정답은 특징 계산 함수에 전달하지 않았다.
candidate CNN의 가중치는 이번 학습에서 고정했다.

| 자료 | 맵 | Shots |
| --- | --- | ---: |
| 학습 | 788700–788731, 32개 | 65,536 |
| 기준 선택 | 788740–788747, 8개 | 16,384 |
| 독립 평가 | 788800–788823, 24개 | 49,152 |

학습 맵에서 모델을 적합하고, 개발 맵에서 확률 차이 기준을 선택했다.
대안 논리 전환 확률에서 원본 유지 확률을 뺀 값이 기준을 초과하면 전환한다.
개발 자료에서 평균 순이득을 최대화한 기준은 0.01이었다.
이 기준에서 80개를 복구하고 63개를 악화시켜 순증가는 17개, +0.1038 pp였다.
개발 기준 후보를 비교해 고른 수치이므로 이 개선량은 개발 성과로 해석한다.

기준 선택 후 학습 모델을 다시 적합하지 않았다.
모델 계수와 기준값은
[policy.json](../../../artifacts/eval/d7_pauli_bias_adoption/policy.json)에 있다.
독립 자료 생성 전에
[evaluation_protocol.json](../../../artifacts/eval/d7_pauli_bias_adoption/evaluation_protocol.json)에
policy와 registry의 해시를 저장했다.
이 기준은 기존 margin 경로의 utility 기준 1.25와 서로 다른 점수에 적용된다.

## 독립 평가

| 경로 | 논리 오류율 (%) | 복구 | 악화 | 원본 대비 정확도 차이 (pp) | 95% 맵 구간 (pp) |
| --- | ---: | ---: | ---: | ---: | --- |
| 편향 DEM의 일반 MWPM | 14.4124 | 기준 | 기준 | 0 | 기준 |
| 기존 margin, utility 기준 1.25 | 14.3778 | 216 | 199 | +0.0346 | [−0.0392, +0.1084] |
| 재학습 채택 모델, 확률 차이 기준 0.01 | 14.3941 | 199 | 190 | +0.0183 | [−0.0702, +0.1068] |

재학습 모델은 새 맵 24개, 49,152 shots에서 199개를 고치고 190개를 해쳤다.
정답 수 순증가는 9개였다.
주 비교의 구간 하한이 0을 초과하지 않아 추가 개선 기준을 충족하지 못했다.
기존 margin 대비 차이는 −0.0163 pp, 95% 구간은 [−0.1092, +0.0767] pp였다.
기존 margin 대비 개선 여부도 불확실하다.

개발 자료의 +0.1038 pp에 비해 독립 평가의 점추정은 +0.0183 pp였다.
이번 재학습은 실제 모델 계수를 새로 적합했으나,
독립 평가에서 PyMatching 대비 추가 이득을 확인하지 못했다.
비용·후보 요약 특징을 사용하는 이번 4-class 선형 채택 모델의 결과로 해석한다.

새 정책의 논리 전환은 389개였고, 기존 margin의 전환은 415개였다.
두 정책 모두 Z 전환이었다.
새 정책은 기존 margin의 악화 191개를 막았지만 복구 202개를 놓쳤다.
추가로 185개를 복구하고 182개를 해쳐,
기존 margin 대비 정답 수는 총 8개 감소했다.
따라서 선택한 shot 집합을 바꾼 효과가 순이득으로 이어졌다는 근거는 얻지 못했다.

원본 오답 7,084개 중 후보 집합에 정답이 포함된 shot은 6,195개, 87.45%였다.
이 값은 정답을 아는 oracle의 후보 회복 여지이며,
실제 선택 정책의 성과는 위 독립 비교로 판단한다.

맵별 결과와 집계는
[summary.json](../../../artifacts/eval/d7_pauli_bias_adoption/summary.json)에 있다.

## 검증과 해석 범위

학습 32개·개발 8개·평가 24개 맵의 dataset, 특징, 예측 파일 해시를 확인했다.
학습 자료에서 정규화의 평균과 표준편차를 다시 계산해 모델 파일과 일치함을 확인했다.
개발 자료의 기준값별 복구·악화와 선택 기준 0.01도 재현했다.
학습 target 분포는 원본 유지 56,328개, X 전환 10개, Z 전환 9,196개, Y 전환 2개였다.

독립 평가에서는 선택한 후보 mask를 실제 syndrome에 적용하고,
같은 편향 MWPM으로 재복호화해 저장된 예측과 전수 대조했다.
정답 라벨을 바꿔도 새 정책의 예측이 일치했다.
저장 배열에서 세 비교의 복구·악화·오류율과 맵별 95% 구간을 재계산해
실행 집계와 일치하는지 확인했다.
candidate CNN과 소스 해시, 독립 평가 전 저장한 policy 해시도 일치했다.
관련 Python 파일의 py_compile 검사와 전체 40개 테스트를 통과했다.

[학습·개발 검증](../../../artifacts/eval/d7_pauli_bias_adoption/training_validation.json)과
[전체 독립 검증](../../../artifacts/eval/d7_pauli_bias_adoption/validation.json)에 검증 수치를 기록했다.
[source_snapshot.zip](../../../artifacts/eval/d7_pauli_bias_adoption/source_snapshot.zip)에 고정한 소스를 보관했다.

결과는 고정 후보 CNN 하나와 이번 학습 자료로 적합한 채택 모델 하나에 조건부다.
독립 비교는 같은 shot의 joint logical error를 사용하며,
두 logical observable 중 하나라도 틀리면 오답으로 센다.
구간은 맵별 paired 차이의 양측 95% Student t 구간이다.
새 정책 대 편향 MWPM이 주 비교이며, 기존 margin 대비 차이는 보조 비교다.

두 CPU 작업으로 자료와 예측을 생성했다.
이번 경과 시간은 전체 배치 지연의 비교 측정값으로 사용하지 않는다.
DEMs에는 approximate_disjoint_errors 근사가 포함되고,
Stim 텍스트의 확률 반올림은 기존 편향 실험의 수치 조건을 따른다.

Anaconda fp 환경에서 실행한다.

```powershell
python tools/train_d7_pauli_bias_adoption.py develop
python tools/train_d7_pauli_bias_adoption.py evaluate
```
