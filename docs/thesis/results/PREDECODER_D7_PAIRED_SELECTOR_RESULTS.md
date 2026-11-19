# d7 Z 전이와 원본 보존 후보를 비교하는 selector head

2026-11-16 · Anaconda `fp` 환경에서 실행

두 개발 맵을 번갈아 검증했을 때 detector 패턴을 사용한 selector head는, 기존 1위가
수정한 2,998쌍 중 2,813쌍에서 원본과 같은 논리 예측을 내는 다른 수정 후보를 골랐다.
원본 오답 100건을 고치고 원본 정답 85건을 해쳐, 원본 PyMatching 대비 순이득은
+0.092 pp였다. [학습 계수와 집계 JSON](../../../artifacts/eval/d7_paired_selector/20261116/summary.json)

## 비교 조건

강한 ZZ 조건(G₁, `zz_factor=0.25`)과 기준 DEM(D₀)의 일반 MWPM을 사용했다. 개발 맵
783000, 783001에서 각각 2,048 shot을 네 checkpoint(2, 11, 54, 57)가 공유한다. 전체
집계의 분모는 16,384 모델–shot 쌍이다. 기존 모델과 저장한 후보 집합을 고정했다.
[detector 패턴 선행 진단](PREDECODER_D7_Z_TRANSITION_PROBE.md)

기존 1위가 비영 수정으로 Z 전이를 일으킨 shot에서, 비영 후보 중 원본과 같은 논리
예측을 내고 기존 utility가 가장 높은 후보를 짝으로 골랐다. 두 개발 맵에서 해당하는
2,998쌍 모두에 이 후보가 있었다. 정답을 기준으로 두 후보 중 하나만 맞은 사례를 사용해
어느 후보가 맞는지 L2 로지스틱 회귀(규제 계수 0.01)로 학습했다. 각 맵에서 학습한 head는
반대편 맵에 적용했다.

대조 head에는 Z 후보와 보존 후보 사이의 utility 차이, Z 후보와 no-edit 사이의 utility
차이, 두 수정의 무게와 3×3×3 패치의 유효 위치·수정 mask, 원본 논리 클래스,
checkpoint를 넣었다. detector 패턴을 사용하는 head에는 각 후보의 패치에 나타난
detector event와 수정 대상으로 지정된 detector 중 실제로 활성화된 수를 추가했다.

주 비교는 head의 log-odds가 0 이상일 때 Z 전이를 선택하고, 그 외에는 같은 shot에서
비영 보존 후보를 선택한다. 수정할 shot은 기존 1위와 같다. 보조 경로에서는 비영 보존
후보 대신 no-edit를 선택하고, 짝을 만들지 않은 shot에는 기존 margin을 적용한다.
학습 맵에서 기존 margin의 개선 건수에 맞춘 cutoff도 두 경로에 적용했다.

## 결과

| 경로 | 검증 맵 783000 개선 / 악화 | 검증 맵 783001 개선 / 악화 | 합산 원본 대비 |
| --- | ---: | ---: | ---: |
| 기존 1위 | 433 / 1,066 | 450 / 1,047 | −7.507 pp |
| 기존 margin | 42 / 63 | 43 / 46 | −0.146 pp |
| 대조 head, log-odds 0 | 19 / 23 | 21 / 19 | −0.012 pp |
| detector 패턴 head, log-odds 0 | 48 / 48 | 52 / 37 | +0.092 pp |
| detector 패턴 head, 학습 맵 margin 기준 | 31 / 29 | 34 / 26 | +0.061 pp |

log-odds 0 기준에서 detector 패턴 head는 기존 1위보다 +7.599 pp, 기존 margin보다
+0.238 pp 높았다. 검증 맵별로 원본 대비 0.000 pp와 +0.183 pp였다. 수정 shot을
고정한 경로는 2,998건을 수정했고, no-edit를 허용한 경로는 185건을 수정했다. 이 두
경로의 논리 예측은 전 shot에서 같았다. head가 Z 전이를 선택하지 않은 shot에서 두
경로 모두 원본과 같은 논리 예측을 내기 때문이다.

이 실험은 detector 패턴을 사용해 기존 수정량을 유지하면서 다른 후보를 선택할 수
있음을 확인했다. 원본 대비 합산 순증은 15건이며, 새 맵에서 순이득이 재현되는지는
별도로 확인해야 한다. 다음 독립 평가는 head 입력과 log-odds 0 규칙을 고정한 뒤,
두 개발 맵에서 다시 학습해 새 맵에 한 번 적용한다.

## 재현

```powershell
conda activate fp
python tools/run_d7_paired_selector.py --source artifacts/eval/noise_shift_d7/20261115_retrained --probe-summary artifacts/eval/d7_z_transition_probe/20261116/summary.json --out artifacts/eval/d7_paired_selector/20261116/summary.json
python -m unittest discover -s tests
```

입력 샘플 2개와 후보 예측 파일 8개의 해시를 앞선 진단의 기록과 대조했다. 집계와 각
fold의 head 계수를 JSON에 저장했다. 새 출력 경로로 실행하면 결과를 다시 계산한다.
