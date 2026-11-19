# d7 Pauli 편향 독립 평가 규약

개발 파일럿을 마친 뒤 대표 편향과 정책을 고정하고, 새로운 local map에서
PyMatching 대비 pre-decoder의 추가 효과를 확인한다.
실행 설정과 소스·checkpoint 해시는 평가 자료 생성 전에
[protocol.json](../../artifacts/eval/d7_pauli_bias/independent/protocol.json)에 저장했다.

## 고정 조건

| 항목 | 값 |
| --- | --- |
| 회로 | d=7, 7 rounds, rotated surface code, Bell-pair readout |
| 잡음 | Stage C, p=0.0015, ZZ 계수 0.05, swap 계수 0.02 |
| 변경 대상 | 데이터 큐비트의 기존 단일 큐비트 gate·idle 채널 |
| 편향 | eta = p_Z/(p_X+p_Y) = 20, p_X = p_Y |
| 기준선 | 해당 맵의 실제 편향 회로에서 만든 DEM의 일반 MWPM |
| pre-decoder | 기존 A/B 재학습 seed 2의 후보 생성기와 margin selector |
| 수정 기준 | 최선의 비영 후보가 선택되고, 원본 유지 대비 utility 차이가 1.25 이상 |
| context | 파일럿과 같은 고정값, 라벨 집계 채널 0 |
| local map seed | 788600–788623 |
| sampler seed | 1886000–1886047 |
| 평가 표본 | 24개 맵 × 2,048 shots = 49,152 shots |
| 실행 | CPU 작업 2개, 작업별 PyTorch thread 2개 |

대표 편향은 파일럿에서 사용한 가장 강한 조건인 eta=20으로 정했다.
일반 MWPM은 파일럿의 편향 DEM 기준 오류율로 선택했다.
모델 seed와 margin은 파일럿에서 사용한 값을 유지한다.
기존 checkpoint의 전역 배포 결정이 raw 반환인 경우에도 후보·selector 경로를 직접 실행한다.
새 평가 결과를 이용한 threshold 조정이나 checkpoint 선택은 포함하지 않는다.

각 맵에서 균등 단일 큐비트 채널 eta=0.5의 회로도 만든다.
이 회로의 DEM은 편향을 반영하지 않은 가중치 대조군에 사용하고,
메타데이터는 파일럿과 같은 context를 구성하는 데 사용한다.
이 회로의 1-shot 샘플은 평가 집계에 포함하지 않는다.
짝수 sampler seed는 이 회로에, 다음 홀수 seed는 eta=20 평가 샘플에 사용한다.
따라서 세 디코더 경로는 같은 eta=20의 2,048 shots를 공유한다.

각 변경 위치의 총 오류 확률을 유지한다.
Stim 회로 텍스트의 반올림과 DEM의 approximate_disjoint_errors 근사는
[파일럿 규약](PREDECODER_D7_PAULI_BIAS_PROTOCOL.md)의 조건을 따른다.

## 주 비교와 보조 비교

주 비교는 편향 DEM의 일반 MWPM에 pre-decoder를 추가한 정확도 차이다.
두 logical observable 중 하나라도 틀리면 joint logical error로 센다.
같은 shot에서 원본 오답을 고친 수와 원본 정답을 해친 수를 구한다.
개선량은 맵마다 100 × (고친 수 − 해친 수) / 2,048로 계산한다.

24개 맵의 개선량을 같은 가중치로 평균한다.
주 비교의 구간은 맵별 차이에 대한 양측 95% Student t 구간이다.
구간 하한이 0을 초과하면 고정 모델의 새 맵에서 추가 개선을 지지한다.
구간이 0을 포함하면 추가 개선은 불확실한 것으로 보고한다.
구간 상한이 0보다 작으면 악화를 지지한다.

같은 shot에서 편향 DEM 대 균등 채널 DEM,
pre-decoder 대 균등 채널 DEM의 두 보조 비교도 집계한다.
이 구간은 탐색적 수치로 보고하며, 주 비교의 채택 결정을 바꾸는 데 사용하지 않는다.

파일럿의 eta=20 맵별 표준편차는 0.2282 pp였다.
같은 표본 크기로 24개 맵을 평가할 때 예상 95% 구간 반폭은 약 0.0964 pp다.
실제 구간은 새 맵에서 관측한 분산으로 계산한다.
표본 수는 평가 전에 고정하며, 중간 효과에 따라 추가하거나 줄이지 않는다.

## 검증과 해석

파일럿의 후보 생성·점수·채택 함수를 재사용한다.
후보의 논리 전이 특징은 현재 맵의 편향 DEM으로 다시 계산한다.
선택 함수에는 dummy label을 전달하고, 실제 정답은 지표 집계에 사용한다.
선택한 후보를 다시 복호화해 전이 예측과 일치하는지 확인한다.

저장된 예측 배열에서 고친 수, 해친 수, 오류율과 맵별 구간을 재계산한다.
모든 맵이 완료된 뒤 주 비교를 확정한다.
checkpoint와 소스 파일의 해시, 샘플 및 예측 파일의 해시를 보관한다.

구간은 고정 checkpoint가 새 잡음 맵에서 보인 변동을 반영한다.
새로운 학습 반복에 대한 일반화와 전체 실행 지연은 별도 평가가 필요하다.
두 CPU 작업을 동시에 실행하므로 이번 경과 시간을 독립적인 지연 측정값으로 사용하지 않는다.

실행 명령은 Anaconda fp 환경에서 사용한다.

```powershell
python tools/evaluate_d7_pauli_bias.py freeze
python tools/evaluate_d7_pauli_bias.py run
```
