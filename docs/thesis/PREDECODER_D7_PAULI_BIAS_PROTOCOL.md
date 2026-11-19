# d7 데이터 큐비트 Pauli 편향 파일럿 규약

2026-11-18에 시작한 개발 실험이다. 실행 전에 저장한 설정, checkpoint,
소스 파일 해시는
[pilot_protocol.json](../../artifacts/eval/d7_pauli_bias/pilot/pilot_protocol.json)에 있다.
아래 문서는 실행 설정을 설명하며, 파일럿 완료 후 독립 평가 규약은 별도로 고정한다.

## 잡음 조건

기존 rotated surface code의 d=7, 7 rounds, Z-basis Bell-pair readout을 사용한다.
Stage C의 기준 오류 확률은 p=0.0015, ZZ 계수는 0.05,
swap 계수는 0.02이다. 기존 local map을 각 편향 조건에서 공유한다.

데이터 큐비트에 적용되는 각 DEPOLARIZE1(q)를 아래 확률의
PAULI_CHANNEL_1로 바꾼다.

\[
\eta = \frac{p_Z}{p_X+p_Y},\qquad
p_X=p_Y=\frac{q}{2(1+\eta)},\qquad
p_Z=\frac{q\eta}{1+\eta}.
\]

각 위치의 총 오류 확률 q를 유지한다. 1-qubit gate 뒤와 idle 위치가 적용 대상이다.
\(\eta=0.5,5,20\)을 비교하며, \(\eta=0.5\)에서는 기존 회로를 그대로 사용한다.
따라서 기준 조건의 circuit, DEM, 같은 sampler seed의 전체 배열이 기존 경로와 일치한다.
메모리에서 계산한 확률 합은 상대 허용오차 \(10^{-12}\)로 검증한다.
Bell readout 구성과 파일 저장에는 Stim의 유효숫자 6자리 회로 텍스트를 사용한다.
전체 저장 회로에서 확률 합의 최대 절대 차이는 \(1.0\times10^{-8}\),
최대 상대 차이는 \(8.12\times10^{-6}\)이었다.
저장 회로 검증에는 상대 허용오차 \(10^{-5}\)를 적용한다.

회로의 2-qubit depolarizing noise, 측정·리셋 오류, ancilla의 단일 큐비트 잡음,
기존 ZZ와 swap 채널은 세 조건에서 같은 명령과 확률을 사용한다.
\(\eta\)는 변경 대상 채널의 편향을 나타낸다.
회로 전체에서 발생한 X/Z 오류의 확률 비나 논리 오류 비율은 별도로 관측한다.
DEM을 만들 때 approximate_disjoint_errors를 사용하며,
서로 배타적인 Pauli 오류를 독립 오류로 근사하는 점을 해석에 포함한다.

## 표본과 비교

| 항목 | 고정값 |
| --- | --- |
| 개발 local map seed | 788500–788507 |
| sampler seed | 1885000–1885023 |
| 편향 조건별 표본 | 8개 맵 × 8,192 shots |
| pre-decoder 비교 표본 | 각 맵의 첫 2,048 shots |
| 기존 checkpoint | d7 재학습 seed 2 |
| selector 입력 | 원본 syndrome, 기하 정보, 라벨을 제거한 기존 context |
| detector event fraction context | 기존 고정값 0.042736961728050596 |

편향 조건마다 별도 sampler seed를 사용한다.
같은 편향 조건의 디코더 비교에는 동일한 detector events와 정답을 사용한다.

1. 각 편향 회로의 shot을 \(\eta=0.5\) 회로에서 만든 DEM으로 복호화한다.
2. 같은 shot을 실제 편향 회로에서 만든 DEM으로 복호화한다.
3. 일반 MWPM과 correlated matching을 모두 실행한다.
4. 편향을 반영한 두 모드 중 개발 맵의 joint logical error가 낮은 모드를 고른다.
   동률이면 일반 MWPM을 사용한다.
5. 선택한 DEM과 매칭 모드에 기존 pre-decoder 후보 생성기와 margin selector를 결합한다.

pre-decoder는 A/B에서 재학습한 기존 seed 2 checkpoint를 사용한다.
저장된 margin은 1.25이며, 후보와 원본 유지의 selector utility 차이에 적용한다.
최선의 비영 후보를 선택한 경우 이 차이가 1.25 이상이어야 수정을 유지한다.
이 checkpoint의 기존 배포 결정이 raw 반환인 경우에도,
파일럿에서는 저장된 후보 생성기와 margin selector를 직접 실행한다.
따라서 후보 경로의 조건부 성능을 평가하며, 기존 배포 정책의 성능과 구별해 기록한다.
학습, threshold 보정, 여러 checkpoint 중 선택은 이번 파일럿에 포함하지 않는다.

후보의 논리 전이 특징을 현재 편향 DEM과 현재 매칭 모드로 다시 계산한다.
후보를 고르는 함수에는 dummy label을 전달하며, 실제 정답은 평가에 사용한다.
원본 유지 후보를 포함해 최대 32개 비영 후보를 평가한다.
선택한 후보는 같은 디코더로 다시 복호화해 저장된 전이 예측과 일치하는지 확인한다.

## 지표와 판단

두 logical observable 중 하나라도 틀리면 joint logical error로 센다.
logical X와 logical Z 오류 수도 함께 기록한다.
pre-decoder의 개선량은 같은 shot의
\(100\times(\text{고친 수}-\text{해친 수})/N\)으로 계산한다.

차이는 맵별로 계산한 뒤 동일 가중치로 평균한다.
구간은 8개 맵의 paired 차이에 대한 양측 95% Student t 구간이다.
여러 편향과 매칭 모드를 조사하므로 파일럿의 구간은 탐색적 수치로 보고한다.
8개 맵에서 baseline 모드도 선택하므로, 이 자료의 개선량으로 최종 채택을 결정하지 않는다.

oracle은 정답을 아는 경우 후보 집합이 원본 오답을 복구할 수 있는 비율이다.
후보의 회복 여지를 진단하는 지표로 사용한다.

저장되는 selector_seconds는 후보 추론 및 확인 복호화 구간의 경과 시간이다.
데이터 준비, 첫 raw 복호화, 모델 적재를 포함한 전체 지연은 별도 시간 실험에서 측정한다.

## 완료 조건과 다음 단계

24개 조건의 baseline 및 pre-decoder 예측을 저장하고, 저장된 배열에서 집계를 재계산한다.
총 오류 확률 보존, 기본 조건의 기존 경로 재현,
label-independent action과 후보 재복호화 일치를 검증한다.

파일럿 결과에 따라 대표 편향과 매칭 모드, pre-decoder 정책,
checkpoint와 효과 크기를 고정한 뒤 새로운 local map에서 독립 평가한다.
RTL 작업은 독립 평가에서 PyMatching 대비 추가 이득이 재현된 뒤 진행한다.
