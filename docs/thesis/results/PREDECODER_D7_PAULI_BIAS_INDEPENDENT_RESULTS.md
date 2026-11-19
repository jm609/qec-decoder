# d7 Pauli 편향 독립 평가 결과

독립 평가 규약은
[고정 규약](../PREDECODER_D7_PAULI_BIAS_INDEPENDENT_PROTOCOL.md)과
[실행 설정](../../../artifacts/eval/d7_pauli_bias/independent/protocol.json)에 있다.
파일럿의 788500–788507 맵을 개발 자료로 사용한 뒤,
새로운 788600–788623 맵에서 고정 정책을 평가했다.

## 고정한 조건과 지표

d=7, 7 rounds, Bell-pair readout, Stage C p=0.0015,
ZZ 계수 0.05, swap 계수 0.02를 사용했다.
데이터 큐비트의 단일 큐비트 gate·idle 채널에서 총 오류 확률을 유지하며
eta = p_Z/(p_X+p_Y) = 20, p_X = p_Y로 정했다.
2-qubit depolarizing noise, 측정·리셋 오류, ancilla 잡음과 ZZ·swap 채널은
각 맵의 기존 확률을 사용했다.

대표 편향은 파일럿에서 조사한 가장 강한 조건인 eta=20으로 정했다.
기준선은 파일럿에서 선택한 편향 DEM의 일반 MWPM이다.
기존 A/B 재학습 seed 2 checkpoint의 후보·margin 경로를 직접 실행했고,
selector utility 차이에 적용하는 margin 1.25를 고정했다.
원본 유지 대비 최선 비영 후보의 utility 차이가 1.25 이상일 때 수정을 유지한다.
checkpoint의 전역 raw 반환 결정을 사용하는 경로와 평가 조건이 다르다.

평가 표본은 24개 맵 × 2,048 shots, 총 49,152 shots다.
균등 채널 DEM, 편향 DEM, pre-decoder의 세 경로는 같은 eta=20 shot을 공유한다.
각 맵의 균등 채널 회로에서 생성한 1-shot 샘플은 집계에 포함하지 않았다.
두 logical observable 중 하나라도 틀리면 joint logical error로 센다.

개선량은 같은 shot의 고친 수에서 해친 수를 뺀 정확도 차이이며 pp로 보고한다.
주 비교는 편향 DEM의 일반 MWPM 대비 pre-decoder의 추가 효과다.
24개 맵의 차이를 동일 가중치로 평균하고 양측 95% Student t 구간을 계산한다.
주 비교의 구간 하한이 0을 초과하면 새 맵의 추가 개선을 지지한다.
보조 비교의 구간은 탐색적 수치다.

## 독립 평가 결과

| 비교 | 기준 오류율 (%) | 적용 후 오류율 (%) | 고친 수 | 해친 수 | 개선량 (pp) | 95% 구간 (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 편향 MWPM에 pre-decoder 추가, 주 비교 | 14.1337 | 14.1296 | 205 | 203 | +0.0041 | [−0.0859, +0.0940] |
| 균등 채널 DEM에서 편향 DEM으로 변경, 보조 비교 | 14.1337 | 14.1337 | 45 | 45 | 0.0000 | [−0.0360, +0.0360] |
| 균등 채널 DEM 대비 편향 DEM과 pre-decoder 결합, 보조 비교 | 14.1337 | 14.1296 | 241 | 239 | +0.0041 | [−0.0791, +0.0872] |

주 비교에서는 원본 오답 205개를 고치고 원본 정답 203개를 해쳤다.
논리 예측을 바꾼 shot은 408개이며, 최종 정답 수는 2개 늘었다.
구간 하한이 0을 초과하지 않아 사전에 정한 추가 개선 기준을 충족하지 못했다.
편향 DEM을 반영하는 보조 비교에서도 정확도 이득은 확인되지 않았다.

파일럿 eta=20의 +0.1892 pp [−0.0016, +0.3800]에 비해
독립 평가의 점추정은 +0.0041 pp였다.
이번 독립 평가에서 파일럿의 양의 점추정에 해당하는 개선을 확인하지 못했다.
이 결과로 실제 효과가 정확히 0이라고 확정할 수는 없으며,
고정한 모델·정책의 추가 효과가 불확실한 것으로 해석한다.

집계와 맵별 결과는
[summary.json](../../../artifacts/eval/d7_pauli_bias/independent/summary.json)에 있다.

## 검증과 해석 범위

24개 맵의 저장 배열에서 세 비교의 고친 수, 해친 수, 오류율,
맵별 개선량과 95% 구간을 다시 계산해 summary.json과 일치하는지 확인했다.
소스·checkpoint·데이터·예측 파일의 해시를 검증했다.
모든 맵의 selector context는 파일럿과 같았고 라벨 집계 채널은 0이었다.
후보 복호화 fallback은 0이었다. 관련 Python 파일의 py_compile 검사와
전체 38개 테스트도 통과했다.

원본 오답 6,947개 중 oracle이 복구할 수 있는 shot은 6,081개,
정답 후보 포함률은 87.53%였다.
이는 후보 집합의 회복 여지를 나타내며, 실제 채택의 추가 이득은 위 주 비교로 판단한다.

균등 채널 회로와 편향 회로의 noise-free circuit, detector 순서,
변경 대상 외 잡음 명령의 일치를 확인했다.
저장된 회로 텍스트에서 총 오류 확률 합의 최대 절대 차이는 9.8×10⁻⁹,
최대 상대 차이는 8.29×10⁻⁶이었다.
파일럿에서 정한 상대 허용오차 10⁻⁵로 통과했다.

재집계 및 수치 검증 기록은
[validation.json](../../../artifacts/eval/d7_pauli_bias/independent/validation.json)에 있다.
고정한 소스는
[source_snapshot.zip](../../../artifacts/eval/d7_pauli_bias/independent/source_snapshot.zip)에 보관했다.

이번 평가는 고정 checkpoint 하나의 새 잡음 맵에 대한 결과다.
편향 데이터로 새 모델을 학습하거나 평가 결과로 threshold를 보정한 과정은 없다.
학습 반복 간 변동과 전체 실행 지연은 별도 검증이 필요하다.
두 CPU 작업을 동시에 실행했고 후보 재복호화 확인 비용도 포함되므로,
이번 평가 경과 시간을 raw 대비 지연 측정값으로 해석하지 않는다.

DEMs에는 approximate_disjoint_errors 근사가 포함된다.
기존 Stim 회로 텍스트의 반올림은
[파일럿의 수치 검증](PREDECODER_D7_PAULI_BIAS_PILOT_RESULTS.md)의 조건을 따른다.
eta=20은 변경 대상인 데이터 큐비트 단일 큐비트 채널의 편향을 나타낸다.

다음 모델 개선 단계에서는 기존 후보 생성기를 유지한 상태에서
편향 잡음에 맞는 별도 학습·개발 자료로 채택 모델을 재학습하는 방안을 검증한다.
이번 독립 결과를 본 뒤 정책을 바꾸면, 바뀐 정책은 다시 새로운 평가 맵에서 확인해야 한다.
RTL 작업에 앞서 PyMatching 대비 추가 정확도 이득을 먼저 확인한다.

실행 명령은 Anaconda fp 환경에서 사용한다.

```powershell
python tools/evaluate_d7_pauli_bias.py freeze
python tools/evaluate_d7_pauli_bias.py run
```
