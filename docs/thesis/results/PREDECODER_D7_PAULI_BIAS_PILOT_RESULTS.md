# d7 데이터 큐비트 Pauli 편향 파일럿 결과

2026-11-18에 시작한 개발 파일럿을 중단 지점부터 재개했다.
회로와 데이터 생성 조건은
[파일럿 규약](../PREDECODER_D7_PAULI_BIAS_PROTOCOL.md)에 기록했다.

## 평가 조건

d=7, 7 rounds, rotated surface code의 Bell-pair readout을 사용했다.
Stage C의 p=0.0015, ZZ 계수 0.05, swap 계수 0.02에서
데이터 큐비트의 단일 큐비트 채널에
\(\eta=p_Z/(p_X+p_Y)=0.5,5,20\)을 적용했다.
각 변경 위치의 총 오류 확률을 유지하며 X와 Y의 확률을 같게 두었다.
2-qubit depolarizing noise, 측정·리셋 오류, ancilla 잡음과 ZZ·swap 채널은
같은 local map에서 같은 확률을 사용했다.

개발 맵은 788500–788507이며, 편향별로 8,192 shots씩 생성했다.
기준선 비교에는 편향별 65,536 shots를 사용했다.
pre-decoder 비교에는 각 맵의 첫 2,048 shots, 편향별 16,384 shots를 사용했다.
각 디코더 비교는 같은 shot을 공유한다.
두 logical observable 중 하나라도 틀리면 joint logical error로 센다.

모든 구간은 맵별 paired 차이에 대한 양측 95% Student t 구간이다.
8개 맵에서 baseline 모드를 선택했고 여러 조건을 조사했으므로,
아래 구간은 개발 자료에 대한 탐색적 수치다.

## 편향을 반영한 PyMatching

같은 편향 shot을 균등 단일 큐비트 채널의 DEM과 실제 편향 회로의 DEM으로 복호화했다.
균등 채널 DEM에도 기존 local map, 측정·리셋 잡음, ZZ·swap 채널이 포함된다.
표의 개선량은 균등 채널 DEM 대비 편향 DEM의 joint logical error 감소량이다.
양수이면 편향 DEM의 정확도가 더 높다.

| 편향 eta | 모드 | 균등 채널 DEM의 오류율 (%) | 편향 DEM의 오류율 (%) | 개선량 (pp) | 95% 구간 (pp) |
| --- | --- | ---: | ---: | ---: | --- |
| 0.5 | 일반 MWPM | 12.9501 | 12.9501 | 0.0000 | [0.0000, 0.0000] |
| 0.5 | correlated | 12.9654 | 12.9654 | 0.0000 | [0.0000, 0.0000] |
| 5 | 일반 MWPM | 14.1495 | 14.1937 | −0.0443 | [−0.1106, +0.0221] |
| 5 | correlated | 14.2105 | 14.2456 | −0.0351 | [−0.0945, +0.0243] |
| 20 | 일반 MWPM | 14.3127 | 14.3494 | −0.0366 | [−0.0892, +0.0160] |
| 20 | correlated | 14.3570 | 14.4318 | −0.0748 | [−0.1470, −0.0025] |

세 편향 조건 모두 편향 DEM을 사용한 일반 MWPM의 개발 오류율이 더 낮아,
pre-decoder 비교의 기준선으로 일반 MWPM을 선택했다.
일반 MWPM에서 편향 반영의 추가 이득은 확인되지 않았다.
correlated의 eta=20 결과도 여러 조건을 조사한 개발 결과로 해석한다.

집계 원본은
[baseline_summary.json](../../../artifacts/eval/d7_pauli_bias/pilot/baseline_summary.json)에 있다.

## 기존 pre-decoder의 추가 효과

다음 표는 일반 MWPM과 pre-decoder를 같은 16,384 shots에서 비교한 결과다.
baseline 전체 65,536 shots의 오류율과 표본 크기가 다르다.
개선량은 편향 DEM을 사용하는 일반 MWPM 대비 값이다.

| 편향 eta | PyMatching 오류율 (%) | pre-decoder 오류율 (%) | 고친 수 | 해친 수 | 개선량 (pp) | 95% 구간 (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5 | 13.0127 | 12.7075 | 114 | 64 | +0.3052 | [+0.1544, +0.4560] |
| 5 | 14.1479 | 14.0198 | 76 | 55 | +0.1282 | [−0.0693, +0.3257] |
| 20 | 14.7278 | 14.5386 | 87 | 56 | +0.1892 | [−0.0016, +0.3800] |

eta=5와 eta=20은 양의 점추정을 보였지만 구간이 0을 포함했다.
eta=20의 구간 하한 −0.0016 pp도 하한이 0을 초과하는 개선 근거를 충족하지 못한다.
eta=0.5의 개발 결과에서는 양의 구간을 얻었다.

같은 local map에서 eta=0.5 대비 pre-decoder 개선량의 차이는
eta=5에서 −0.1770 pp [−0.4521, +0.0981],
eta=20에서 −0.1160 pp [−0.2877, +0.0558]이었다.
이 비교도 구간에 0이 포함돼 편향 조건에서 추가 효과가 더 커졌는지는 불확실하다.
각 편향 조건의 shot은 별도 seed로 생성했으므로 이 비교는 맵 단위로 짝지었다.

기존 A/B 재학습 seed 2 checkpoint의 후보 생성기와 margin selector를 사용했다.
selector utility 차이에 적용하는 저장된 margin은 1.25다.
후보의 논리 전이 특징은 편향 DEM과 일반 MWPM으로 다시 계산했다.
기존 배포 결정이 raw 반환이어도 파일럿에서는 후보·selector 경로를 직접 실행했다.
따라서 이 결과에는 해당 경로를 실행한 효과가 포함된다.
이번 파일럿에서 bias 데이터로 모델을 재학습하거나 threshold를 보정한 과정은 없다.

원본 오답 중 정답 후보를 포함한 비율은 eta=0.5에서 87.10%,
eta=5에서 87.70%, eta=20에서 88.48%였다.
최종 논리 예측을 바꾼 shot은 각각 178, 131, 143개였다.
원본 유지 후보도 포함하므로 oracle의 회복 여지와 실제 채택 결과를 함께 해석한다.
모든 조건에서 후보 복호화 fallback은 0이었다.

예측 집계는
[selector_summary.json](../../../artifacts/eval/d7_pauli_bias/pilot/selector_summary.json)에 있다.
24개 데이터셋과 24개 pre-decoder 예측 배열에서 오류 수·개선량·구간을 다시 계산한
[pilot_validation.json](../../../artifacts/eval/d7_pauli_bias/pilot/pilot_validation.json)과 일치했다.

## 검증과 해석 범위

기본 eta=0.5 조건은 기존 생성 경로와 circuit, DEM, 같은 sampler seed의
전체 배열이 일치했다. 24개 저장 회로에서 noise-free circuit과 detector 순서,
변경 대상 외의 잡음 명령이 일치했다.
메모리에서의 채널 확률 합은 상대 허용오차 10⁻¹²로 검증했다.
Stim의 유효숫자 6자리 회로 텍스트에서는 확률 합의 최대 절대 차이가 1.0×10⁻⁸,
최대 상대 차이가 8.12×10⁻⁶이었다.
저장 회로 검증은 상대 허용오차 10⁻⁵로 통과했다.

실제 eta=20 입력 32 shots에서 정답 라벨을 바꿔도 후보 mask, 특징,
selector 점수와 선택 결과가 일치했다. 학습 입력의 라벨 집계 채널은 0으로 유지했다.
실행 전 고정한 소스 파일 해시와 checkpoint 해시를 확인했으며,
해당 소스의 스냅샷을 보관했다.
관련 Python 파일의 py_compile 검사와 전체 37개 테스트를 통과했다.

selector_seconds는 후보 추론과 선택 후보의 재복호화 확인 구간을 측정한다.
모델 적재, 입력 준비, 첫 raw 복호화를 포함한 전체 실행 지연은 별도 실험에서 평가한다.
DEM에는 approximate_disjoint_errors 근사가 포함된다.
이번 eta는 데이터 큐비트의 변경 대상 단일 큐비트 채널에 적용한 값이다.

판단 범위는 고정 checkpoint 하나와 8개 개발 맵이다.
독립 평가에서는 편향, DEM, 매칭 모드, checkpoint, selector 기준을 고정하고
새 맵에서 PyMatching 대비 joint logical error 감소량의 95% 구간을 계산한다.
개선이 재현되면 실행 비용을 측정한 뒤 RTL로 옮길 블록을 정한다.

검증 기록은
[잡음 확률 및 회로 검증](../../../artifacts/eval/d7_pauli_bias/pilot/noise_validation.json)과
[소스 및 label-independence 검증](../../../artifacts/eval/d7_pauli_bias/pilot/source_and_label_validation.json)에 있다.
실행 소스는
[source_snapshot.zip](../../../artifacts/eval/d7_pauli_bias/pilot/source_snapshot.zip)에 보관했다.

재실행은 Anaconda fp 환경에서 다음 명령을 사용한다.

```powershell
python tools/run_d7_pauli_bias.py baseline --out artifacts/eval/d7_pauli_bias/pilot
python tools/run_d7_pauli_bias.py selector --out artifacts/eval/d7_pauli_bias/pilot
```
