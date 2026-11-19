# d7 후보 생성 전 게이트 결과

2026-11-17 · 강한 ZZ, D₀ 일반 PyMatching · [동결 전 평가 규약](../PREDECODER_D7_PRE_GATE_PROTOCOL.md)

## 개발과 기준 고정

라운드별 detector event 수와 첫 MWPM 비용으로 게이트를 적합했다. 개발 맵 24개에서 맵을 하나씩 제외하고 점수를 계산했다. 우선 비교 기준은 기존 후단의 **막은 악화 − 놓친 복구**였다. 통과 목표 90%에서 45건의 악화를 막고 35건의 복구를 놓쳤다. 검토한 일곱 기준 중 가장 높은 순효과는 +10건(+0.010 pp)였다. 실제 개발 통과율은 90.14%였다. 전체 개발 맵에서 다시 적합한 점수 기준값 −0.0018453737913093741, 게이트 계수 및 기존 후단 기준값 0.025를 새 데이터를 생성하기 전에 고정했다. [개발 집계](../../../artifacts/eval/d7_pre_gate/20261117/development.json) · [동결 설정](../../../artifacts/eval/d7_pre_gate/20261117/protocol.json)

## 새 맵의 정확도와 시간

새 잡음 맵 788400–788423에서 512 shot씩, 총 12,288 shot을 평가했다. 정확도는 고정 checkpoint 8개의 같은 shot을 맵 안에서 먼저 평균했다. 복구·악화의 분모는 98,304 모델–shot 쌍이다. 실제 후보 탐색 실행률은 물리 shot을 분모로 90.69%(11,144 / 12,288)였다.

| 경로 | 정확도 | PyMatching 대비 차이 (95% 맵 구간) | 복구 / 악화 | 512-shot CPU 배치 시간 |
| --- | ---: | --- | ---: | ---: |
| PyMatching 단독 | 86.157% | 기준 | 0 / 0 | 3.642 ms |
| 기존 후단 정책 | 86.199% | +0.042 pp [−0.144, +0.227] | 418 / 377 | 1,212.662 ms |
| 같은 후단 + 게이트 | 86.186% | +0.028 pp [−0.148, +0.205] | 350 / 322 | 1,073.673 ms |

게이트는 기존 후단의 악화 55건을 막고 복구 68건을 놓쳤다. 기존 후단 대비 순효과는 −13건, **−0.013 pp**였고 95% 구간은 [−0.079, +0.052] pp였다. 개발 맵에서 관찰한 양의 순효과는 새 맵에서 재현되지 않았다.

맵마다 checkpoint 하나를 순환 배정해 각 checkpoint를 3개 맵에서 측정했다. 세 경로는 같은 512-shot batch를 사용했다. 경로마다 예열 1회, 측정 3회를 같은 프로세스에서 순서만 순환해 실행했다. 첫 MWPM, 특징·게이트·클래스별 비용 계산, 실제 후보 탐색과 최종 선택을 각각의 실행 경로에서 측정했다. 파일·checkpoint 로딩과 정적 맵 구조 준비는 제외했다. 표의 시간은 맵별 반복 중앙값을 다시 맵에서 중앙값으로 집계한 수치다. 기존 후단 대비 맵별 시간 절감의 평균은 **119.050 ms**였고, 95% 구간은 **[97.196, 140.905] ms**였다. 24개 맵 모두에서 배치 시간이 줄었다.

**실행 시간은 줄었고, 정확도 이득은 불확실했다.** PyMatching 대비 정확도 차이의 구간 하한이 0을 넘지 않아 두 채택 기준을 동시에 충족하지 못했다. 이 결론은 시뮬레이션의 강한 ZZ 조건, 고정 checkpoint와 CPU 배치에 해당한다. [전체 집계와 반복별 시간](../../../artifacts/eval/d7_pre_gate/20261117/summary.json)

## 검증과 재현

독립 표본 및 예측 파일 24개씩을 해시로 검증하고 저장 출력에서 복구·악화, 맵별 신뢰구간과 반복 시간을 재계산했다. 통과 shot의 예측은 모든 맵·모델에서 기존 후단과 일치했다. 시간 측정의 각 반복에서 같은 맵·checkpoint의 예측을 재현했다. 원본 비용 계산과 잡음 평가 단위 테스트를 통과했다.

```powershell
conda activate fp
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$env:OPENBLAS_NUM_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
python -m py_compile tools/evaluate_d7_pre_gate.py
python tools/evaluate_d7_pre_gate.py develop
python tools/evaluate_d7_pre_gate.py freeze
python tools/evaluate_d7_pre_gate.py run
```

[평가 코드](../../../tools/evaluate_d7_pre_gate.py)
