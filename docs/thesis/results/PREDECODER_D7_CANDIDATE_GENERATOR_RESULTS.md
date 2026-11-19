# d7 후보 생성기 대조 실험 결과

2026-11-17 · 강한 ZZ, D₀ 일반 PyMatching · [평가 규약](../PREDECODER_D7_CANDIDATE_GENERATOR_PROTOCOL.md)

## 개발 및 독립 비교

기존 개발 맵 4개에서 checkpoint 8개의 후보 mask, 순서, 논리 예측을 저장 예측과 전수 대조했다. 모두 일치했다. 비학습 국소 열거의 원본 오답 회복률은 반경 2에서 79.72%, 반경 4에서 72.51%였다. 반경 2를 고정한 뒤 새 잡음 맵 24개에서 평가했다. 총 12,288 shot 중 원본 PyMatching이 틀린 shot은 1,658개였고, 원본 정확도는 86.507%였다. [개발 집계](../../../artifacts/eval/d7_candidate_generators/20261117/development.json) · [동결 규약](../../../artifacts/eval/d7_candidate_generators/20261117/protocol.json)

| 비영 후보 예산 | 원본 오답 회복률, 학습 | 원본 오답 회복률, 국소 | 차이 (pp) | 평균 논리 클래스 수, 학습 | 평균 논리 클래스 수, 국소 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 15.21% | 13.51% | +1.70 | 1.09 | 1.59 |
| 8 | 30.75% | 27.79% | +2.97 | 1.20 | 2.00 |
| 16 | 58.51% | 51.63% | +6.88 | 1.69 | 2.47 |
| **32** | **88.70%** | **77.63%** | **+11.07** | **2.47** | **2.93** |

오답 회복률은 원본 오답 중 정답 논리 클래스를 내는 후보가 하나 이상 있는 비율이다. 맵마다 학습 방식은 8개 checkpoint를 동일 가중했고, 두 방식 모두 맵을 동일 가중했다. 주 비교인 후보 32개에서 학습 방식이 24개 중 22개 맵에서 앞섰다. 맵별 짝차이의 평균은 **+11.07 pp**, 양측 95% 신뢰구간은 **[+7.63, +14.51] pp**였다. 후보 4·8·16의 결과는 기술 통계다.

후보 32개를 사용한 oracle 정확도는 학습 방식 98.449%, 국소 방식 96.997%였다. 네 논리 클래스가 모두 나온 비율은 학습 방식 21.16%, 국소 방식 30.41%였다. 국소 방식이 더 많은 논리 클래스를 냈고, 학습 방식이 원본 오답에서 정답 클래스를 더 자주 제공했다. 두 방식 모두 매칭 불가 후보는 0개였다. [맵별 집계와 신뢰구간](../../../artifacts/eval/d7_candidate_generators/20261117/summary.json)

## 비용과 해석

| 512-shot CPU 배치 중앙값 | 학습 | 국소 |
| --- | ---: | ---: |
| 후보 패턴 배치 준비 | 20.2 ms | 4.8 ms |
| 후보 mask 생성 | 1,198.8 ms | 96.0 ms |
| 후보 후단 디코딩 | 185.6 ms | 312.0 ms |

후보 mask 생성 시간은 학습 방식에서 약 12.5배 길었다. 학습 방식의 생성 시간에는 입력 텐서 구성과 CNN 추론을 포함한다. 데이터 로딩과 checkpoint 로딩은 제외했다. 두 CPU worker가 맵을 병렬 처리한 실행에서 학습 방식은 192개 맵–checkpoint 배치, 국소 방식은 24개 맵 배치의 중앙값을 보고한다. 기존 채택 정책은 이 실행에서 호출하지 않았다.

같은 후보 예산에서 고정한 학습 생성기의 회복률이 이 국소 규칙보다 높았다. 실제 비영 후보 수는 평균 32.00개와 31.96개였다. 학습 방식이 선택한 비영 후보는 전부 2비트였고, 국소 방식은 65.9%가 2비트였다. 국소 규칙은 반경 2를 사용했고, 학습 방식은 학습된 패턴의 위치 범위를 사용했다. 따라서 효과 크기는 후보 범위, 편집 weight 구성과 순위가 함께 달라진 두 생성 방식의 비교에 해당한다. 신뢰구간은 고정한 checkpoint 8개에 조건부다. 정답을 사용한 oracle 회복률은 후보 집합의 상한이며, 채택 정책이 실제로 원본보다 나아지는지는 별도 평가가 필요하다.

## 재현

실행은 Anaconda `fp`, `MKL_THREADING_LAYER=SEQUENTIAL`에서 진행했다. 새 평가 생성 전 checkpoint와 코드 해시, 패키지 버전, 개발 규칙과 seed를 고정했다. 완료 후 새 데이터 24개의 해시 및 예측 파일 24개의 해시를 검증하고, 저장 예측에서 회복률과 논리 클래스 수, 맵별 차이와 구간을 재계산했다.

```powershell
conda activate fp
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
python -m py_compile tools/evaluate_d7_candidate_generators.py
python tools/evaluate_d7_candidate_generators.py develop
python tools/evaluate_d7_candidate_generators.py freeze
python tools/evaluate_d7_candidate_generators.py run
```

실행 스크립트는 이전 결과를 보호하기 위해 개발 집계, 동결 규약과 최종 집계 파일을 새로 만들 때만 기록한다. 저장 결과는 [실행 폴더](../../../artifacts/eval/d7_candidate_generators/20261117/)에 있다. [실험 코드](../../../tools/evaluate_d7_candidate_generators.py)
