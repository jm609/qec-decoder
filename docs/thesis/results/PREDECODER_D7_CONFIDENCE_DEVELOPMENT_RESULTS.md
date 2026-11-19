# d7 논리 클래스 신뢰도와 selector 개발 비교

2026-11-16 · Anaconda `fp` · 논문 완성 계획 2단계

강한 ZZ의 개발 교차검증에서 기존 shot별 margin은 원본 PyMatching 대비 **+0.012 pp**였다. Paired detector head는 **−0.116 pp**, 논리 클래스 비용 차이를 추가한 head는 **−0.095 pp**였다. 같은 head에 비용 차이를 추가했을 때 **+0.021 pp** 개선했고 맵 4개 중 3개에서 높았으나, 사전에 정한 **+0.05 pp** 채택 기준에는 못 미쳤다. 다음 단계에서 사용할 정책은 기존 shot별 margin으로 정했다. [개발 규약](../PREDECODER_D7_CONFIDENCE_DEVELOPMENT_PROTOCOL.md) · [전체 집계](../../../artifacts/eval/d7_confidence_development/20261116/stage2_summary.json)

## 클래스별 비용 검증

D₀ 일반 MWPM 그래프에서 논리 observable이 붙은 edge는 경계에 있었다. 내부 연결 성분별로 같은 observable 조합을 가진 경계 edge를 하나의 보조 detector에 연결했다. 보조 detector 값은 해당 edge 중 선택된 개수의 홀짝을 나타낸다. 가능한 홀짝 조합마다 매칭하고, 같은 논리 클래스를 만드는 조합의 최소 비용을 취했다. 두 observable을 함께 바꾸는 경계 그룹도 포함했다. 기존 d7 검증 그래프에는 336개 detector와 경계 그룹 5개가 있었고, 원본 syndrome의 연결 성분별 홀짝과 맞는 경계 조합은 각 syndrome당 8개였다.

- 작은 그래프의 전체 32개 syndrome에서 각 논리 클래스 비용을 edge 집합 전수 검사와 대조했다. 도달 불가 클래스와 지원하지 않는 그래프도 확인했다.
- 기존 맵 783000의 원본 및 수정 syndrome 44개에서 네 클래스의 비용을 Dijkstra 거리와 NetworkX의 독립 최소 매칭 구현으로 다시 계산했다. 최대 절대 차이는 **0.00000117**, 허용 오차는 0.0003이었다. 보조 경계 변환은 두 구현에서 공유했고, 변환의 정확성은 작은 그래프의 전수 검사로 대조했다.
- 새 맵에서는 원본과 비교 대상 수정 syndrome에서, PyMatching이 고른 클래스의 비용과 네 클래스 중 최소 비용이 모두 제약 없는 원본 PyMatching 비용과 일치하는지 검사했다. 기존 후보 예측과 다시 디코딩한 클래스도 대조했다.

신뢰도 특징은 현재 예측에서 Z bit만 다른 클래스까지의 비용 차이다. 같은 syndrome에서 계산하므로 수정 전후 최적 비용을 빼는 기존 특징과 구분된다. [비용 검증 기록](../../../artifacts/eval/d7_confidence_development/20261116/stage2_gap_probe.json)

## 개발 교차검증과 보정 확인

개발 맵 787000–787003과 별도 보정 맵 787100–787101을 사용했다. 맵·강도마다 1,024 shot을 d7 재학습 checkpoint seed 0–7이 공유했다. 개발 단계에서는 G₁의 세 맵으로 head를 학습하고 남은 맵의 G₀와 G₁을 평가했다. 보정 맵에는 G₁ 개발 맵 네 개로 다시 학습한 head를 적용했다. 아래 수치는 원본 유지가 가능한 경로의 raw 대비 순이득이다.

| 방법 | 개발 G₀ (pp) | 개발 G₁ (pp) | 개발 G₁ 개선 / 악화 | 보정 G₀ (pp) | 보정 G₁ (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 top-1 | −4.758 | −5.124 | 1,096 / 2,775 | −3.613 | −3.888 |
| 기존 shot별 margin | −0.009 | **+0.012** | 58 / 54 | +0.061 | −0.018 |
| Paired, detector 특징 제거 | −0.003 | −0.058 | 4 / 23 | 0.000 | +0.012 |
| Paired detector | **+0.064** | −0.116 | 45 / 83 | +0.024 | 0.000 |
| Paired detector + gap | +0.027 | −0.095 | 66 / 97 | +0.067 | 0.000 |
| Z 후보 대 no-edit | −0.055 | −0.064 | 23 / 44 | +0.043 | −0.043 |
| 원본 정보 Z 전환 | 0.000 | 0.000 | 0 / 0 | 0.000 | 0.000 |
| 원본 정보 Z 전환 + raw 비용·gap | −0.003 | 0.000 | 0 / 0 | 0.000 | 0.000 |

개발 G₁의 비교 분모는 4,096 shot에서 얻은 32,768 모델–shot 쌍이다. 기존 margin의 +0.012 pp는 정답 4쌍 순증에서 나왔다. 보정 G₁에서는 29쌍을 고치고 32쌍을 해쳐 −0.018 pp였다. 두 단순 Z 전환 head는 개발 G₁에서 전환을 적용하지 않았다. Paired detector와 gap 변형은 개발과 보정의 두 강도에서 수정 shot 고정 경로와 원본 유지 경로의 논리 예측이 각각 일치했다.

개발 G₁의 gap 변형은 detector head보다 개선과 악화를 모두 늘렸다. 이득 차이는 +0.021 pp였고, margin에 비해서는 −0.107 pp였다. G₀ 개발에서 detector 특징을 추가한 head의 이득이 더 컸지만, 방법 선택에는 사전에 지정한 G₁만 사용했다. 새로운 8개 seed 패널과 다시 학습한 head에서는 기존 네 checkpoint의 독립 평가에서 관측한 강한 ZZ 이득이 재현되지 않았다. 패널, 맵, 학습 데이터가 함께 달라졌으므로 차이를 한 요인에 돌릴 수 없다. [기존 네 checkpoint 평가](PREDECODER_D7_PAIRED_SELECTOR_INDEPENDENT.md)

## Gap 구간별 오류율과 실행 비용

개발 G₁에서 구한 raw Z gap 사분위 경계는 5.093, 7.354, 11.826이었다. 별도 보정 맵에서 두 observable 중 하나라도 틀린 비율은 낮은 구간부터 G₀ **18.4%, 28.7%, 5.0%, 2.8%**, G₁ **21.7%, 27.8%, 2.8%, 4.6%**였다. 정확한 클래스 비용을 계산해도 이 Z gap 하나로 전체 logical_class4 오류율을 단조롭게 나타내기 어려웠다. 이 비교는 강도당 보정 맵 두 개의 관측값이다.

후보 생성은 worker 2개로 **36.3분** 걸렸다. 전체 96개 후보 묶음에서 매칭 실패는 0건이었다. Gap 특징 계산 자체의 집계 시간은 **12.7초**였다. 후보·특징 계산 시간은 개발 실행 조건에서 측정했다.

96개 후보 파일 해시를 대조했다. 저장한 클래스 예측과 정답에서 경로별 개선·악화·예측 변경 건수를 다시 계산했고, 요약과 모두 일치했다. 새 특징 경로에서 정답을 바꿔도 입력 특징과 후보 preselection이 같았다. 문법 검사와 회귀 테스트 35개를 통과했다.

```powershell
conda activate fp
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
python tools/collect_d7_confidence_candidates.py
python tools/evaluate_d7_confidence_development.py probe
python tools/evaluate_d7_confidence_development.py compare
python -m unittest discover -s tests
```

후보 수집 명령은 저장 파일을 해시로 확인하고 재사용한다. `probe`와 `compare`는 결과가 없는 실행 경로에서 출력 파일을 생성하도록 설계했다. [저장 예측](../../../artifacts/eval/d7_confidence_development/20261116/stage2_predictions.npz) · [검증 기록](../../../artifacts/eval/d7_confidence_development/20261116/stage2_verification.json)

3단계에서는 선택된 **기존 shot별 margin**을 주 정책으로 두고, **refit paired detector**를 이전 결과와 연결하는 대조 경로로 사용한다. 각 셀의 raw를 기준으로 동일 mask 재디코딩과 전체 경로 재계산을 분리한다. 4단계의 주 비교는 방법과 비교군이 하나의 정책으로 겹치는 문제를 최종 규약 동결 전에 정리한다.
