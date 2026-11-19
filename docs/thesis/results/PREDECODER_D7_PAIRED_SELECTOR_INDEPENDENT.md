# d7 paired selector의 새 잡음 맵 평가

2026-11-16 · Anaconda `fp` 환경에서 실행

원본 유지를 허용한 paired selector는 강한 ZZ의 새 맵 8개에서 원본 PyMatching 대비
**+0.148 pp**를 기록했다. 맵 단위 95% 구간은 **+0.020~+0.296 pp**였다. 65,536
모델–shot 쌍에서 원본 오답 537건을 고치고 원본 정답 440건을 해쳐 97건이 순증했다.
[집계 JSON](../../../artifacts/eval/d7_paired_selector_independent/20261116/summary.json)

## 평가 조건

앞선 개발 맵 교차 검증에서 detector 패턴을 사용한 head는 원본 PyMatching 대비
+0.092 pp였다. 이번에는 개발 맵 783000과 783001의 사례를 합쳐 같은 head를 학습했다.
입력과 log-odds 0 선택 기준을 고정해 새 맵을 생성하기 전에 저장했다. 학습에는 Z 전이
후보와 원본 예측을 보존하는 수정 후보 중 하나만 정답이었던 2,996쌍을 사용했다.
[개발 맵 결과](PREDECODER_D7_PAIRED_SELECTOR_RESULTS.md) · [고정 계수](../../../artifacts/eval/d7_paired_selector_independent/20261116/calibration.json)

강한 ZZ 조건(G₁, `zz_factor=0.25`)에서 기준 DEM(D₀, `zz_factor=0.05`)으로 일반 MWPM을
적용했다. 이전 실험과 겹치지 않는 맵 786100–786107에서 맵당 2,048 shot을 평가했다.
네 checkpoint(2, 11, 54, 57)는 같은 16,384 shot을 공유한다. 건수는 모델–shot 65,536쌍에서
집계했다. head의 후보 선택 경로에는 label 대신 0을 넣었고, 선택을 마친 뒤 정답을 사용해
개선·악화를 계산했다. [고정 평가 규약](../../../artifacts/eval/d7_paired_selector_independent/20261116/protocol.json)

주 비교는 원본 유지를 허용한 paired selector와 원본 PyMatching의 정확도 차이다.
보조 비교에는 기존 1위, 기존 margin, 기존 1위가 수정한 shot을 고정한 paired selector를
포함했다. 정확도 차이는 각 맵 안에서 네 checkpoint를 평균한 뒤, 맵에 같은 가중치를 줘
계산했다. 95% 구간은 맵 단위 paired bootstrap 20,000회로 구했다.

## 결과

| 경로 | 수정 | 개선 | 악화 | 원본 대비 정확도 차이 | 95% 구간 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 1위 | 14,354 | 4,291 | 10,057 | −8.798 pp | −10.124~−7.445 pp |
| 기존 margin | 897 | 414 | 483 | −0.105 pp | −0.279~+0.078 pp |
| 수정 shot 고정 paired selector | 14,354 | 537 | 440 | +0.148 pp | +0.020~+0.296 pp |
| 원본 유지 허용 paired selector | 977 | 537 | 440 | +0.148 pp | +0.020~+0.296 pp |

원본 유지 허용 경로는 기존 margin보다 +0.253 pp 높았다(95% 구간 +0.110~+0.386 pp).
원본 대비 맵별 차이는 −0.122~+0.574 pp였다. 5개 맵에서 개선했고, 2개 맵에서
악화했으며, 1개 맵에서 정확도가 같았다.

기존 1위가 적용한 Z 전이 14,354건 모두에 원본을 보존하는 비영 수정 후보가 있었다.
수정 shot을 고정한 경로에서 13,377건을 그 후보로 바꿨다. 원본 유지를 허용한 경로는
이 shot에서 no-edit를 골랐다. 두 경로가 낸 논리 예측은 65,536쌍 모두 같았다. 평가한
후보는 2,162,688개이며, 수정 적용 건수와 실제 매칭 호출·처리 시간은 별도의 지표다.

강한 ZZ와 고정 checkpoint에서 원본 대비 순이득을 확인했다. 다음 평가는 학습 계수와
선택 기준을 고정한 상태에서 약한 ZZ 조건의 정확도, 추가 매칭 호출, 처리 시간을 확인한다.

## 재현

```powershell
conda activate fp
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
python tools/evaluate_d7_paired_selector.py --out artifacts/eval/d7_paired_selector_independent/20261116 --jobs 4
python tools/evaluate_d7_paired_selector.py --out artifacts/eval/d7_paired_selector_independent/20261116 --jobs 4 --replay
```

`fp` 환경에서 NumPy의 MKL과 PyTorch를 함께 불러와 head를 학습할 때 OpenMP 충돌이
발생했다. 해당 실행의 MKL만 단일 스레드로 설정해 완료했다. `--replay`는 고정 계수와
저장된 후보 예측의 해시를 확인하고 경로별 결과와 bootstrap 구간을 다시 계산한다.
예측 파일 32개로 재현한 결과는 최초 집계와 같았다. 회귀 테스트 32개를 통과했다.
