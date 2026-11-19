# d7 paired selector의 약한 ZZ 평가와 추론 비용

2026-11-16 · Anaconda `fp` 환경에서 실행

고정 head는 약한 ZZ에서 원본 PyMatching 대비 **+0.203 pp**를 기록했다. 맵 단위 95%
구간은 **+0.096~+0.327 pp**였다. 같은 CPU의 512-shot 배치에서 selector 경로는
11.585초, 원본 PyMatching은 2.638 ms였다.
[정확도 집계](../../../artifacts/eval/d7_paired_selector_weak/20261116/summary.json) · [처리 시간 JSON](../../../artifacts/eval/d7_paired_selector_weak/20261116/benchmark.json)

## 평가 조건

강한 ZZ 검증에 사용한 맵 786100–786107에서 저장해 둔 약한 ZZ(G₀, `zz_factor=0.05`)
shot을 평가했다. G₀와 G₁은 맵별 물리 잡음 구성을 공유하고, ZZ 강도와 sampler seed가
다르다. 각 강도에서 맵당 2,048 shot을 네 checkpoint(2, 11, 54, 57)가 공유한다. 기준
DEM(D₀)과 일반 MWPM, head의 학습 계수와 log-odds 0 기준을 고정했다. 주 비교는 원본
유지를 허용한 selector와 G₀ 원본 PyMatching의 정확도 차이다. G₁과의 차이는 맵별로
각 강도의 원본 대비 이득을 뺀 뒤 비교했다. 구간은 맵 단위 bootstrap 20,000회로 구했다.
[강한 ZZ 평가](PREDECODER_D7_PAIRED_SELECTOR_INDEPENDENT.md) · [고정 규약](../../../artifacts/eval/d7_paired_selector_weak/20261116/protocol.json)

기존 selector는 후보 utility를 계산하면서 원본과 각 후보의 논리 예측 차이를 전이
특징에 기록한다. 이 값을 사용해 후보 예측을 복원하고, 같은 후보를 다시 매칭하는 호출을
제외한 추론 경로를 측정했다. 후보 생성과 utility, head 선택 규칙은 이전 평가와 같다.
처리 시간은 맵 786100의 checkpoint 2에서 각 강도의 첫 512 shot을 사용해 별도로
측정했다. 단일 프로세스에서 PyTorch CPU 스레드 2개를 사용했다. DEM, 모델과 입력
텐서는 메모리에 준비한 상태에서 첫 실행으로 워밍업하고 이후 3회 중앙값을 계산했다.
원본 경로는 PyMatching 호출을, selector 경로는 원본 매칭·신경망 추론·후보 매칭·
head 선택을 포함한다. 디스크 접근과 초기 로딩은 시간에서 제외했다.

## 정확도

| 약한 ZZ 경로 | 수정 | 개선 | 악화 | 원본 대비 정확도 차이 |
| --- | ---: | ---: | ---: | ---: |
| 기존 1위 | 13,348 | 3,950 | 9,380 | −8.286 pp |
| 기존 margin | 780 | 396 | 383 | +0.020 pp |
| 수정 shot 고정 paired selector | 13,348 | 511 | 378 | +0.203 pp |
| 원본 유지 허용 paired selector | 891 | 511 | 378 | +0.203 pp |

원본 유지 허용 경로는 65,536 모델–shot 쌍에서 원본 오답 511건을 고치고 원본 정답
378건을 해쳤다. 정확도는 원본 86.926%에서 87.129%로 올랐다. 기존 margin 대비 차이는
+0.183 pp(95% 구간 +0.053~+0.311 pp)였으며,
원본 대비 이득은 맵마다 +0.024~+0.500 pp였다. 수정 shot을 고정한 경로에서는 기존
수정 13,348건 중 12,457건을 다른 수정으로 바꿨다. 두 경로의 논리 예측은 모두 같았다.

강한 ZZ의 이득은 +0.148 pp였다. 맵별로 두 강도의 원본 대비 이득을 빼면
`G₁ − G₀ = −0.055 pp`였고, 95% 구간은 −0.244~+0.136 pp였다. 이 구간에는 0이
포함되므로, 이번 결과로 강도에 따른 이득 차이를 확인하기 어렵다.

## CPU 추론 비용

| 조건 | 원본 시간 | selector 시간 | 시간 비율 |
| --- | ---: | ---: | ---: |
| G₀, 512 shot | 2.638 ms | 11.585 s | 4,391배 |
| G₁, 512 shot | 3.328 ms | 11.483 s | 3,451배 |

두 강도 모두 512 shot당 원본 경로는 매칭 1회로 512개 syndrome을, selector 경로는
513회로 17,408개 syndrome을 처리했다. G₀ 전체에서 매칭 배치 호출은 32회에서
65,568회로, 처리한 syndrome은 65,536개에서 2,228,224개로 늘었다. 추가 매칭은 모두
배치 호출이었다. 512-shot 실측에서 selector 안의 매칭 함수는 약 0.12초를 사용했다.
원본 유지를 허용해 적용한 수정이 891건으로 줄었어도, 그 결정을 내리기까지 후보 생성과
점수 계산을 거친다. 이 구현의 CPU 비용은 두 강도에서 관측한 정확도 이득과 함께
고려해야 한다.

## 재현

```powershell
conda activate fp
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
python tools/evaluate_d7_paired_selector_weak.py evaluate --out artifacts/eval/d7_paired_selector_weak/20261116 --jobs 4
python tools/evaluate_d7_paired_selector_weak.py replay --out artifacts/eval/d7_paired_selector_weak/20261116 --jobs 4
python tools/evaluate_d7_paired_selector_weak.py benchmark --out artifacts/eval/d7_paired_selector_weak/20261116
```

`evaluate`와 `replay`는 저장한 예측과 요약이 있으면 해시와 결과를 확인한다. 처리 시간
파일은 실행마다 달라지므로 다시 측정할 때는 새 출력 경로에서 먼저 평가한다.
G₀ 예측 파일 32개의 후보 2,162,688개를 D₀로 직접 다시 매칭해 복원한 예측과 대조했다.
모두 일치했고, 회귀 테스트 33개를 통과했다.
[검증 JSON](../../../artifacts/eval/d7_paired_selector_weak/20261116/verification.json)
