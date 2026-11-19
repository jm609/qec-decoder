# 논문 집필용 근거 묶음

2026-11-17 · 논문 완성 계획 6단계 · d7 국소 syndrome 수정의 진단과 독립 평가

**핵심 주장.** 평가한 d7 회로에서 사용 가능한 후보에는 회복 여지가 컸다. 원본과 수정 결과를 같은 shot에서 비교하자 순위·채택에서 발생하는 손실을 확인할 수 있었다. 선정한 shot별 margin은 24개 독립 맵의 두 주 비교에서 개선 여부가 불확실했고, 현 CPU 구현에서는 추가 계산 비용이 컸다.

## 주장과 근거

| 본문에서 다룰 주장 | 근거 | 해석 범위 |
| --- | --- | --- |
| 수정 후 기존 predecoder의 성능은 거리별로 달랐다. | d3 **+2.260 pp**, d5 **+0.022 pp**, d7 **0.000 pp**. d5에서 1/8, d7에서 0/8 checkpoint가 수정을 채택했다. [입력 수정 후 재검증](PREDECODER_INDEPENDENT_REVALIDATION.md) | 기존 family 정책의 배경 결과다. d7 shot별 margin·paired 독립 평가와 checkpoint 및 평가 규약에 차이가 있다. |
| 같은 shot의 후보 비교로 d7 실패 단계를 나눌 수 있다. | 강한 ZZ의 D₀ 일반 MWPM에서 **25,126쌍**의 top-1 피해 모두에 원본 정답을 보존하는 다른 비영 후보가 있었다. top-1 **−8.501 pp**, 수정 shot 고정 oracle **+6.107 pp**, 전체 후보 oracle **+11.543 pp**. [전이 진단](results/PREDECODER_D7_LOGICAL_TRANSITION_DIAGNOSTIC.md) | 이 비교는 8맵·4 checkpoint의 사후 진단이다. 두 oracle은 정답을 사용하는 상한이다. |
| 강한 ZZ에서 후보 oracle과 실제 순위의 격차가 커졌다. | D₀ 일반 기준 격차는 **18.731 pp**에서 **20.045 pp**로 늘었다. 사전 지정한 top-1 순이득 변화는 **−0.469 pp**, 97.5% 구간 [−0.939, +0.065] pp였다. [잡음 변화 평가](results/PREDECODER_D7_NOISE_SHIFT_RESULTS.md) | 격차 증가는 진단 결과다. top-1 효과의 강도 차이는 불확실하다. ZZ 확률 증가와 총 잡음 증가가 함께 일어났다. |
| 개발 단계에서 클래스별 matching 비용 특징을 확인했으나 채택 기준을 충족하지 않았다. | 작은 그래프 전수 검사와 기존 d7 syndrome 44개의 독립 매칭을 통과했다. 강한 ZZ에서 paired detector의 **−0.116 pp**가 gap 추가 후 **−0.095 pp**가 됐으나 차이 **+0.021 pp**는 **+0.05 pp** 기준보다 작았다. margin을 선정했다. [B 개발 비교](results/PREDECODER_D7_CONFIDENCE_DEVELOPMENT_RESULTS.md) | 8개 checkpoint의 개발 4맵 교차검증에 근거한다. 동결한 비교 방식에서는 원본 syndrome만 본 두 Z 전환 대조군도 개발 G₁에서 전환하지 않았다. |
| 후단 디코더와 수정 선택의 효과를 구분해야 한다. | 보정 G₁의 matched D₁ correlated raw는 기준 D₀ 일반 raw보다 **+0.244 pp**였다. 같은 D₁ correlated 셀에서 margin의 행동 고정은 **−0.031 pp**, 전체 경로는 **+0.006 pp**였다. [활성 정책의 교차 평가](results/PREDECODER_D7_ACTIVE_PRIOR_RESULTS.md) | 2개 보정 맵의 기술 통계다. 4단계 새 맵에서는 같은 원본 간 차이가 **+0.114 pp**였다. 같은 셀에서 margin 추가 효과는 **−0.004 pp**였다. |
| 선정한 margin의 독립 개선 근거는 확보되지 않았다. | G₁/D₀ 일반 주 비교에서 원본 대비 **−0.035 pp**, 97.5% [−0.082, +0.013] pp. Paired 대비 **−0.008 pp**, 97.5% [−0.081, +0.065] pp. Oracle은 **98.115%**, 원본은 **86.344%**였다. [독립 최종 평가](results/PREDECODER_D7_FINAL_EVALUATION_RESULTS.md) | 새 맵 24개, 강한 ZZ 12,288 shot, 여덟 고정 checkpoint에 조건부다. 두 구간 모두 0을 포함한다. |
| 현 selector 구현의 배치 처리 비용이 컸다. | 256 shot 기준 raw **1.6–2.4 ms**, margin **9.2–12.5초**, paired **8.2–10.2초**. 두 selector가 배치당 8,704개 syndrome 행을 처리했다. [CPU 비용과 재현성](results/PREDECODER_D7_FINAL_COST_REPRODUCIBILITY.md) | 한 CPU·맵·checkpoint의 배치 측정이다. 반복 간 변동이 커 두 selector의 속도 순위는 판단하기 어렵다. |

## 그림과 함께 넣을 표

그림 표기는 영문으로 통일했다. 아래 문단을 캡션의 바탕으로 쓰고, 단위·표본과 비교 범위를 유지한다.

| 번호 | 그림과 캡션 범위 | 표 데이터 |
| --- | --- | --- |
| 1 | [동일-shot 방법 흐름](../../artifacts/figures/d7_paper_evidence/20261117_final/fig01_same_shot_method.pdf). 원본·수정 syndrome의 두 observable 예측을 얻어 margin 또는 paired로 선택한다. 정답을 사용하는 평가는 선택이 끝난 뒤 수행한다. | [방법 설명](PREDECODER_METHOD_DESCRIPTION.md) |
| 2 | [후보 순위·채택 진단](../../artifacts/figures/d7_paper_evidence/20261117_final/fig02_failure_diagnostic.pdf). 왼쪽은 D₀ 일반의 top-1과 두 oracle의 정확도 차이, 오른쪽은 강한 ZZ의 top-1, margin, family gate가 개선·악화한 모델–shot 쌍 수다. 8맵·4 checkpoint를 사용했다. | [잡음 교차 평가 표](../../artifacts/figures/d7_paper_evidence/20261117_final/noise_shift.csv) · [활성 정책 prior 표](../../artifacts/figures/d7_paper_evidence/20261117_final/active_prior.csv) |
| 3 | [선택 특징과 단순 대조군](../../artifacts/figures/d7_paper_evidence/20261117_final/fig03_selector_development.pdf). 각 원은 강한 ZZ 개발 검증 맵 하나, 마름모는 4맵의 동일 가중 평균이다. 비교의 기준은 D₀ 일반 raw다. | [특징 비교 표](../../artifacts/figures/d7_paper_evidence/20261117_final/selector_ablation.csv) |
| 4 | [독립 평가의 주 효과](../../artifacts/figures/d7_paper_evidence/20261117_final/fig04_independent_effects.pdf). 왼쪽은 주 비교의 맵별 점과 전체 평균의 97.5% 구간, 오른쪽은 고정 checkpoint별 기술 통계다. 세로 흩뿌림은 겹치는 점을 보여 주기 위한 표시다. | [주·보조 비교 표](../../artifacts/figures/d7_paper_evidence/20261117_final/independent_cells.csv) · [맵별 순이득](../../artifacts/figures/d7_paper_evidence/20261117_final/independent_map_effects.csv) |
| 5 | [배치 비용과 정확도](../../artifacts/figures/d7_paper_evidence/20261117_final/fig05_cost_and_accuracy.pdf). 왼쪽은 한 맵·checkpoint의 CPU 배치 중앙값과 반복 범위, 오른쪽은 D₀ 일반에서 24개 독립 맵의 raw 대비 정확도 차이다. 원과 사각형으로 약한·강한 ZZ를 구분한다. | [CPU 비용 표](../../artifacts/figures/d7_paper_evidence/20261117_final/cpu_cost.csv) |

거리별 재검증 수치는 그림 1 뒤 연구 배경에 넣는다. d7의 `[0, 0]` 구간은 family 정책이 전부 raw를 선택한 결과다. [별도 거리 비교 표](../../artifacts/figures/d7_paper_evidence/20261117_final/revalidation.csv)

## 원고 순서와 범위

1. 문제 정의에서 실제 선택 이후의 logical_class4 성능을 연구 질문으로 제시한다. d3/d5 재검증은 배경에 두고 d7 실패의 원인을 진단할 필요를 연결한다.
2. 방법에서 D₀/D₁, 두 observable과 논리 전이, 후보 oracle, shot별 margin과 family gate의 차이를 정의한다. 논리 클래스 gap은 **동일 syndrome**에서 두 클래스의 최적 비용 차이로 정의한다.
3. 결과에서 A의 순위·채택 진단, B의 개발 비교, 활성 정책의 prior 비교 순서로 최종 정책 선택의 이유를 설명한다. 수정 전후 최적 matching 비용을 더한 실험은 고정 수정 shot의 논리 예측이 같았고, 전이 상수 보정은 고정 수정 shot의 후보를 바꾸지 않았다. 이 두 결과를 설계 근거나 부록으로 남긴다. [Matching 비용 재순위](results/PREDECODER_D7_MATCHING_COST_RERANK_RESULTS.md) · [전이 상수 보정](results/PREDECODER_D7_TRANSITION_RERANK_RESULTS.md)
4. 독립 평가를 주 결과로 제시한다. G₁/D₀ 일반의 두 주 비교, 각 97.5% 구간, 개선·악화 수를 한 표에 둔다. 이전 네 checkpoint의 paired 결과는 그때의 조건부 독립 결과로 배치한다. 그때와 이번의 맵·패널·head가 달라 원인을 한 요인에 귀속하기 어렵다. [이전 paired 평가](results/PREDECODER_D7_PAIRED_SELECTOR_INDEPENDENT.md)
5. 한계에서 고정 모델 패널, 시뮬레이션 회로와 두 ZZ 강도, 후보 집합의 oracle 상한, 한 CPU의 배치 비용 범위를 명시한다. 여러 거리의 개선을 주장할 때 각 거리의 실제 자료와 정책을 함께 적는다. 과거 입력 누출의 영향은 방법 또는 재현성 절에서 설명한다. [입력 누출·수정 근거](PREDECODER_RESEARCH_INTEGRITY_AUDIT.md)

## 수치 재생성과 파일 인계

강한 ZZ 주 비교에 해당하는 예측 파일 192개의 해시를 확인하고 저장 예측과 정답에서 개선·악화 수, 맵별 효과와 두 97.5% 구간을 새로 계산했다. 4단계 집계와 일치했다. [주 비교 재계산](../../artifacts/figures/d7_paper_evidence/20261117_final/primary_recalculation.json) · [그림·표 매니페스트](../../artifacts/figures/d7_paper_evidence/20261117_final/manifest.json)

```powershell
conda activate fp
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'artifacts/figures/d7_paper_evidence/mpl-cache'
python tools/build_d7_paper_evidence.py --out artifacts/figures/d7_paper_evidence/new_run
```

묶음에는 이 문서와 근거 보고서, PDF·PNG 5쌍, CSV 7개, 재계산 JSON, 생성 코드와 원본 집계 해시를 포함한다. 전체 예측·가중치는 5단계 매니페스트에서 추적한다. [집필용 근거 묶음 ZIP](../../artifacts/figures/d7_paper_evidence/20261117_final/paper_evidence.zip) · [CPU·평가 재현 매니페스트](../../artifacts/eval/d7_final_benchmark/20261117/reproducibility_manifest.json)
