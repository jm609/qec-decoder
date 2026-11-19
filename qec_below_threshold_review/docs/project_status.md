# QEC Below-Threshold 논문 리뷰 프로젝트 현황

> 기준일: 2026-08-24\
> 현재 단계: **공개 Willow 데이터 재디코딩과 결과 통합 완료, 포스터 내용 확정 단계**

## 1. 전체 작업 현황

| 작업 | 상태 | 주요 산출물 |
|---|---|---|
| 논문 본문 검토 | 완료 | `paper_summary.md` |
| Supplementary Information 검토 | 완료 | decoder, ZXXZ, latency 관련 보완 내용 |
| 2026 저자 정정 반영 | 완료 | Fig. 3a 축·범례 정정 기록 |
| 프로젝트 범위와 실험 설계 | 완료 | `team_project_plan.md` |
| Stim/PyMatching 실행 경로 확인 | 완료 | `week1_progress.md` |
| 3조건 실험 runner | 완료 | `run_experiment.py` |
| Baseline threshold sweep | 완료 | 54개 결과 행과 baseline figure |
| Correlated-ZZ calibration | 완료 | `calibration_grid.csv` |
| Detection-probability matching | 완료 | 모든 최종 조건 $|\Delta p_{det}|<0.0005$ |
| Correlated distance scaling | 완료 | 18개 raw result 행과 correlated figure |
| Correlation-strength sensitivity | 완료 | $d=3$, 4개 `zz_factor`, X/Z 각 50,000 shots |
| Bootstrap·다중비교 보정 | 완료 | threshold/$\Lambda$ interval, Holm-adjusted paired test |
| 공개 Willow 데이터 재현 계획 | 완료 | `willow_data_reproduction_plan.md` |
| 105Q archive·README 검증 | 완료 | 5.7GB archive 크기·공식 MD5 일치 |
| 단일 configuration 재디코딩 pilot | 완료 | `d3_at_q4_5/X/r01`, direct/provided 모두 306/50,000 failures |
| 실제-data $d=3,5,7$ $\Lambda$ 재현 | 완료 | Public Libra 2.029, direct PyMatching+SI1000 1.525 |
| Results/Discussion 초안 | 완료 | `results_discussion_draft.md` |
| 내용 기준본 작성 | 완료 | `content_master.md`, 논문·decoder·simulation·가설 구분 |
| 포스터 내용 원고 | 1차 완료 | 최신 통계와 포함·삭제 우선순위를 반영한 `poster_content_draft.md` |
| 실제 포스터 파일 제작 | 대기 | 학교/학회 template, 규격과 팀 정보 확보 후 진행 |
| 최종 보고서 통합 | 완료 | `final_report_draft.md` |
| 발표 자료·리허설 | 대기 | 발표 대본과 예상 질문 정리 필요 |

역할분담은 현재 진행의 제약으로 사용하지 않고 필요한 작업을 순서대로 수행하고 있다.

## 2. 확정된 프로젝트 범위

### 수행하는 내용

- 논문의 below-threshold scaling, correlated error와 decoder 관련 주장 검토
- Stim rotated surface-code memory simulation
- Code distance $d=3,5,7$
- X/Z memory 별도 평가
- PyMatching MWPM decoding
- Cycle당 logical error rate $\epsilon_d$와 suppression factor $\Lambda$ 계산
- Spatial correlated ZZ surrogate와 decoder-prior mismatch 분석
- Zenodo 공개 Willow $d=3,5,7$ syndrome의 matching-decoder 재분석

### 수행하지 않는 내용

- Willow hardware 또는 calibrated device noise의 정확한 재현
- 실제 ZXXZ gate schedule 구현
- Willow의 neural-network decoder 또는 AlphaQubit 재현
- Leakage/DQLR의 완전한 물리 모델
- Sparse Blossom streaming stack과 quantum feedback 구현
- 논문의 $\Lambda$ 또는 logical error 수치에 대한 직접 재현 주장
- PyMatching을 이용한 neural-network decoder $\Lambda=2.14$의 동일 재현 주장

## 3. 구현 현황

### `run_experiment.py`

한 basis에서 다음 세 조건을 함께 실행한다.

1. Stage B independent data + independent DEM
2. ZZ-only Stage C data + independent DEM
3. 같은 Stage C sample + correlated-ZZ DEM

구현된 기능:

- X/Z 및 `rounds=distance`
- Adaptive 또는 fixed-shot sampling
- `--independent-p`를 이용한 detection-probability matching
- Shot logical failure probability $p_L$
- Cycle당 logical error rate $\epsilon$
- Wilson 95% interval
- Mean detector-event fraction $p_{det}$
- 같은 sample에서 두 decoder가 엇갈린 failure 수 기록
- 기존 CSV overwrite 방지
- basis와 단독/동시 실행 방식에 무관한 seed 재현성

### 분석 script

- `plot_baseline.py`: baseline $\epsilon(p)$, $\Lambda(p)$와 threshold crossing 생성
- `plot_correlated.py`: correlated scaling, paired exact test와 $\Lambda$ 생성
- `analyze_statistics.py`: threshold/$\Lambda$ bootstrap, Holm correction과 sensitivity 요약
- `analyze_willow_data.py`: 공개 `.b8` observable prediction의 logical failure 평가

추가 dependency는 설치하지 않았으며 기존 NumPy, Matplotlib, Stim, PyMatching을 사용한다.

## 4. Baseline 결과

### 실험 조건

- Noise: Stage B local independent model
- $p=\{0.0015,0.003,0.006,0.007,0.008,0.009,0.01,0.015,0.02\}$
- $d=3,5,7$
- X/Z memory
- 각 점 20,000 fixed shots
- 총 54개 결과 행

### Threshold와 suppression factor

| 항목 | X memory | Z memory |
|---|---:|---:|
| Pilot threshold crossing | 0.008852 | 0.008089 |
| Bootstrap 95% interval | 0.008581–0.009188 | 0.007793–0.008526 |
| $p=0.003$에서 $\Lambda$ | 2.551 | 2.364 |
| $p=0.006$에서 $\Lambda$ | 1.300 | 1.225 |
| $p=0.009$에서 $\Lambda$ | 0.990 | 0.963 |

낮은 $p$에서는 $\epsilon_7<\epsilon_5<\epsilon_3$이고 높은 $p$에서는 순서가 뒤집혔다. 따라서 단순화된 모델에서 below-threshold scaling과 threshold crossing을 정성적으로 확인했다.

Threshold 수치는 Stim noise instruction의 합성 parameter에 대한 값이며 Willow hardware threshold가 아니다.

## 5. Correlated-ZZ 결과

### 최종 실험 조건

- Stage C $p=0.003$
- `zz_factor=1.0`
- Swap-like component 비활성화
- $d=3,5,7$, X/Z memory
- 각 조건 50,000 fixed shots
- Stage B의 $p$를 basis와 distance별로 보정
- 최종 최대 $|\Delta p_{det}|=0.000377$

`zz_factor=1.0`은 decoder-prior 효과를 관찰하기 위한 강한 stress-test 값이며 실제 장치의 calibrated correlation strength가 아니다.

### Suppression factor

| Basis | Independent | Correlated + mismatched prior | Correlated + matched prior |
|---|---:|---:|---:|
| X | 1.880 | 1.374 | 1.400 |
| Z | 1.831 | 2.202 | 2.202 |

Bootstrap 95% interval은 X의 independent, mismatched, matched에서 각각 1.805–1.958, 1.333–1.417, 1.357–1.444였고, Z에서는 1.754–1.913, 2.090–2.324, 2.090–2.324였다.

### Paired decoder 결과

| Basis | Distance | Mismatched only | Matched only | Net gain | Holm-adjusted $p$ |
|---|---:|---:|---:|---:|---:|
| X | 3 | 128 | 82 | 46 | 0.00733 |
| X | 5 | 368 | 249 | 119 | $9.46\times10^{-6}$ |
| X | 7 | 439 | 303 | 136 | $4.03\times10^{-6}$ |
| Z | 3 | 1 | 1 | 0 | 1.0 |
| Z | 5 | 10 | 10 | 0 | 1.0 |
| Z | 7 | 5 | 5 | 0 | 1.0 |

### Correlation-strength sensitivity

$d=3$에서 `zz_factor=0.05,0.25,0.5,1.0`을 비교한 결과 X의 net gain은 2, 15, 19, 45 shots였고, 8개 X/Z-factor 비교에 Holm correction을 적용하면 `zz_factor=1.0`만 유의했다($p_{adj}=0.0181$). Z의 net gain은 모두 0이었다.

### 핵심 해석

- X memory에서는 correlated ZZ가 logical-error suppression을 약화시켰다.
- Correlation-aware DEM은 X memory의 오류를 작지만 일관되게 줄였다.
- Z memory에서는 matched/mismatched decoder가 같은 성능을 보였다.
- 같은 $p_{det}$에서도 logical 성능이 달라져 평균 detector-event 비율만으로 error damage를 설명할 수 없었다.
- X/Z를 평균하면 correlated noise의 방향성을 가릴 수 있으므로 basis별 결과를 유지해야 한다.

## 6. 원 논문과의 관계

원 논문은 Willow hardware와 neural-network decoder에서 $\Lambda=2.14\pm0.02$를 보고했다. 본 simulation도 낮은 $p$에서 $\Lambda>2$를 보였지만, noise model, gate schedule, decoder와 통계 절차가 다르므로 수치 재현으로 해석하지 않는다.

본 결과가 정성적으로 지지하는 메시지는 다음 두 가지다.

1. Threshold 아래에서는 code distance를 늘리면 logical error가 감소한다.
2. Detection probability가 비슷해도 correlation structure와 decoder model에 따라 logical performance가 달라진다.

### 공개 Willow 데이터의 직접 검증

공개 105Q archive의 main sweep 364 configurations을 논문의 fitting 절차로 다시 집계했다. Public Libra RL prediction에서는 $\epsilon_{3,5,7}=(0.006802,0.003372,0.001652)$와 $\Lambda=2.029$를 얻어, 논문 Libra ensemble 101의 2.04에 가까운 scaling을 확인했다. 같은 syndrome을 noisy SI1000 circuit에서 만든 DEM과 PyMatching으로 직접 decode한 결과는 $\epsilon_{3,5,7}=(0.010438,0.007102,0.004485)$, $\Lambda=1.525$였다.

따라서 본 프로젝트는 이제 synthetic 정성 검증뿐 아니라 실제 Willow syndrome에서 below-threshold scaling을 재계산하고 decoder 경로의 영향을 비교한다. 다만 공개 archive에는 neural decoder prediction이 없으므로 논문의 neural $\Lambda=2.14$ 또는 AlphaQubit을 직접 재현했다는 뜻은 아니다.

## 7. 초기 목표·가설과 결과의 일치도

초기 계획의 목표와 기대 결과를 변경하지 않고 현재 결과와 대조했다.

| 초기 목표 또는 가설 | 판정 | 근거 |
|---|---|---|
| Willow의 수치 재현이 아니라 단순화된 모델에서 핵심 현상을 정성적으로 검증한다. | 일치 | Stim rotated memory와 PyMatching만 사용했고 모든 문서에서 Willow 재현이 아님을 명시함 |
| 낮은 $p$에서 $\epsilon_7<\epsilon_5<\epsilon_3$가 나타난다. | 강하게 지지 | X/Z 모두 낮은 $p$에서 거리 증가에 따라 $\epsilon$ 감소 |
| 높은 $p$에서는 큰 code distance의 이점이 사라지고 threshold crossing이 나타난다. | 강하게 지지 | X 약 0.008852, Z 약 0.008089에서 fitted $\Lambda=1$ crossing |
| $\Lambda>1$인 영역이 존재하고 $p$가 낮을수록 $\Lambda$가 커진다. | 강하게 지지 | X: 0.003에서 2.551, 0.006에서 1.300, 0.009에서 0.990; Z도 같은 경향 |
| Detection probability가 비슷해도 correlated noise의 logical error가 더 높을 수 있다. | 부분 지지·basis 의존 | X에서는 correlated condition이 더 나빴지만 Z에서는 오히려 $\epsilon$과 scaling이 더 좋았음 |
| Correlation-aware decoder prior가 mismatched prior보다 낫다. | 부분 지지·basis 의존 | X에서는 세 distance 모두 paired improvement, Z에서는 차이 없음 |
| Noise correlation 자체의 영향과 decoder-prior mismatch를 분리한다. | 일치 | Independent/correlated sample 비교와 동일 correlated sample의 두 decoder 비교를 분리함 |
| X/Z basis를 모두 실행하고 구분해 해석한다. | 일치 | basis별 threshold, $\Lambda$, paired 결과를 각각 보고함 |
| Poster에 사용할 threshold와 extension graph를 만든다. | 대부분 일치 | 세 300 dpi figure와 A1 3단 원고·배치안 완료, 실제 template 파일 제작은 남음 |
| Offline throughput을 측정한다. | 미평가 | 초기 계획에서도 선택 항목이었고 핵심 실험에 집중하기 위해 수행하지 않음 |

### 종합 평가

- **핵심 목표인 below-threshold scaling 정성적 검증은 계획과 매우 잘 일치한다.**
- **Correlated-noise 확장 목표도 수행했지만, 결과는 최초의 단순한 예상보다 basis 의존적이었다.**
- X memory만 보면 “correlation이 suppression을 약화시키고 matched prior가 일부 회복한다”는 예상과 일치한다.
- Z memory 결과는 “correlated noise는 같은 $p_{det}$에서 항상 더 나쁘다”는 일반화를 지지하지 않는다.
- 따라서 실험 실패가 아니라 초기 가설을 더 정확한 형태로 수정하게 된 결과로 해석한다.

### 결과를 반영한 수정 가설

> **평균 detection probability가 같더라도 logical error suppression은 noise correlation의 연산자 방향, logical basis와 decoder prior의 적합성에 따라 달라진다. Correlation-aware prior의 이점도 이러한 correlation이 decoder 판정을 실제로 바꾸는 basis에서 나타난다.**

이 수정 가설은 X/Z의 서로 다른 결과와 paired decoder comparison을 모두 설명하며, 원 논문의 “평균 error 지표뿐 아니라 오류 구조와 decoder가 중요하다”는 메시지에도 더 정확히 연결된다.

## 8. 핵심 산출물

### 문서

| 파일 | 내용 |
|---|---|
| `paper_summary.md` | 논문 본문, SI, 저자 정정 통합 요약 |
| `week1_progress.md` | 공식 자료와 실행 환경 확인 |
| `week2_progress.md` | runner, calibration과 baseline pilot |
| `week3_progress.md` | correlated scaling 상세 기록 |
| `results_discussion_draft.md` | 보고서용 결과·논의·caption 초안 |
| `content_master.md` | 논문 사실, decoder 구분, simulation 결과와 최종 주장 기준본 |
| `final_report_draft.md` | 논문 리뷰·방법·결과·논의·한계 통합 보고서 초안 |
| `willow_data_reproduction_plan.md` | 공개 105Q 데이터 재디코딩 단계와 완료 기준 |
| `willow_data_reproduction_progress.md` | README/schema 및 pilot 검증 결과 |
| `project_status.md` | 현재 통합 현황 |

### 최종 분석 결과

| 경로 | 내용 |
|---|---|
| `artifacts/qec_below_threshold_review/baseline_distance_pilot.csv` | Baseline 54개 결과 행 |
| `artifacts/qec_below_threshold_review/baseline_lambda_pilot.csv` | Baseline $\Lambda$ |
| `artifacts/qec_below_threshold_review/baseline_distance_pilot.png` | Baseline 원본 figure |
| `artifacts/qec_below_threshold_review/baseline_distance_pilot_poster.png` | A1용 글자·선 크기를 보강한 Baseline figure |
| `artifacts/qec_below_threshold_review/correlated_scaling.csv` | Correlated raw 18개 행 |
| `artifacts/qec_below_threshold_review/correlated_scaling_summary.csv` | Correlated 비교 6개 행 |
| `artifacts/qec_below_threshold_review/correlated_scaling.png` | Correlated 원본 figure |
| `artifacts/qec_below_threshold_review/correlated_scaling_poster.png` | A1용 범례·선 크기를 보강한 Correlated figure |
| `artifacts/qec_below_threshold_review/sensitivity_d3.csv` | Correlation-strength sensitivity 통합 결과 |
| `artifacts/qec_below_threshold_review/statistics/` | Bootstrap, Holm correction과 통계 요약 |
| `artifacts/qec_below_threshold_review/willow_data/*/` | decoder별 364-configuration 집계와 $\epsilon_d$, $\Lambda$ |
| `artifacts/qec_below_threshold_review/willow_reproduction_comparison.csv` | 논문 기준과 공개 데이터 재현 비교 |
| `artifacts/qec_below_threshold_review/willow_reproduction_comparison.png` | Willow decoder 비교 원본 figure |
| `artifacts/qec_below_threshold_review/willow_reproduction_comparison_poster.png` | A1용 글자·선 크기를 보강한 Willow 비교 figure |

`calibration_*.csv`, `corr_calibration_*.csv`, `matched_pdet_*.csv`와 개별 distance CSV는 보정·추적용 중간 산출물이며 최종 figure 입력은 위 통합 CSV다.

## 9. 검증 현황

- Stim 1.15.0
- PyMatching 2.3.1
- NumPy 2.2.6
- Matplotlib 3.10.8
- 여섯 Python script `py_compile` 통과
- `run_experiment.py --self-test` 통과
- `plot_baseline.py --self-test` 통과
- `plot_correlated.py --self-test` 통과
- `analyze_statistics.py --self-test` 통과
- `analyze_willow_data.py --self-test` 통과
- `plot_willow_reproduction.py --self-test` 통과
- Willow 105Q archive 크기와 MD5 공식값 일치
- Willow pilot `stim m2d` detection/observable SHA-256 일치
- Willow pilot direct/provided prediction SHA-256 일치
- 제공 decoder 5개 경로와 direct PyMatching 경로 각각 364 configurations 집계
- Baseline CSV 54행 확인
- Correlated raw CSV 18행, summary CSV 6행 확인
- 세 300 dpi PNG를 직접 확인

## 10. 해석상의 한계

1. Strong ZZ surrogate를 사용했으므로 실제 장치의 효과 크기로 해석할 수 없다.
2. Detection matching을 위해 independent $p$를 distance별로 변경했으므로 correlated figure의 independent $\Lambda$는 엄밀한 fixed-$p$ scaling이 아니다.
3. `decompose_errors=True`는 correlated error를 graphlike component로 근사한다.
4. PyMatching은 Willow 논문의 neural-network decoder나 AlphaQubit 구현이 아니다.
5. Leakage, rare temporal burst, drift와 실시간 feedback은 포함하지 않았다.
6. Parametric bootstrap은 shot noise만 반영하며 calibration, model, seed와 finite-distance uncertainty는 포함하지 않는다.
7. Sensitivity test는 $d=3$에 한정된다.
8. Direct Willow 경로의 $p_L\ge0.5$ late-round point 4개는 log-fidelity fit에서 제외했으며 raw CSV에는 보존했다.

## 11. 남은 작업 우선순위

1. 학교·학회 포스터 규격과 팀 정보 확정
2. 확정된 내용으로 실제 포스터 파일 제작
3. 발표 예상 질문과 답변 정리

105Q surface-code 공개 데이터 재현과 synthetic simulation은 완료됐다. 72Q·repetition-code 대용량 archive 확장은 현재 리뷰 범위에 필요하지 않아 진행하지 않는다.
