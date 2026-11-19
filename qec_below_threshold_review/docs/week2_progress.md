# 2주차 진행 기록: 통제 실험 runner와 첫 pilot

## 1. 완료한 구현

`qec_below_threshold_review/run_experiment.py`를 추가했다. 새 회로·decoder 구현은 만들지 않고 저장소의 기존 `circuits.py`, `config.py`, `noise_willowcore.py`, `decoders/baseline_pymatching.py`를 재사용한다.

한 basis마다 다음 세 조건을 CSV 한 행씩 기록한다.

| 조건 | Sample noise | MWPM prior |
|---|---|---|
| `independent_matched` | Stage B local independent noise | Stage B DEM |
| `correlated_mismatched` | ZZ-only Stage C | Stage B DEM |
| `correlated_matched` | 동일한 ZZ-only Stage C sample | Stage C DEM |

두 correlated 조건은 같은 detector/observable sample을 해독한다. 따라서 둘의 차이는 decoder prior에서만 발생한다. Stage C는 `enable_swap_component=False`와 `swap_factor=0`으로 고정했다.

## 2. Sampling과 CSV 규칙

- 기본 batch: 2,000 shots
- 기본 중단: 각 비교 decoder가 logical failure 100개에 도달하거나 20,000 shots 도달
- `p_L`: shot logical failure probability
- `epsilon`: $0.5[1-(1-2p_L)^{1/r}]$
- `p_L` 불확실성: Wilson 95% interval
- `p_det`: 전체 detector bit 중 1의 비율
- `p_corr`: 실제 edge별 ZZ 확률의 평균
- 기존 CSV는 `--overwrite` 없이는 덮어쓰지 않음

Wilson interval은 adaptive stopping을 엄밀히 보정한 sequential confidence interval이 아니다. 본 실험에서 95% coverage 자체가 중요한 경우 고정 shot 수를 사용하거나 별도의 sequential interval을 도입해야 한다.

## 3. $d=3$, $p=0.003$ 첫 통제 pilot

실행 결과: `artifacts/qec_below_threshold_review/pilot_d3_p003.csv`

| Basis | 조건 | Shots | Failures | $p_L$ | $\epsilon$ | $p_{det}$ |
|---|---|---:|---:|---:|---:|---:|
| X | independent matched | 4,000 | 104 | 0.0260 | 0.008821 | 0.07105 |
| X | correlated mismatched | 6,000 | 119 | 0.01983 | 0.006701 | 0.07075 |
| X | correlated matched | 6,000 | 119 | 0.01983 | 0.006701 | 0.07075 |
| Z | independent matched | 6,000 | 145 | 0.02417 | 0.008189 | 0.07292 |
| Z | correlated mismatched | 6,000 | 142 | 0.02367 | 0.008017 | 0.07171 |
| Z | correlated matched | 6,000 | 142 | 0.02367 | 0.008017 | 0.07171 |

이 결과는 pilot이며 독립 조건과 correlated 조건의 `p_det`가 아직 통계적으로 matching되었다고 판정하지 않았다. 기본 `zz_factor=0.05`에서 matched/mismatched 결과가 같은 것은 이 표본에서 prior 차이가 MWPM 판정을 바꾸지 못했다는 뜻이지, 일반적인 동등성을 뜻하지 않는다.

## 4. 상관 세기 민감도 점검

runner 경로가 실제로 서로 다른 prior를 적용하는지 확인하기 위해 `zz_factor=2.0`으로 짧은 진단 실행을 했다. 이 값은 최종 물리 모델이 아니라 기능 확인용이다.

결과: `artifacts/qec_below_threshold_review/diagnostic_d3_zz2.csv`

- X basis, 2,000 shots: mismatched 118 failures, matched 114 failures
- Z basis, 2,000 shots: 두 prior 모두 38 failures

X basis에서 decoder 결과가 갈라졌으므로 두 matching graph가 실제로 구분되어 적용된다. 차이의 통계적 유의성은 이 진단 실행으로 주장하지 않는다.

## 5. 검증

```powershell
python -m py_compile qec_below_threshold_review/run_experiment.py
python qec_below_threshold_review/run_experiment.py --self-test
```

자체 점검은 Wilson interval 경계, 세 조건 생성, correlated sample 공유, ZZ 확률의 양수 여부를 확인한다.

## 6. 다음 작업

## 6. Calibration grid

$d=3$, X/Z, 4,000 fixed shots에서 다음 grid를 실행했다.

- $p \in \{0.0015, 0.003, 0.006\}$
- `zz_factor` $\in \{0.05, 0.25, 0.5, 1.0\}$
- 통합 결과: `artifacts/qec_below_threshold_review/calibration_grid.csv`

관찰 결과는 다음과 같다.

- `zz_factor`가 증가할수록 correlated data의 `p_det`가 대체로 증가했다.
- X basis에서는 prior에 따른 failure 수 차이가 나타났다. 예를 들어 $p=0.006$, `zz_factor=1.0`에서 mismatched가 matched보다 23 failures 많았다.
- Z basis에서는 이 grid의 모든 점에서 두 prior의 failure 수가 같았다. ZZ error가 basis에 따라 다르게 작용하는 예상 가능한 비대칭이므로 X/Z를 합쳐 보고하면 안 된다.
- 4,000-shot grid는 parameter 선정용이며 유의성 검정 결과가 아니다.

## 7. Detection-probability matching trial

runner에 `--independent-p`를 추가하여 Stage B의 $p$를 Stage C와 별도로 조절할 수 있게 했다. $p=0.003$, `zz_factor=1.0`인 Stage C에 대해 basis별 보간 후 각 20,000 fixed shots를 실행했다.

| Basis | Stage B $p$ | Stage B $p_{det}$ | Stage C $p_{det}$ | 절대 차이 |
|---|---:|---:|---:|---:|
| X | 0.00343 | 0.079481 | 0.079902 | 0.000421 |
| Z | 0.00325 | 0.077360 | 0.076898 | 0.000463 |

결과 파일은 `matched_pdet_x.csv`와 `matched_pdet_z.csv`이다. 임시 tolerance $|\Delta p_{det}|<0.0005$를 만족했다.

X basis에서는 detection probability가 비슷해도 Stage B $p_L=0.0291$, correlated mismatched $p_L=0.04235$, correlated matched $p_L=0.04180$이었다. 다만 matched/mismatched 차이는 각 95% interval보다 작으므로 이 20,000-shot 결과만으로 decoder 개선을 주장하지 않는다. Z basis에서는 matched/mismatched가 각각 0.02075와 0.02085로 사실상 같았다.

단독 Z 실행과 `--basis both`의 Z 실행이 같은 seed를 쓰지 않던 문제도 발견해 수정했다. 이제 X는 입력 seed, Z는 입력 seed+10,000을 사용하므로 실행 방식과 무관하게 재현된다.

## 8. 다음 작업

## 8. Baseline distance-scaling pilot

Stage B independent noise에서 $d=3,5,7$, X/Z, 각 20,000 fixed shots를 실행했다.

- $p \in \{0.0015,0.003,0.006,0.007,0.008,0.009,0.01,0.015,0.02\}$
- 통합 결과: `artifacts/qec_below_threshold_review/baseline_distance_pilot.csv`
- 총 54행: 9개 $p$ × 3개 distance × 2개 basis

$\ln\epsilon_d$를 distance에 선형 적합해 얻은 pilot $\Lambda$는 다음과 같다.

| $p$ | X $\Lambda$ | Z $\Lambda$ |
|---:|---:|---:|
| 0.003 | 2.551 | 2.364 |
| 0.006 | 1.300 | 1.225 |
| 0.007 | 1.129 | 1.093 |
| 0.008 | 1.056 | 1.004 |
| 0.009 | 0.990 | 0.963 |
| 0.010 | 0.931 | 0.915 |

따라서 이 surrogate model과 MWPM 조합의 threshold crossing은 대략 X basis $p\approx0.0088$, Z basis $p\approx0.0081$ 부근이다. 이는 두 인접 pilot 점의 $\Lambda=1$ 선형 보간값이며 confidence interval을 포함한 정밀 threshold 추정치는 아니다.

낮은 $p$에서는 $\epsilon_7<\epsilon_5<\epsilon_3$이고, 높은 $p$에서는 순서가 뒤집혀 below-threshold scaling의 정성적 재현에 성공했다. 서로 다른 distance에서 `rounds=distance`를 사용했으므로 shot failure probability $p_L$가 아니라 cycle당 $\epsilon$을 비교해야 한다.

## 9. Baseline figure

`plot_baseline.py`를 추가하여 다음 산출물을 생성했다.

- `artifacts/qec_below_threshold_review/baseline_distance_pilot.png`: X/Z $\epsilon(p)$와 fitted $\Lambda(p)$
- `artifacts/qec_below_threshold_review/baseline_lambda_pilot.csv`: fitted 및 pairwise $\Lambda$

그림은 300 dpi PNG이며 error bar는 $p_L$의 Wilson interval을 $\epsilon$으로 변환한 값이다. 세 distance의 $\ln\epsilon_d$ 선형 적합으로 얻은 crossing은 X 0.008852, Z 0.008089이다.

## 10. 다음 작업

1. calibration 결과에서 최종 `zz_factor` 또는 sensitivity 범위 결정
2. threshold 아래의 선택된 $p$에서 correlated distance scaling 실행
3. 논문 결과와 simulation 결과의 차이 및 한계 문서화
