# 3주차 진행 기록: Correlated-ZZ distance scaling

## 1. 실험 질문

평균 detection probability가 비슷한 조건에서도 spatially correlated ZZ noise가 distance scaling과 decoder 성능을 바꾸는지 확인한다.

이 실험은 실제 Willow noise를 재현하는 것이 아니라 상관 구조의 영향을 분리해 보는 정성적 stress test다.

## 2. 조건

- Stage C physical parameter: $p=0.003$
- `zz_factor=1.0`, swap-like component 비활성화
- Distance: $d=3,5,7$
- Rounds: `rounds=distance`
- Basis: X, Z 별도
- Shots: 각 조건 50,000 fixed shots
- Decoder: PyMatching MWPM
- Correlated DEM: `decompose_errors=True`
- Detection-probability tolerance: $|\Delta p_{det}|<0.0005$

`zz_factor=1.0`은 calibration grid에서 decoder-prior 차이를 안정적으로 관찰하기 위해 선택한 강한 stress-test 값이다. 실제 장치의 calibrated correlation strength로 해석하지 않는다. 실제 평균 $p_{corr}$은 distance에 따라 약 0.00299–0.00313이었다.

## 3. Detection-probability matching

Stage C는 $p=0.003$으로 고정하고, Stage B의 `independent_p`를 basis와 distance별로 조정했다.

| Basis | Distance | Stage B $p$ | $p_{det}^{ind}$ | $p_{det}^{corr}$ | $\Delta p_{det}$ |
|---|---:|---:|---:|---:|---:|
| X | 3 | 0.003449 | 0.079388 | 0.079513 | +0.000125 |
| X | 5 | 0.003500 | 0.090083 | 0.090460 | +0.000377 |
| X | 7 | 0.003575 | 0.093552 | 0.093758 | +0.000206 |
| Z | 3 | 0.003275 | 0.077324 | 0.077356 | +0.000032 |
| Z | 5 | 0.003421 | 0.088049 | 0.088050 | +0.000001 |
| Z | 7 | 0.003477 | 0.091643 | 0.091784 | +0.000141 |

모든 조건이 사전에 정한 tolerance를 만족했다.

## 4. Logical error per cycle

| Basis | Distance | Independent | Corr. mismatched | Corr. matched |
|---|---:|---:|---:|---:|
| X | 3 | 0.010144 | 0.013445 | 0.013121 |
| X | 5 | 0.004733 | 0.009732 | 0.009218 |
| X | 7 | 0.002869 | 0.007117 | 0.006694 |
| Z | 3 | 0.008669 | 0.007073 | 0.007073 |
| Z | 5 | 0.004459 | 0.003082 | 0.003082 |
| Z | 7 | 0.002585 | 0.001458 | 0.001458 |

## 5. Error suppression factor

$\ln\epsilon_d$를 $d$에 선형 적합해 $\Lambda$를 계산했다.

| Basis | Independent | Corr. mismatched | Corr. matched |
|---|---:|---:|---:|
| X | 1.880 | 1.374 | 1.400 |
| Z | 1.831 | 2.202 | 2.202 |

X memory에서는 correlated ZZ가 error suppression을 크게 약화시켰다. Correlation-aware DEM은 mismatched prior보다 결과를 개선했지만 independent condition의 scaling을 회복하지는 못했다.

Z memory에서는 correlated 조건의 $\Lambda$가 더 크게 측정됐고 두 decoder prior는 같은 성능을 냈다. 이는 ZZ error가 logical basis에 따라 비대칭적으로 작용하며, 단순한 평균 detection probability만으로 logical damage를 결정할 수 없다는 점을 보여준다. 이를 correlated noise가 일반적으로 성능을 개선한다는 뜻으로 해석하면 안 된다.

## 6. Paired decoder comparison

두 correlated decoder는 완전히 같은 syndrome sample을 해독했다. `mismatched only`는 mismatched decoder만 실패한 shot, `matched only`는 matched decoder만 실패한 shot이다.

| Basis | Distance | Mismatched only | Matched only | Net gain | Exact paired $p$ |
|---|---:|---:|---:|---:|---:|
| X | 3 | 128 | 82 | 46 | 0.00183 |
| X | 5 | 368 | 249 | 119 | $1.89\times10^{-6}$ |
| X | 7 | 439 | 303 | 136 | $6.72\times10^{-7}$ |
| Z | 3 | 1 | 1 | 0 | 1.0 |
| Z | 5 | 10 | 10 | 0 | 1.0 |
| Z | 7 | 5 | 5 | 0 | 1.0 |

X basis에서는 matched prior의 개선 방향이 세 distance에서 일관됐다. 위 값은 discordant shot에 대한 양측 exact binomial test의 raw 결과다. 후속 통계 분석에서 여섯 개 X/Z-distance 비교에 Holm correction을 적용한 뒤 X의 adjusted $p$는 각각 0.00733, $9.46\times10^{-6}$, $4.03\times10^{-6}$였고 Z는 모두 1.0이었다.

## 7. 산출물

- `artifacts/qec_below_threshold_review/correlated_scaling.csv`: 18개 raw result row
- `artifacts/qec_below_threshold_review/correlated_scaling_summary.csv`: 6개 비교 summary row
- `artifacts/qec_below_threshold_review/correlated_scaling.png`: 300 dpi 포스터용 figure
- `qec_below_threshold_review/plot_correlated.py`: summary/figure 재생성 script

## 8. 해석상의 한계

1. `zz_factor=1.0`은 의도적으로 강한 stress test이며 장치 보정값이 아니다.
2. Stage B의 $p$를 distance별로 조금씩 바꿨으므로 independent $\Lambda$는 엄밀한 fixed-$p$ scaling이 아니다. 별도의 fixed-$p$ baseline 결과를 우선 기준으로 사용해야 한다.
3. `decompose_errors=True`는 correlated fault를 graphlike component로 근사한다.
4. PyMatching은 Willow 논문의 neural-network decoder나 AlphaQubit을 재현하지 않는다.
5. 이 결과는 synthetic surrogate noise에 한정된다.

## 9. 다음 작업

1. 논문 결과와 baseline/correlated simulation을 연결하는 Results·Discussion 초안 작성
2. 두 figure와 핵심 수치를 포스터용 문구로 축약
3. figure caption과 재현 명령 정리
