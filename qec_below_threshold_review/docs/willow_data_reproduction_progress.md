# Willow 공개 데이터 재디코딩 진행 기록

> 기준일: 2026-08-24\
> 상태: **105Q archive 검증, 제공 decoder 집계, 전체 PyMatching 직접 재디코딩 완료**

## 0. 먼저 보는 결론

- 5.7GB 파일에는 이번 프로젝트에 필요한 $d=3,5,7$ surface-code 데이터가 모두 들어 있었다.
- 공개 Libra 결과를 다시 계산해 $\Lambda=2.029$를 얻었다. 논문의 matching-decoder 값 2.04와 가깝다.
- 같은 오류 측정값을 PyMatching으로 직접 처리하면 $\Lambda=1.525$였다.
- 두 값 모두 1보다 크므로 코드를 키울수록 오류가 감소했다. 다만 디코더에 따라 감소 폭이 달랐다.

### 이 문서의 용어

| 용어 | 쉬운 뜻 |
|---|---|
| Archive | 여러 데이터 파일을 하나로 묶은 ZIP 파일 |
| Patch | 칩에서 surface code를 배치한 위치 |
| Configuration | patch, X/Z 종류와 cycle 수가 정해진 실험 조건 하나 |
| Syndrome / detection event | 오류가 발생했음을 알려 주는 측정 신호 |
| Observable flip | 최종 논리값이 뒤집혔는지를 나타내는 값 |
| DEM | 오류 신호 사이의 관계와 발생 가능성을 적은 디코더용 설명서 |
| Prediction | 디코더가 예측한 최종 논리값 |
| Fitting | 여러 측정점을 가장 잘 설명하는 직선이나 곡선을 찾는 계산 |

## 1. 데이터 출처와 범위

- Source: Google Quantum AI, Zenodo DOI `10.5281/zenodo.13273331`
- 1차 archive: `google_105Q_surface_code_d3_d5_d7.zip`
- 공식 크기: 5,716,907,033 bytes
- 공식 MD5: `21fa6ad35b395d838ebcdbc92e364a12`
- ZIP 내부 파일: 9,959개
- 디코딩할 전체 실험 조건: 420개
- 칩에서 사용한 위치: $d=3$ 9곳, $d=5$ 4곳, $d=7$ 1곳
- 보호한 논리값의 종류: X/Z 각각 210개 조건
- Local archive MD5 검증: 공식값과 일치
- 72Q surface-code와 65GB repetition-code archive는 받지 않음

먼저 ZIP의 일부만 읽어 README와 작은 시험 데이터를 확인했다. 이후 5.7GB ZIP 전체를 내려받아 파일 크기와 MD5 지문이 공식값과 같은지 확인했다. 저장공간을 아끼기 위해 ZIP 전체를 풀지 않고 필요한 파일을 ZIP 안에서 바로 읽었다.

## 2. README에서 확인한 구조

각 실험 조건은 칩의 위치, X/Z 종류와 QEC cycle 수로 구분된다. 폴더에는 다음 자료가 들어 있다.

```text
patch/basis/rounds/
├── circuit_ideal.stim
├── circuit_noisy_si1000.stim
├── measurements.b8
├── sweep_bits.b8
├── detection_events.b8
├── obs_flips_actual.b8
├── metadata.json
└── decoding_results/<pathway>/
    ├── error_model.dem
    └── obs_flips_predicted.b8
```

`.b8`은 0과 1을 압축해서 저장한 이진 파일이다. 실제 최종값과 디코더가 예측한 값을 비교해 서로 다르면 그 반복 실험을 논리 오류로 계산했다.

README에 적힌 직접 디코딩 방법은 다음과 같다.

1. Stim으로 회로에서 디코더용 오류 설명서(DEM)를 만든다.
2. PyMatching이 오류 측정 신호를 읽고 최종 논리값을 예측한다.
3. 예측값과 실제값을 비교해 실패 횟수를 센다.

Cycle 수는 `1, 10, 13, 30, 50, ..., 250`이다. 13-cycle 자료는 디코더의 오류 가정을 조정하는 데 사용된 특별 자료이므로 최종 계산에서 제외했다. 주요 분석에는 10-cycle과 30–250-cycle 자료를 사용했다.

공개 ZIP에는 디코더별 예측값도 들어 있다. Correlated matching과 Harmony는 오류 가정별로 각각 420개 조건, Libra는 주요 분석용 364개 조건의 예측값을 제공한다.

논문의 계산 방법도 그대로 따라갔다. 먼저 cycle 수가 늘어날 때 실패율이 얼마나 빠르게 증가하는지 직선으로 맞춰 distance별 $\epsilon_d$를 구했다. 그다음 $d=3,5,7$의 $\epsilon_d$ 감소 속도로 $\Lambda$를 계산했다. 논문은 장치 성능이 시간에 따라 조금씩 변하는 현상까지 오차 범위에 넣었지만, 본 분석의 간이 오차 범위는 그 효과를 완전히 포함하지 않는다.

## 3. 작은 데이터로 먼저 확인한 시험

| 항목 | 값 |
|---|---|
| 칩의 위치 | `d3_at_q4_5` |
| 보호한 논리값 | X |
| QEC cycle | 1 |
| 반복 횟수 | 50,000 |
| 정보를 저장하는 큐비트 | 9 |
| 오류를 측정하는 큐비트 | 8 |

Artifact 경로:

```text
artifacts/qec_below_threshold_review/willow_data/pilot/d3_at_q4_5/X/r01/
```

## 4. 데이터 파일을 올바르게 읽었는지 확인

README의 예제를 실행해 원시 측정값에서 오류 검출 신호와 실제 최종 논리값을 다시 만들었다.

| 비교 | 결과 |
|---|---|
| Recomputed `detection_events.b8` 대 공개 파일 | SHA-256 완전 일치 |
| Recomputed `obs_flips_actual.b8` 대 공개 파일 | SHA-256 완전 일치 |

두 파일이 완전히 같았으므로 회로와 이진 데이터의 0/1 순서를 올바르게 읽었음을 확인했다.

## 5. 작은 시험의 PyMatching 결과

README 예제대로 SI1000이라는 기본 오류 가정에서 DEM을 만들고 PyMatching 2.3.1로 직접 최종값을 예측했다.

| 처리 방법 | 실패 / 반복 횟수 | 논리 실패율 | 95% 오차 범위 |
|---|---:|---:|---:|
| Direct PyMatching + SI1000 | 306 / 50,000 | 0.00612 | 0.00547–0.00684 |
| Provided correlated matching + SI1000 | 306 / 50,000 | 0.00612 | 0.00547–0.00684 |

직접 만든 예측 파일과 공개 예측 파일은 완전히 같았다. DEM 내부의 표현에는 일부 차이가 있었지만, 이 작은 $d=3$, 1-cycle 자료에서는 모든 반복 실험의 최종 예측이 같았다.

Python 코드로 50,000회 자료를 한꺼번에 처리해도 공개 예측과 0/1 값이 모두 같았다. 이후 전체 데이터도 이 방법으로 처리했다.

이 시험은 데이터를 올바르게 읽고 PyMatching을 제대로 실행했는지 확인하기 위한 것이다. 한 위치와 한 cycle만 사용했으므로 이 결과만으로 $\epsilon_d$나 $\Lambda$를 계산하지는 않았다.

## 6. 분석 프로그램이 한 일

`analyze_willow_data.py`는 실제값과 예측값을 읽어 다음 작업을 수행한다.

- 파일에 저장된 반복 횟수와 0/1 형식 확인
- 논리 실패 횟수와 실패율 계산
- 실패율의 95% 오차 범위 계산
- 같은 데이터에 대한 여러 디코더 결과 비교
- 여러 cycle의 실패율로 distance별 $\epsilon_d$ 계산
- $d=3,5,7$ 결과로 $\Lambda$ 계산
- 공개 예측값 재계산과 PyMatching 직접 실행
- CSV와 JSON 결과 저장

Pilot 결과:

```text
artifacts/qec_below_threshold_review/willow_data/pilot_decode.json
```

## 7. 전체 주요 데이터 분석 결과

13-cycle 조정용 자료를 제외하고 10-cycle과 30–250-cycle의 주요 실험 조건 364개를 분석했다. 표의 95% 범위는 본 프로젝트에서 간단히 계산한 오차 범위이며, 시간에 따라 실제 장치 성능이 변하는 효과까지 포함한 논문의 오차 범위와는 다르다.

| 디코더 처리 방법 | $\epsilon_3$ | $\epsilon_5$ | $\epsilon_7$ | $\Lambda$ | 간이 95% 범위 |
|---|---:|---:|---:|---:|---:|
| Public Libra RL, ensemble 51 | 0.006802 | 0.003372 | 0.001652 | **2.029** | 1.961–2.103 |
| Correlated matching RL | 0.007307 | 0.003876 | 0.002002 | 1.910 | 1.844–1.983 |
| Harmony RL | 0.007074 | 0.003842 | 0.002039 | 1.863 | 1.807–1.923 |
| Correlated matching SI1000 | 0.008072 | 0.004201 | 0.002180 | 1.924 | 1.862–1.993 |
| Harmony SI1000 | 0.007530 | 0.004064 | 0.002114 | 1.887 | 1.828–1.952 |
| **Direct PyMatching + SI1000** | **0.010438** | **0.007102** | **0.004485** | **1.525** | **1.476–1.578** |

논문의 기준값은 neural decoder $\Lambda=2.14\pm0.02$, Libra 디코더 101개를 합친 결과 $\Lambda=2.04\pm0.02$다. 공개 ZIP에는 Libra 디코더 51개를 합친 예측이 들어 있다. 구성은 조금 다르지만 다시 계산한 $\Lambda=2.029$는 논문의 2.04와 가까웠다. 같은 실제 오류 신호를 기본 SI1000 오류 가정과 단일 PyMatching으로 처리하면 $\Lambda=1.525$로 낮아졌다.

직접 PyMatching한 $d=3$ 자료 중 마지막 cycle의 측정점 4개는 실패율이 50% 이상이었다. 이 값은 논문의 로그 계산식에 넣을 수 없어 직선 맞추기에서만 제외했다. 측정 결과 자체는 원본 CSV에 그대로 남겨 두었다.

이 비교가 지지하는 결론은 다음과 같다.

1. 공개된 실제값과 예측값만으로 논문의 오류 억제 결과를 다시 계산할 수 있었다.
2. 모든 디코더에서 $\Lambda>1$이므로 code distance를 키울수록 논리 오류가 감소했다.
3. 직접 PyMatching의 오류율은 Libra보다 $d=3$에서 1.53배, $d=5$에서 2.11배, $d=7$에서 2.71배 높았다. 즉 코드가 커질수록 디코더 성능 차이가 더 커졌다.
4. 공개 ZIP에는 neural decoder 예측이 없으므로 neural decoder나 AlphaQubit 자체를 재현한 것은 아니다.

## 8. 산출물

```text
artifacts/qec_below_threshold_review/willow_data/<pathway>/
├── logical_error_by_configuration.csv
├── epsilon_by_patch_basis.csv
├── epsilon_by_distance.csv
└── summary.json

artifacts/qec_below_threshold_review/willow_reproduction_comparison.csv
artifacts/qec_below_threshold_review/willow_reproduction_comparison.png
```

포스터와 보고서에서는 공개 Willow 결과를 가장 중요한 결과로 사용한다. 별도의 가상 오류 모의실험은 왜 디코더에 따라 차이가 생길 수 있는지 설명하는 보조 결과로 사용한다.
