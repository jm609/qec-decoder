# d7 최종 평가의 CPU 비용과 재현성

2026-11-17 · Anaconda `fp` · 논문 완성 계획 5단계

같은 256-shot 배치에서 원본 PyMatching은 **1.6–2.4 ms**, 기존 shot별 margin과 paired detector는 **8.2–12.5초**였다. 실측 중앙값 기준 selector의 처리 시간은 각 조건의 원본보다 약 **3,900–7,600배** 길었다. 4단계에서 관측한 정확도와 함께 보고할 CPU 구현 비용이다. [측정 JSON](../../../artifacts/eval/d7_final_benchmark/20261117/benchmark.json) · [독립 정확도 평가](PREDECODER_D7_FINAL_EVALUATION_RESULTS.md)

## 측정 조건과 결과

4단계에서 첫 번째로 예약한 맵 788100, 기존 벤치마크와 같은 checkpoint 2, 약한·강한 ZZ 각각 첫 256 shot을 사용했다. 두 강도에서 같은 맵의 D₀ 일반 MWPM을 적용했다. 장치는 Intel Core i7-1165G7이며, 각 조건과 경로는 한 프로세스씩 순차 실행했다. PyTorch 스레드는 2개로 설정했고 MKL은 SEQUENTIAL이다. 한 번 워밍업한 뒤 세 번의 중앙값을 집계했다.

구간은 메모리의 원본 detector event에서 최종 논리 예측을 얻기까지다. Selector는 입력 텐서 변환, 원본 매칭, 후보 생성과 매칭, 점수 계산, 선택을 포함한다. 이미 구한 후보의 논리 예측을 최종 출력으로 재사용한다. 파일 읽기와 고정 레이아웃·모델·DEM 준비는 별도로 기록했다. DEM 준비 기록에는 공통 설정 함수에서 두 matching 모드의 객체를 생성하는 시간이 포함된다. 최대 메모리는 각 워커를 시작한 뒤 입력·모델 준비, 워밍업과 반복 측정까지의 Windows peak working set이다. 공통 Python·PyTorch 의존성의 메모리를 포함하고, 원본 경로에 checkpoint는 로드하지 않았다.

| 조건 | 경로 | 배치 시간 중앙값 | Shot/s | 최대 working set (MiB) |
| --- | --- | ---: | ---: | ---: |
| 약한 ZZ | 원본 PyMatching | 2.373 ms | 107,871 | 268.0 |
| 약한 ZZ | Margin | 9.173 s | 27.91 | 346.6 |
| 약한 ZZ | Paired detector | 10.152 s | 25.22 | 350.8 |
| 강한 ZZ | 원본 PyMatching | 1.635 ms | 156,546 | 268.6 |
| 강한 ZZ | Margin | 12.471 s | 20.53 | 350.4 |
| 강한 ZZ | Paired detector | 8.246 s | 31.05 | 347.7 |

원본은 배치당 matching 호출 1회로 syndrome 256행을 처리했다. Margin과 paired는 각각 257회, 8,704행으로 **34배**의 syndrome을 처리했다. 그래도 selector의 matching 함수에서 소비한 시간은 약 0.081–0.106초였다. 총시간의 대부분은 후보 생성과 신경망·후보 점수 계산을 묶은 구간에서 나왔다. 이 측정은 해당 구간 안의 연산별 비용을 분리하지 않는다.

강한 ZZ margin의 반복 시간은 7.25–12.48초로 흔들렸다. 배치 조건은 공통이지만 경로별 실행 순서와 시스템 상태의 시간차가 있어, 이 자료에서 margin과 paired 간 속도 우열을 결론 내리기 어렵다. 이 정도 정확도 차이에서는 현 CPU 구현의 추가 비용이 컸다. 측정 범위는 한 장치·맵·checkpoint의 배치 처리량이다.

## 실행과 재현 자료

측정한 모든 반복에서 세 경로의 논리 예측이 4단계의 저장 예측과 같았다. Margin·paired는 선택 mask와 전체 후보 mask도 일치했다. 메모리의 event만으로 만든 입력 텐서가 기존 평가 입력과 같음을 별도 확인했다. 선택 경로에는 실제 정답 대신 0을 넘기고, 저장한 두 observable의 순서대로 예측을 대조했다.

4단계의 48개 입력 사례와 384개 예측 파일 해시를 다시 확인했다. 24개 맵의 최종 수치와 구간을 맵별 기록으로 재집계했고, 벤치마크에 쓴 맵 788100의 원본·후보·선택 결과를 따로 재디코딩해 기존 집계와 일치함을 확인했다. 전체 24개 맵 재디코딩은 4단계에서 수행했으며, 아래 명령으로 반복할 수 있다.

```powershell
conda activate fp
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
python tools/summarize_d7_final.py
python tools/benchmark_d7_final.py --worker g1 margin
```

벤치마크 워커에서 `g0` 또는 `g1`과 `raw`, `margin`, `paired` 중 하나를 지정하면 같은 조건의 개별 측정 JSON을 표준 출력으로 받는다. `--worker`를 생략하면 여섯 조건을 실행해 `benchmark.json`을 새로 생성한다. 전체 재현에는 원래 workspace에 저장한 checkpoint와 평가 데이터가 필요하다.

[재현 매니페스트](../../../artifacts/eval/d7_final_benchmark/20261117/reproducibility_manifest.json)는 환경, 코드, checkpoint, 평가 입력·예측의 해시와 확인 기록을 포함한다. [소스·규약 아카이브](../../../artifacts/eval/d7_final_benchmark/20261117/reproducibility_sources.zip)에는 동결한 소스, 규약, 집계와 매니페스트가 들어 있다. 학습된 가중치와 전체 예측·데이터 파일은 매니페스트에 경로와 해시를 기록했다.

다음은 6단계다. 누출 수정 이후에 확보한 근거를 논문 주장에 연결하고, 실패 진단·선택 비교·독립 성능·계산 비용을 표와 그림으로 통합한다. [논문 완성 계획](../PREDECODER_PAPER_COMPLETION_PLAN.md)
