# QEC Below-Threshold 논문 리뷰 프로젝트

## 프로젝트 목표

Google Quantum AI의 *Quantum error correction below the surface code threshold*를 리뷰하고, 단순화된 Stim–PyMatching simulation으로 threshold와 correlated-noise 효과를 분석한다. 추가로 공개된 Willow $d=3,5,7$ 실험 데이터를 PyMatching으로 재디코딩해 matching-decoder logical scaling을 재현한다.

공개 hardware data는 사용하지만 Google neural-network decoder, AlphaQubit 또는 실시간 streaming stack의 정확한 재현은 목표로 하지 않는다.

## 폴더 구성

```text
qec_below_threshold_review/
├── README.md
├── run_experiment.py
├── plot_baseline.py
├── plot_correlated.py
├── analyze_statistics.py
├── analyze_willow_data.py
├── plot_willow_reproduction.py
├── docs/
│   ├── paper_summary.md
│   ├── week1_progress.md
│   ├── week2_progress.md
│   ├── week3_progress.md
│   ├── results_discussion_draft.md
│   ├── project_status.md
│   ├── poster_content_draft.md
│   ├── poster_simulation_section.md
│   ├── content_master.md
│   ├── final_report_draft.md
│   ├── paper_equations_explained.md
│   ├── willow_data_reproduction_plan.md
│   ├── willow_data_reproduction_progress.md
│   ├── team_project_plan.md
│   └── role_assignment_plan.md
└── paper/
    ├── quantum_error_correction_below_surface_code_threshold.pdf
    ├── supplementary_information.pdf
    └── author_correction_2026.pdf
```

## 문서 안내

GitHub에는 구현 코드와 Markdown 기록을 포함한다. 다운로드한 논문 PDF,
발표 파일, 공개 실험 데이터 archive와 생성 결과는 로컬 자료로 제외한다.
아래 `paper/` 링크는 로컬 보관본이며,
[공식 논문 페이지](https://doi.org/10.1038/s41586-024-08449-y)에서도 원문을 확인할 수 있다.

- [논문 요약](docs/paper_summary.md): 원문 핵심 결과, 수치, 한계와 후속 확인 과제
- [1주차 진행 기록](docs/week1_progress.md): 공식 자료 확인과 smoke-test 결과
- [2주차 진행 기록](docs/week2_progress.md): 통제 실험 runner, CSV schema와 첫 pilot 결과
- [3주차 진행 기록](docs/week3_progress.md): detection-probability-matched correlated scaling 결과
- [Results/Discussion 초안](docs/results_discussion_draft.md): 공개 Willow 재분석, simulation 비교, 해석과 figure caption
- [통합 작업 현황](docs/project_status.md): 완료 작업, 핵심 결과, 산출물과 남은 우선순위
- [내용 기준본](docs/content_master.md): 논문 사실, decoder 구분, 공개-data·simulation 결과와 포함 우선순위
- [최종 보고서 초안](docs/final_report_draft.md): 논문 리뷰, 방법, 통계 보강 결과와 한계를 통합한 보고서
- [주요 수식 해설](docs/paper_equations_explained.md): 논문의 핵심 수식을 가정부터 단계별로 도출하고 프로젝트 사용 여부를 구분
- [Willow 공개 데이터 재디코딩 계획](docs/willow_data_reproduction_plan.md): 105Q surface-code archive의 pilot, PyMatching 재디코딩과 $\Lambda$ 재현 절차
- [Willow 공개 데이터 진행 기록](docs/willow_data_reproduction_progress.md): archive 검증, pilot, 전체 decoder scaling 결과
- [포스터 원고·배치안 내용본](docs/poster_content_draft.md): 공개 Willow 결과를 주 figure로 반영한 내용 원고
- [A1 세로형 포스터 프로젝트 분석 파트](docs/poster_simulation_section.md): Willow 공개 데이터 재디코딩과 Figure 2A·2B 시뮬레이션의 영문 문구·배치·핵심 수치
- [팀 프로젝트 계획](docs/team_project_plan.md): 연구 범위, 실험 설계, 포스터와 일정
- [상세 역할분담](docs/role_assignment_plan.md): 3인 역할, 산출물, 완료 기준과 인계 일정
- [논문 원문](paper/quantum_error_correction_below_surface_code_threshold.pdf): Nature 본문 8쪽
- [Supplementary Information](paper/supplementary_information.pdf): 6개 절, Fig. S1–S25, Table S1–S10
- [2026 저자 정정](paper/author_correction_2026.pdf): Fig. 3a 축과 범례 표기 정정

## 재사용할 기존 코드

기존 졸업프로젝트 코드는 중복 복사하지 않고 저장소 루트의 다음 파일을 재사용한다.

- `../circuits.py`: Stim rotated memory circuit
- `../noise_willowcore.py`: correlated ZZ surrogate
- `../sample_dataset.py`: detector/observable sampling과 metadata
- `../decoders/baseline_pymatching.py`: PyMatching 평가
- `../requirements.txt`: 실행 dependency

프로젝트 전용 구현이 필요해지면 이 폴더에 최소한의 runner만 추가하고, 공통 기능은 위 파일을 재사용한다.

## 생성 결과 위치

Dataset, CSV, graph와 evaluation 결과는 저장소 규칙에 따라 다음 경로에 둔다.

```text
../artifacts/qec_below_threshold_review/
```

생성 결과는 source document처럼 직접 편집하지 않는다.

## 실험 실행

세 실험 조건을 같은 실행에서 평가한다.

1. Stage B data + independent prior
2. ZZ-only Stage C data + independent prior
3. 같은 Stage C sample + correlated-ZZ prior

```powershell
python qec_below_threshold_review/run_experiment.py `
  --out-csv artifacts/qec_below_threshold_review/pilot_d3_p003.csv `
  --distance 3 --basis both --p 0.003 `
  --batch-shots 2000 --max-shots 10000 --target-failures 100
```

`p_det` matching에서는 basis별 Stage B parameter를 `--independent-p`로 지정한다. 자체 점검은 `python qec_below_threshold_review/run_experiment.py --self-test`로 실행한다. 기존 결과를 덮어쓰려면 명시적으로 `--overwrite`가 필요하다.

Baseline figure는 다음 명령으로 재생성한다.

```powershell
python qec_below_threshold_review/plot_baseline.py `
  --input-csv artifacts/qec_below_threshold_review/baseline_distance_pilot.csv `
  --output-png artifacts/qec_below_threshold_review/baseline_distance_pilot.png `
  --lambda-csv artifacts/qec_below_threshold_review/baseline_lambda_pilot.csv
```

Correlated-scaling figure는 다음 명령으로 재생성한다.

```powershell
python qec_below_threshold_review/plot_correlated.py `
  --input-csv artifacts/qec_below_threshold_review/correlated_scaling.csv `
  --output-png artifacts/qec_below_threshold_review/correlated_scaling.png `
  --summary-csv artifacts/qec_below_threshold_review/correlated_scaling_summary.csv
```

통계 요약은 다음 명령으로 재생성한다.

```powershell
python qec_below_threshold_review/analyze_statistics.py `
  --baseline-csv artifacts/qec_below_threshold_review/baseline_distance_pilot.csv `
  --correlated-csv artifacts/qec_below_threshold_review/correlated_scaling.csv `
  --sensitivity-csv artifacts/qec_below_threshold_review/sensitivity_d3.csv `
  --out-dir artifacts/qec_below_threshold_review/statistics
```

공개 Willow pilot의 actual/predicted observable을 비교한다.

```powershell
python qec_below_threshold_review/analyze_willow_data.py `
  --data-dir artifacts/qec_below_threshold_review/willow_data/pilot/d3_at_q4_5/X/r01 `
  --prediction direct_pymatching=direct_pymatching_si1000/obs_flips_predicted.b8 `
  --prediction provided_matching=provided_correlated_matching_si1000/obs_flips_predicted.b8 `
  --out-json artifacts/qec_below_threshold_review/willow_data/pilot_decode.json
```

전체 archive 다운로드·MD5 검증 뒤 제공 decoder pathway의 scaling을 집계한다.

```powershell
python qec_below_threshold_review/analyze_willow_data.py `
  --archive artifacts/qec_below_threshold_review/willow_data/google_105Q_surface_code_d3_d5_d7.zip `
  --pathway libra_decoder_with_rl_optimized_prior `
  --out-dir artifacts/qec_below_threshold_review/willow_data/libra_rl
```

동일 main sweep을 README의 SI1000 prior로 직접 PyMatching decode하려면 `--pathway` 대신 `--direct-si1000`을 사용한다.

```powershell
python qec_below_threshold_review/analyze_willow_data.py `
  --archive artifacts/qec_below_threshold_review/willow_data/google_105Q_surface_code_d3_d5_d7.zip `
  --direct-si1000 `
  --out-dir artifacts/qec_below_threshold_review/willow_data/direct_pymatching_si1000
```

논문 기준, Public Libra 재집계와 direct PyMatching을 한 figure로 만든다.

```powershell
python qec_below_threshold_review/plot_willow_reproduction.py `
  --input-root artifacts/qec_below_threshold_review/willow_data `
  --output-png artifacts/qec_below_threshold_review/willow_reproduction_comparison.png `
  --output-csv artifacts/qec_below_threshold_review/willow_reproduction_comparison.csv
```

완료 결과: Public Libra RL $\Lambda=2.029$, direct PyMatching+SI1000 $\Lambda=1.525$.

## 현재 후속 작업

- 학교·학회 포스터 template, 규격과 팀 정보 확정
- 확정 내용으로 실제 포스터 제작
- 발표 예상 질문과 답변 정리
