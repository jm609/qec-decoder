# 기존 predecoder 연구 무결성 감사

2026-11-15. 범위: 현재 predecoder의 입력·분할·후보 생성·선택·채택·평가,
공통 입력을 쓰는 neural baseline, 논문 성능 주장의 근거를 확인했다.
기존 dataset, checkpoint, 결과 JSON은 덮어쓰지 않았다.

## 1. 확인된 결함과 수정

### 정답 집계값이 입력에 포함됨

`sample_dataset.py`는 `qc_stats.logical_flip_fraction`을 label 평균으로 만든다.
수정 전 `research_noise_aware_3d._build_context_planes`는 이를 모든 shot의
상수 입력 채널로 복제했다. 원본 family 전체 2,048 shot의 통계이므로,
1,024-shot subset을 고르거나 train/validation을 나누어도 라벨 정보가 사라지지
않는다. Stage C 평가에서도 해당 family 정답의 집계값이 모델에 들어간다.
이는 개별 정답을 직접 주는 형태는 아니지만, 정답 없이 실행 가능한 평가 조건을
위반한다.

영향 경로는 predecoder, noise-aware 3D decoder, FLFD이다. 별도 learned hybrid
router도 `logical_flip_fraction`, `logical_x_flip_fraction`,
`logical_z_flip_fraction`을 직접 특징으로 사용했다.

공통 volume의 해당 slot을 `reserved_label_context=0`으로 바꾸고, router의
세 slot도 0으로 고정했다. 기존 36채널과 router feature 폭은 유지한다.
새 bundle에는 `label_context_policy=zero_reserved`를 기록한다.
Predecoder의 legacy checkpoint 평가는 경고와 입력 정책을 기록한다.
정답 통계는 결과 분석용 metadata에 남기되 모델 입력에서는 읽지 않는다.

이 변경은 기존 가중치가 정답 집계 정보를 사용해 학습됐다는 이력을 없애지 않는다.
새 성능 주장을 위해서는 수정 입력으로 재학습해야 한다. 관측 syndrome event
fraction은 label과 무관한 batch 통계로 유지했고, simulator의 noise 설명도
입력 가정으로 유지했다. 온라인·실제 장치 적용 가능성은 별도 검증 대상이다.

### d7의 최종 guard가 평가 결과 확인 뒤 변경됨

`archive/NEXT_SESSION_HANDOFF.md`의 seed54 실패 및 support-guard calibration
기록에 따르면, seed54의 Stage C selected delta `−0.006835938`을 확인한 뒤
`positive_min_nonzero=5`를 도입했다. 이어 기존 seed 0..54에 사후 적용해
seed54를 차단하고 seed0도 포기했다. 최종 58-seed 표는 이 정책을 사용한다.

개별 채택 함수가 validation 통계만 읽는다는 사실과, 정책 자체가 test 결과와
독립적으로 정해졌다는 주장은 다르다. 따라서 기존 논문의 “held-out 결과를
threshold 결정에 사용하지 않았다”는 포괄적 문장을 수정했다. Stage C는
신경망 gradient 학습에서는 제외됐지만, 후속 설계까지 독립적인 최종 test는
아니다. `17/17` harmful seed 차단은 관측된 사후 결과이며 안전 보장이 아니다.
관련 근거는 [기존 handoff](archive/NEXT_SESSION_HANDOFF.md)와
[support-guard 비교](../../artifacts/eval/nn/sedp_d7_candidatefirst_idmargin05_diagselect_posharmcap2_posmaxmargin15_posminnz5_plateauguard_mixed_selection_compare_seed0_57.json)에 남긴다.

### 불확실성 및 정책 설명의 범위가 불명확함

d3/d5의 bootstrap과 sign test는 같은 1,024 평가 shot을 공유하는 model seed의
변동을 다룬다. 이를 새 shot·잡음 맵에 대한 신뢰구간이나, 적응적으로 선택한
연구 전체에 대한 확증 검정으로 읽을 수 없다. 한·영 논문에 조건부 해석을 추가했다.

또한 실제 채택용 metric은 `accuracy + 0.001 × macro_f1`의 family 평균이다.
기존 candidate branch에는 shot margin이 이미 적용되고, family gate에는
strong·weak-positive·near-tie 분기가 있다. 논문의 threshold 요약이 모든 분기를
설명하는 것처럼 읽히지 않도록 수정했다. 성능표의 순이득은 계속 정확도 차이다.

### manifest 학습의 split 재현 정보 누락

기존 manifest checkpoint는 seed와 family는 저장하지만, 변경 가능한 split 비율과
실제 split index를 저장하지 않았다. 새 checkpoint에는 `max_shots`,
train/validation/test 비율, family별 세 index 목록을 추가했다.
이는 기존 checkpoint의 누락 정보를 복구하지 않는다. 기본 분할은 1,024 shot에서
717/154/153이며, 내부 test 부분까지 학습에 사용했다는 식의 논문 설명도 수정했다.

## 2. 결함으로 확인되지 않은 경로

- motif vocabulary와 selector 학습 bundle은 각 training family의 train split에서
  만든다. 확인한 active 경로에서 Stage C oracle target을 motif 학습에 넣는
  흐름은 발견하지 않았다.
- selector의 정답 배열과 target score는 loss·평가에 쓰인다. 고정된 candidate
  feature와 logit에 대해 해당 정답을 바꿔도 선택 mask는 변하지 않는 회귀 검사를
  추가했다. 이것은 context 누출과 별도 확인이다.
- target artifact의 raw PyMatching 예측과 현재 DEM 재디코딩, 두 observable의
  `logical_class4` 변환은 입력 민감도 점검에서 대조한다.
- detector-bit edit은 이 연구의 분류 입력 변환이다. 이를 물리 오류 보정이나
  모든 decoder의 성능 상한으로 해석하지 않는다. 두 oracle의 후보 집합도 다르다.

## 3. 기존 checkpoint 입력 민감도 점검

`tools/audit_predecoder_input_leakage.py`는 d3/d5 seed 0..7과 d7 seed
2,11,54,57을 비교한다. 같은 저장 checkpoint, 1,024 shot, margin, 채택 모드를
유지한 채 legacy label aggregate와 zero slot만 바꾼다. 각 legacy 결과가 저장된
candidate 정확도와 selected delta를 재현해야 진행한다. 원본 PyMatching 결과가
두 입력 조건에서 같은지도 확인한다. d7 네 seed는 선택된 사례 패널이며 58-seed
평균의 대체값이 아니다.

실행 예시:

```powershell
python tools/audit_predecoder_input_leakage.py --out-dir artifacts/eval/research_integrity_<new_run>/context_ablation
```

기존 출력 경로는 거부하며, legacy 입력 복원은 이 감사 스크립트 안에서만 한다.
평가 라벨을 사용하는 legacy 변형은 재현용이며 배포 또는 새 성능 주장에 사용하지
않는다.

총 20개 checkpoint의 legacy 결과 재현을 통과했다. 같은 입력 변경에 대한 결과는
다음과 같다. 정확도 향상의 단위는 pp이며, 기준선은 같은 shot의 raw PyMatching이다.

| 평가 범위 | legacy 입력의 selected gain | label slot=0의 selected gain | 변경된 candidate mask 수 |
| --- | ---: | ---: | ---: |
| d3 seed 0..7 평균 | +0.659180 | +0.659180 | 0 |
| d5 seed 0..7 평균 | +0.561523 | +0.561523 | 0 |
| d7 seed 2 | +0.488281 | +0.488281 | 0 |
| d7 seed 11 | +0.390625 | +0.585938 | 4 |
| d7 seed 54, 57 | 각각 0 | 각각 0 | 0 |

d7 seed11의 정확도는 `0.876953125`에서 `0.87890625`로 바뀌었다.
기준선 대비 improved/harmed는 `10/6`에서 `9/3`으로 바뀌어 정답이 순증 2개다.
이 결과는 label 집계 입력이 실제 행동을 바꿀 수 있음을 보이지만,
기존 d3/d5 향상이 그 입력 때문에 생겼다고 보여주지는 않는다. 입력 제거로
성능이 같거나 좋아졌더라도, 기존 가중치의 학습 이력과 평가 데이터 재사용 문제는
해소되지 않는다. d7 네 사례를 전체 58개 seed의 수정 결과로 일반화하지 않는다.

기계 판독 가능한 [감사 집계 JSON](../../artifacts/eval/research_integrity_20261115/integrity_audit_summary.json)에
원본 audit summary 경로와 해시를 기록했다. 중단된 첫 실행의 d3 완료분을 보존하고,
d5/d7은 별도 새 경로에서 완료했다. 첫 실행의 미완료 d5 파일은 집계에 포함하지 않았다.

검증 결과:

- `python -m unittest discover -s tests -v`: 21개 통과. 정답 집계값 변경·제거에
  대한 입력 불변성, router의 세 label 집계값 차단, selector 정답 변경 시 행동
  불변성을 포함한다.
- 32-shot, 1-epoch manifest 학습과 checkpoint 재평가 통과. `zero_reserved` 정책,
  저장된 split 비율, 서로 겹치지 않고 전체를 덮는 `22/5/5` split index를 확인했다.
  이 smoke check는 연구 성능 재학습 실험이 아니다.
- 수정 Python 파일의 `py_compile`, 문서 링크, `git diff --check` 통과.
- 한·영 논문 정적 검사 각각 오류·경고 0개, 기존 결과표 37개 수치 대조 통과.
  이는 수치·문서 정합성 검사이며 실험 설계의 타당성 검정이 아니다.
- `graphify update .` 완료. LaTeX 실제 컴파일·레이아웃 검사는 이번 범위에서 하지 않았다.

주 검증 환경은 `C:/Users/82108/anaconda3/envs/fp/python.exe`이다.
기존 Overleaf ZIP은 감사 정정 전 원고를 포함하므로 현재 제출용으로 쓰지 않는다.

## 4. 성능 주장 복구 조건

1. 정답 집계값을 읽지 않는 입력으로 d3/d5/d7을 재학습한다. 모델·후보·채택 규칙은
   개발 데이터만으로 정하고 split, source hash, package version을 저장한다.
2. 기존 Stage C와 sentinel seed는 개발·회귀 점검용으로 표시한다. 최종 맵과
   sampler seed는 새로 예약하고, 평가 전에 설정과 주 비교를 고정한다.
3. 같은 shot의 raw/edited 예측을 짝지어 비교한다. 독립 맵 변동과 모델 seed 변동을
   구분하고, 정확도 차이·개선·악화 개수와 불확실성을 보고한다.
4. 새 결과가 기존 향상을 뒷받침하는 경우에만 논문의 개선 주장을 갱신한다.
   그 전까지 기존 숫자는 탐색적 기록이다. 입력을 0으로 바꿔도 결과가 같다는 것만으로
   재학습이나 독립 평가를 생략하지 않는다.
