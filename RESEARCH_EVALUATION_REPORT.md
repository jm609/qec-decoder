# 연구 평가 보고서

작성일: 2026-05-16 (전면 재작성판)
대상: `C:\Users\82108\fp` 졸업논문 프로젝트
주제: Transition-aware Neural Pre-decoder for Surface-Code Logical-Frame Inference
한국어 제목: 표면 코드 양자 오류 정정을 위한 전이 정보 기반 신경망 사전 디코더의 설계 및 성능 분석

---

## 0. 요약 (Executive Summary)

**종합 점수: 8.7 / 10 (A 진입선의 졸업논문)**

**핵심 주장:**
> 전이 정보 기반 신경망 사전 디코더가 candidate-first safety 정책 하에 raw PyMatching을 d3/d5 held-out `stage_c_corr`에서 개선하며, d7는 후보 oracle headroom이 남아있음에도 selector ranking/일반화가 부족해 통제된 한계로 처리됨.

**검증된 수치 (held-out `stage_c_corr`, raw PyMatching 대비):**

| 거리 | seeds | mean delta | 95% bootstrap CI | one-sided sign p | 진술 강도 |
|---|---:|---:|---|---:|---|
| d3 | 0..7 (N=8) | **+0.006592** | `[+0.004517, +0.008545]` | **0.003906** | uniformly positive |
| d5 | 0..7 (N=8) | +0.005615 | `[+0.000000, +0.013672]` | 0.250 | selected-safe with positive mean |
| d7 | 0..57 (N=58) | +0.000152 | (보고 안 함) | — | controlled limitation |

**연구 강점:** 정직한 음성 결과 보고, 좁고 방어 가능한 주장, safety fallback이 baseline이 아닌 방법의 일부로 통합, 부트스트랩 CI + paired test + 15건 회귀 테스트의 다층 검증.

**남은 약점:** d5 CI 하한이 0에 닿음(시드 6개가 raw fallback), Stage D/E는 노이즈 주입 코드 자체가 미구현, 핵심 모델 파일이 10,212 LOC 단일 파일, 검증→held-out 일반화 균열(d7).

**9.0+ 진입 경로:** §F cross-paper 비교 표 1행(~4시간) + §C 민감도 sweep 2개(~1일). Stage D/E 평가는 `postprocess.py` 신규 모듈이 필요해 졸업논문 마감 일정 밖.

---

## 1. 연구 개요

### 1.1 최종 추론 경로 (`PREDECODER_METHOD_DESCRIPTION.md` 기준)

```
36채널 신드롬/노이즈 볼륨 입력
  → 3D residual 신경망 trunk (SyndromeEditPreDecoder, 118,834 파라미터)
  → 로컬 motif 후보 생성 (top-32, identity 포함, policy 후보 비활성)
  → CandidateEditSelector (patch-head MLP, benefit/harm 라벨로 학습)
  → 선택된 detector 편집 OR raw_no_edit fallback (candidate_first_safety)
  → PyMatching
  → logical_class4 (I/X/Z/Y)
```

**핵심 설계 결정:** 신경망은 PyMatching을 **대체하지 않음**. 국소 detector-event edit을 적용할지만 결정하고, 최종 logical-frame 디코더는 PyMatching이 담당. `raw_no_edit` fallback은 baseline이 아니라 **방법의 일부**.

### 1.2 적용 범위 (스코프)

| 항목 | 값 |
|---|---|
| 거리 | d3, d5, d7 |
| 노이즈 패밀리 (구현됨) | Stage A `si1000`, Stage B `local`, Stage C `corr` |
| 노이즈 패밀리 (미구현) | **Stage D `leak`, Stage E `dqlr`** (config 스키마만 정의, 노이즈 주입 코드 부재) |
| 학습/검증 | Stage A + Stage B 합산 |
| Held-out 평가 | Stage C (correlated) |
| Seeds | d3/d5: 0..7 (N=8), d7: 0..57 (N=58) |
| 입력 형상 | d3 `[36,4,4,4]`, d5 `[36,6,6,6]`, d7 `[36,8,8,8]` |
| 모델 파라미터 | 118,834 (24 hidden ch, 64 dense, 0.1 dropout) |

### 1.3 비교 baseline 경계

- **공정한 메인 baseline:** 동일 manifest 위의 raw no-edit PyMatching (`PREDECODER_BASELINE_COMPARISON.md`)
- **컨텍스트 baseline (음성 결과):** FLFD, M3D-FLFD, RectCNN — 직접 NN 분류기는 사전 디코더 채택 근거 제공용
- **상한 reference:** target local-edit oracle delta

---

## 2. 강점 (Strengths)

### 2.1 방법론적 신중함 — ★★★★★

- **Benefit/harm 라벨링:** detector 재구성 정확도가 아닌 **PyMatching 최종 정확도 향상 여부**를 직접 감독.
- **`local_motif_selector` + `--selector-policy-candidate-mode none`:** 임의 고가중치 편집을 허용하지 않음. 관측된 hard-shot motif 16 클래스, top-32 후보만.
- **`candidate_first_safety` adoption:** 검증 증거가 약한 시드는 `raw_no_edit`로 fallback. 이 fallback이 baseline이 아닌 방법의 일부로 통합되어 d5/d7에서 결정적으로 손해를 차단.
- **직접 NN 분류기 baseline을 음성 결과로 정직하게 보고** (FLFD/M3D/RectCNN, `PREDECODER_BASELINE_COMPARISON.md`).

### 2.2 한계 표시의 정직성 — ★★★★★

- d7 +0.000152를 "사실상 0"으로 명시 (`PREDECODER_FINAL_RESULT_TABLES.md`).
- d7 실패 원인을 **후보 커버리지 부족이 아닌 selector ranking/일반화 실패**로 진단 (`PREDECODER_D7_TARGETED_BOTTLENECK_ANALYSIS.md`). 후보 oracle headroom mean +0.0967은 유지됨.
- **183,040개 threshold 정책 sweep → 0개가 preserve/recover/block 게이트 통과** (`sedp_d7_sentinel_adoption_grid_summary.json`). 명확한 stop sign.
- Cross-family hard-negative loss, candidate-compatibility top-k 모두 음성 결과로 기각하고 문서화 (`PREDECODER_ABLATION_FAILURE_SYNTHESIS.md`).

### 2.3 통계적 엄밀성 — ★★★★★

- d3/d5 seed 0..7 부트스트랩 95% CI (20,000 resamples).
  - d3: `[+0.00452, +0.00854]` — 0 미포함, uniformly positive
  - d5: `[+0.00000, +0.01367]` — 하한 0에 닿음, conservative
- **Exact paired/sign test 보고** (`PREDECODER_D3_D5_PAIRED_STATISTICS.md`):
  - d3 one-sided sign p = `0.003906`, two-sided p = `0.007812`
  - d5 one-sided p = `0.25` — 강한 유의성 회피의 명시 근거
- d7는 N=58로 validation/held-out 산점도 + Pearson `-0.4529` 보고. "validation-positive 시드의 59.09%가 held-out에서 harmful"이 단일 수치가 아닌 분포 증거.

### 2.4 재현성 패키지 — ★★★★★

- `requirements.txt`: numpy 2.2.6, torch 2.10.0, stim 1.15.0, PyMatching 2.3.1 등 핀.
- `ENVIRONMENT.md`: Windows 10.0.26200, Python 3.10.20, CPU-only.
- `PREDECODER_REPRODUCIBILITY_PACKAGE.md`: 결과 → JSON → builder 명령 매핑.
- `sedp_final_result_consistency_check.json`: Markdown ↔ JSON 일관성 자동 검증 **37/37 pass**.
- 모든 디코더가 `schema_version` emit.
- 모든 결과는 산출물 재실행만으로 재생성 가능 (별도 학습 불필요).

### 2.5 코드 신뢰성 (회귀 테스트) — ★★★★☆

- `tests/test_predecoder_regression.py` **15건 / 3 클래스 / 320 LOC, 15/15 pass**:
  - `CandidateFirstSafetyRegressionTest` (4): adoption 결정 경계 (strong delta / harm guard / tie+high margin / global allow flag)
  - `SelectorHelperRegressionTest` (5): feature dim, top-k policy, benefit/harm score, transition feature, patch-head shape/invalid slice
  - `SummaryArtifactRegressionTest` (6): consolidated_evidence, seed CI, paired statistics, validation/heldout scatter, oracle recovery, hyperparameter sensitivity JSON 값 직접 검증
- 일관성 체크 37건 + 회귀 테스트 15건의 두 층 안전망.

### 2.6 Figure 패키지 — ★★★★★

6장 SVG 모두 존재 (`artifacts/figures/predecoder/`):
- `fig1_predecoder_pipeline.svg`
- `fig2_model_architecture.svg`
- `fig3_main_accuracy_comparison.svg`
- `fig4_d7_oracle_gap_false_positive.svg`
- `fig5_d7_validation_heldout_scatter.svg`
- `fig6_oracle_recovery_distribution.svg`

`tools/convert_predecoder_figures.ps1`로 PDF/PNG 변환 자동화. `main.tex`에 fallback 매크로로 삽입 준비됨.

### 2.7 메타-평가 / 작업 트래킹 — ★★★★★

- `RESEARCH_EVALUATION_ACTION_PLAN.md`: 평가 보고서 권고와 실제 작업의 명시 매핑. 어떤 항목을 완료/지연 처리했는지 정직 기록.
- `PREDECODER_REMAINING_WORK.md`: 남은 작업 분류, stop sign, 우선순위 매트릭스.
- `NEXT_SESSION_HANDOFF.md`: 세션 간 컨텍스트 보존.
- 졸업논문 수준에서 보기 드문 메타-문서 위생.

### 2.8 음성 결과의 가치 — ★★★★★

- "직접 NN 분류기는 d7에서 0.195로 붕괴" → 사전 디코더 아키텍처 선택의 근거 제공.
- "183,040개 threshold sweep 중 0개 통과" → d7 추가 시간 투입에 대한 명확한 stop sign.
- Cross-family hard-negative, candidate-compatibility top-k 둘 다 시도하고 명시 기각.
- 동급 졸업논문 중 부정 결과 정직성 상위.

---

## 3. 약점 및 우려사항 (Critical Concerns)

### 3.1 d5 CI 하한이 0에 닿음 — 🟡 보통

- d5 부트스트랩 95% CI `[+0.00000, +0.01367]` — 평균은 양수지만 하한이 정확히 0.
- 시드 0..7 중 6개가 `raw_no_edit` fallback (평균에 +0 기여), 2개만 selector 채택 (+0.0215, +0.0234).
- One-sided sign p = `0.25`. "통계적으로 유의" 같은 강한 표현 회피해야 하며, 본문은 "selected-mode safe with positive mean"으로 진술 boundary를 그어두고 있음.
- **현재 진술이 정확하므로 약점이 아닌 정직성의 일부.** 다만 d5 결과만으로는 강한 주장이 안 된다는 점을 본문 reviewer가 물을 수 있음.

### 3.2 절대 개선폭이 작고 oracle 회복률이 낮음 — 🟡 보통

| 거리 | 평균 selected delta | 평균 selected recovery | oracle headroom (target) |
|---|---:|---:|---:|
| d3 | +0.00659 | 10.38% | +0.06348 |
| d5 | +0.00562 | 6.25% | +0.08984 |
| d7 | +0.00015 | 0.14% | +0.11133 |

- d7 candidate-oracle recovery는 86.84% — **후보 자체는 더 큰 개선 여지를 가지지만 selector가 활용하지 못함**.
- 이는 selector 설계의 미완성 신호로 정직하게 결론에 남겨두는 것이 옳음.

### 3.3 d7 검증 → held-out 일반화 균열 — 🔴 심각, 다만 통제됨

- d7 검증-positive 후보 22개 중 held-out 결과: **harmful 13 / neutral 4 / positive 5** (false-positive 비율 59.09%).
- Validation delta ↔ held-out candidate delta Pearson **`-0.4529`** — 음의 상관까지 보임.
- 모든 17개 harmful candidate seed는 `candidate_first_safety`로 **17/17 100% 차단** → 선택형은 safe.
- 결론: d7는 **"controlled limitation"**. 통제는 되지만 학습된 일반화는 아직 안 됨. 약점이 아닌 limitation 챕터의 핵심.

### 3.4 d7 sentinel gate 100% 실패 — 🔴 심각

- 183,040개 단순 threshold 정책 중 preserve(`2,11`) ∧ recover(`0,28,43,45` 중 1개) ∧ block(`8,13,17,18,32,33,53,54`)을 동시에 만족하는 정책 **0개**.
- 최적 정책 mean selected delta조차 `-0.00046` (음수).
- `PREDECODER_REMAINING_WORK.md`에서 d7 deprioritize 결정의 근거.

### 3.5 노이즈 패밀리 좁음 — 🟡 보통 (구현 갭이 원인)

- Stage A/B 학습/검증 + Stage C held-out — **3개 패밀리만 평가**.
- Stage D (leakage surrogate), Stage E (DQLR)는 **노이즈 주입 코드 자체가 미구현**:
  - `config.py` — `WillowLeakageConfig`, `WillowDQLRConfig` dataclass + factory만 정의.
  - `noise_willowcore.py:160-165` — Stage D/E 요청 시 `UnsupportedWillowStageError`로 명시 차단.
    > *"Stage D/E leakage and DQLR are shot-level dynamics and should be implemented later in postprocess.py, not by silently reusing the Stage-C circuit."*
  - `noise_si1000.py:113-114` — `stage_d_postprocess_pending: True` 플래그만 메타데이터.
  - 참조된 `postprocess.py` 모듈이 부재.
- 정확한 진술: "Stage A/B 학습 → Stage C held-out 일반화 한 단계만 검증됨. Stage D/E는 shot-level postprocess 모듈 미구현으로 평가 범위 밖."
- limitation 챕터에서 *"미평가"*가 아닌 *"shot-level postprocess 모듈 미구현"*으로 정확히 적어야 함 (의도하지 않은 약점이 아니라 의도된 scope 결정).

### 3.6 핵심 모델 파일이 모놀리식 — 🟡 보통

- `decoders/syndrome_edit_predecoder.py`: **10,212 LOC** 단일 파일.
- selector / router / compatibility head / transition prior / motif vocabulary / CLI / 평가 전부 한 파일.
- 60+개 `DEFAULT_*` 상수가 ablation 없이 고정.
- §2.5 회귀 테스트 15건이 향후 분리 작업의 안전망 역할.
- 학위논문 마감 관점에서는 수용 가능, follow-up paper로 갈 경우 분리/체계적 sweep 필요.

### 3.7 하이퍼파라미터 민감도 범위 — 🟡 경미

- `PREDECODER_HYPERPARAMETER_SENSITIVITY.md`: **1개 변수** (`selector_identity_margin_loss_weight`) × **3개 sentinel 시드** (`0, 2, 5`).
- 결과: 0.25 → harmful, 0.5 → 최적, 1.0 → 과보수.
- 다른 핵심 하이퍼파라미터 (top-k, hard-shot weight, harm cap 등)는 미스윕.
- 본문에서 "compact sensitivity check"로 정직히 표시 중. 더 많이 하면 점수 +, 안 해도 마감 가능.

### 3.8 비트 일치 재현 불가 — 🟢 경미

- `ENVIRONMENT.md`가 "bit-identical determinism 보장 안 함"을 명시.
- PyTorch BLAS / CUBLAS_WORKSPACE_CONFIG / PYTHONHASHSEED 강제 없음.
- 평가-only 재실행은 결정론적 (이미 JSON 산출물). 학습부터 비트 일치 재현이 필요한 경우는 명시되지 않음 → 졸업논문 범위에서 수용 가능.

### 3.9 d5 paired test power 부족 — 🟢 경미

- d5 sign test가 nonzero seed 2개로만 계산되어 one-sided p=0.25.
- raw fallback이 평균에 +0 기여하지만 paired test power는 떨어뜨림.
- 진술을 "uniformly positive"가 아닌 "selected-safe with positive mean"으로 굳히면 약점이 정직성으로 변환됨.

---

## 4. 학위논문 관점에서의 평가

### 4.1 방어 가능한 주장 (Defensible)

- ✅ d3/d5 held-out `stage_c_corr`에서 시드 0..7, candidate-first safety 정책 하에 raw PyMatching을 개선
  - d3: +0.0066 (CI `[+0.0045, +0.0085]`, sign p=0.0039) — **uniformly positive**
  - d5: +0.0056 (CI `[+0.0000, +0.0137]`, sign p=0.25) — **selected-safe with positive mean**
- ✅ d7는 후보 커버리지가 아닌 **selector ranking/일반화** 병목
  - candidate oracle recovery 86.84% vs selected recovery 0.14%
  - validation vs held-out Pearson `-0.4529`
  - 183,040 단순 정책 sweep 전부 sentinel 실패
- ✅ 엔드투엔드 NN 분류기보다 사전 디코더+PyMatching 하이브리드가 우월
  - FLFD/M3D는 d7에서 0.195까지 붕괴 vs PyMatching 0.873
- ✅ `candidate_first_safety` fallback이 "장식"이 아닌 필수: d5/d7의 17/17 harmful candidate 차단.

### 4.2 회피한 것이 옳은 주장

- ❌ d7 해결 — 명시적으로 부정
- ❌ d3 oracle 갭 완전 회복 — 10.38%만 회복
- ❌ Stage A/B/C/D/E 전 노이즈 패밀리 일반 robust — A/B → C 한 방향만, D/E는 미구현
- ❌ d5에 "통계적으로 유의" — CI 하한 0에 닿음, sign p=0.25
- ❌ d>7 확장성 — d7에서 이미 selector 일반화 실패

### 4.3 종합 점수

| 평가축 | 점수 | 비고 |
|---|:---:|---|
| 연구 주제 명확성 | 9/10 | "사전 디코더"의 의미를 정확히 좁힘 |
| 방법론 신중함 | 9/10 | benefit/harm 라벨링과 safety fallback의 통합 |
| 실험 설계 | 7/10 | d3/d5 N=8, d7 N=58, held-out 분리. Stage D/E 미구현 |
| 통계적 엄밀성 | 8/10 | CI + exact paired test, d3 p=0.0039 |
| 음성 결과 정직성 | 10/10 | 동급 상위 |
| 코드 조직 | 7.5/10 | 모놀리식 단일 파일, 회귀 테스트 15건이 안전망 |
| 재현성 패키지 | 9/10 | requirements.txt + ENVIRONMENT + consistency check |
| Figure/문서 패키지 | 9.5/10 | 6장 SVG + 일관성 자동 검증 + action plan 메타-문서 |
| **종합** | **8.7 / 10** | **A 진입선의 졸업논문** |

---

## 5. 마감 전 권장 조치 (졸업논문 본문 보호)

### 5.1 필수 (논문 신뢰성 직결)

1. **d5 본문 진술 통일:** "uniformly positive"가 아닌 *"selected-mode safe with positive mean, CI lower bound at zero, paired sign p=0.25"*로 일관.
2. **d7 한계 챕터:** 후보 oracle recovery (86.84%) vs selected recovery (0.14%) figure로 1회 더 강조 → "candidate coverage가 아니라 ranking 문제"라는 메시지.
3. **Stage D/E limitation 명시:** *"shot-level postprocess 모듈(`postprocess.py`)이 미구현 상태이며, `noise_willowcore.py:160-165`에서 명시적으로 차단됨. 따라서 본 평가는 Stage A/B/C 범위로 한정한다."* — *"미평가"*가 아닌 *"미구현으로 인한 의도된 scope 결정"*으로 정확히 표기.
4. **"통계적 유의" 표현 grep 한 번 더:** d5 관련 강한 표현이 본문에 남아있지 않은지 확인.

### 5.2 권장 (시간 허용 시)

5. **§F cross-paper 비교 표 1행 (~4시간):** AlphaQubit, NVIDIA Ising decoder, Google neural decoder 등 SI1000 유사 셋업 정확도를 본문 표 1행으로 추가. "같은 셋업 아님"을 명시, head-to-head 주장 금지.
6. **§C 민감도 sweep 1~2개 추가 (~1일):** `--selector-local-motif-top-k` (8/32/64) 또는 `--selector-hard-shot-solved-weight` (3.0/6.0/12.0). d3 sentinel 2~3 시드만.
7. **d5 시드별 selected vs raw 분포 본문 표:** fallback이 실제로 어느 시드를 차단했는지 보이면 "selected-mode가 일을 하고 있다"는 메시지가 더 분명.

### 5.3 의도적으로 회피

- ❌ d7 추가 학습/sweep — `PREDECODER_REMAINING_WORK.md` 결정과 일치
- ❌ "통계적으로 유의" 또는 "robust generalization" 강한 표현
- ❌ 시드 추가 확장 — d3/d5 N=8 충분, d7 N=58 광범위
- ❌ Stage D/E 평가 — `postprocess.py` 신규 작성 필요, 마감 risk 큼

---

## 6. 발전 방안 (Score Improvement Roadmap)

### 6.1 1차 로드맵 (이미 실현된 작업) ✅

| 항목 | 1차 추천 | 실제 결과 | 효과 |
|---|---|---|---|
| §B paired/sign test | ✅ 권장 (방법2 저비용) | ✅ 완료 — d3 p=0.0039, d5 p=0.25 | 통계 7→8 |
| §D 회귀 테스트 5건 | ✅ 권장 | ✅ 초과달성 (15건) | 코드 7→7.5 |

이 두 작업으로 **8.4 → 8.7** 달성. 1차 로드맵의 "반나절 작업으로 +0.3" 예측이 정확히 실현됨.

### 6.2 8.7 → 9.0 도달용 (현재 추천)

#### §F. Cross-paper 비교 표 1행 — 방법론 9→9.5, +0.2 ⭐ **가성비 1순위**

- AlphaQubit, NVIDIA Ising decoder, Google neural decoder 등 SI1000-유사 셋업 정확도를 본문 표 1행으로 추가.
- **주의:** "같은 셋업이 아님"을 명시. 직접 head-to-head 주장 금지.
- **공수:** ~4시간 (참고문헌 확인 포함).
- **점수 영향:** 방법론 축이 9→9.5로 상승, 본문 깊이 인식 ↑.

#### §C. 민감도 sweep 1~2개 추가 — 실험 설계 7→7.5, +0.2

- 현재 1개 변수 × 3 sentinel 시드뿐.
- **추가 권장 변수:**
  1. `--selector-local-motif-top-k` (8 / 32 / 64): "후보 다양성이 selector를 망치는가?" 직접 답.
  2. `--selector-hard-shot-solved-weight` (3.0 / 6.0 / 12.0): hard-shot 가중치 ablation.
- **공수:** 2변수 × 3값 × 3시드 = 18 run, d3는 빠르므로 ~1일.

#### §I. d5 시드별 fallback 분포 본문 표 — 통계 8→8.5, +0.1

- `PREDECODER_D3_D5_PAIRED_STATISTICS.md`의 d5 seed-level fallback 표를 본문에 직접 인용.
- fallback이 어느 시드를 차단했는지 (seed 4, 6 candidate harm)를 명시 → safety 정책의 실효성을 본문에서 한 번 더 강조.
- **공수:** ~2시간 (이미 데이터 있음, 본문 인용만).

### 6.3 9.0 → 9.5 추가 작업

#### §G. 결정론 환경 강화 — 재현성 9→10, +0.1

- `environment.yml` (conda) 또는 `Dockerfile` 추가.
- 학습 진입점에 `torch.use_deterministic_algorithms(True)` + `PYTHONHASHSEED=0` 가드.
- 결정론적 모드 회귀 테스트 1건 추가.
- **공수:** ~반나절.

#### §H. Stage A vs Stage B 노이즈 일반화 ablation — 실험 설계 7.5→8, +0.2

- 현재 A/B 합산 학습 → C held-out. 추가로 "A only 학습 → B held-out" 1회 보고.
- 노이즈 패밀리 일반화 주장의 정량 범위 확보.
- **공수:** ~1일 (학습 1회).

#### §E. 모놀리식 파일 부분 분리 — 코드 조직 7.5→9, +0.3

- `syndrome_edit_predecoder.py` 10,212 LOC를 최소 3개로 분리:
  - `decoders/_predecoder/model.py` (trunk + selector)
  - `decoders/_predecoder/candidate_generation.py` (motif vocabulary + top-k)
  - `decoders/_predecoder/training.py` + `evaluation.py`
- **안전망:** 회귀 테스트 15건이 통과하면 안전. 일관성 체크 37건도 cross-link.
- **공수:** 1~2일.
- **마감 risk:** 졸업논문 직전엔 미루는 것이 합리적, follow-up paper 단계 권장.

### 6.4 의도적으로 회피해야 할 것

#### §A. Stage D/E 노이즈 패밀리 평가 — ❌ 본 마감 일정 외

- **구현 갭이 본질:**
  - `config.py`에 dataclass 스키마만 정의됨.
  - `noise_willowcore.py:160-165`가 `UnsupportedWillowStageError`로 명시 차단.
  - `postprocess.py` 모듈 부재.
- **필요 작업:** `postprocess.py` 신규 작성 (cycle-level leakage state evolution, neighbor coupling, DQLR lifetime reduction) + DEM 일관성 검증 + 차단 가드 해제 + 회귀 점검.
- **공수:** **1~2주**. 마감 risk 큼.
- **대안:** §3.5 limitation 챕터에서 *"shot-level postprocess 모듈 미구현, follow-up 작업으로 명시"*라고 정직히 적으면 약점이 *통제된 한계*로 자리잡음.

#### 기타 회피 항목

- **d7 추가 학습/sweep** — 183,040 정책 sweep으로 이미 stop sign
- **시드 16개 이상 광범위 확장** — d3는 N=8로 uniformly positive, ROI 낮음
- **direct NN 분류기 추가 튜닝** — context baseline 역할로 충분, 더 튜닝하면 비교 boundary 흐림
- **새 selector 아키텍처 도입** — 졸업논문 시점 risk 큰 카드

### 6.5 마감 시간별 권장 순서

| 가용 시간 | 권장 작업 | 예상 점수 |
|---|---|---|
| ✅ 완료 | §B paired test + §D 회귀 15건 | **8.4 → 8.7 (현재)** |
| ~4시간 | §F cross-paper 표 1행 | → 8.8 |
| 반나절 | + §I d5 fallback 분포 표 | → 8.9 |
| 1일 | + §C 민감도 sweep 2개 (d3 sentinel) | → 9.0 |
| 2~3일 | + §G 결정론 환경 강화 | → 9.1~9.2 |
| 1주 | + §H A-only → B 일반화 (1회 학습) | → 9.3~9.4 |
| 2주+ | + §E 파일 부분 분리 | → 9.5+ |
| 졸업 후 | §A Stage D/E (`postprocess.py` 구현 + 평가) | follow-up paper |

**현재 가성비 1순위는 §F (cross-paper 표 1행).** ~4시간이며 방법론 축 +0.5의 효과. §C 민감도 sweep은 학습이 필요하지만 d3는 빠르므로 1일 안에 가능.

---

## 7. 핵심 수치 요약

### 7.1 메인 결과 (held-out `stage_c_corr`, raw PyMatching 대비)

| 거리 | seeds | raw PyMatching | selected predecoder | mean delta | 95% bootstrap CI | one-sided sign p | selected modes |
|---|---:|---:|---:|---:|---|---:|---|
| d3 | 0..7 | 0.928711 | 0.935303 | **+0.006592** | `[+0.004517, +0.008545]` | **0.003906** | local 8/8 |
| d5 | 0..7 | 0.888672 | 0.894287 | **+0.005615** | `[+0.000000, +0.013672]` | 0.250 | local 2/8, raw 6/8 |
| d7 | 0..57 | 0.873047 | 0.873198 | +0.000152 | (보고 안 함) | — | local 2/58, raw 56/58 |

### 7.2 Oracle 회복률

| 거리 | target oracle delta | selected recovery | candidate recovery | candidate-oracle recovery |
|---|---:|---:|---:|---:|
| d3 | +0.0635 | 10.38% | 10.38% | — |
| d5 | +0.0898 | 6.25% | 3.53% | — |
| d7 | +0.1113 | 0.14% | -1.36% | **86.84%** |

### 7.3 d7 검증 → held-out 균열

| 항목 | 값 |
|---|---:|
| 검토 시드 | 58 |
| 검증-positive 시드 | 22 |
| → held-out harmful | 13 |
| → held-out neutral | 4 |
| → held-out positive | 5 |
| **False-positive 비율** | **59.09%** |
| validation vs held-out Pearson | −0.4529 |
| 검토 정책 (단순 threshold sweep) | 183,040 |
| preserve/recover/block 통과 | **0** |
| 최적 정책 mean selected delta | −0.000455 |
| harmful candidate seeds (총 17개) → fallback 차단 | **17/17** |

### 7.4 직접 NN 분류기 context (사전 디코더 채택 근거)

| 모델 | d3 | d5 | d7 |
|---|---:|---:|---:|
| FLFD-small | 0.793 | 0.761 | **0.195** |
| M3D-FLFD | 0.732 | 0.761 | — |
| M3D-FLFD stronger | — | **0.077** | — |
| raw PyMatching (동일 manifest) | 0.925 | 0.900 | 0.875 |

### 7.5 노이즈 패밀리 적용 범위

| Stage | 정의 | 구현 | 평가 |
|---|---|:---:|:---:|
| A `stage_a_si1000` | SI1000 base | ✅ `noise_si1000.py` | ✅ 학습/검증 |
| B `stage_b_local` | + WillowCore local 비균일성 | ✅ `noise_willowcore.py` | ✅ 학습/검증 |
| C `stage_c_corr` | + Stray correlated interaction | ✅ `noise_willowcore.py` | ✅ Held-out |
| D `stage_d_leak` | + Leakage surrogate | ❌ `noise_willowcore.py:160-165` 차단 | ❌ |
| E `stage_e_dqlr` | + DQLR | ❌ | ❌ |

---

## 8. 한 줄 결론

> 좁게 잡고 정직하게 보고한, 졸업논문 수준의 견고한 연구. 부트스트랩 CI + exact paired test(d3 sign p=0.003906) + 15건 회귀 테스트 + 6장 figure + 일관성 자동 검증 37/37 + action plan 메타-문서까지 갖춰져 **8.7/10**. d3 uniformly positive, d5 selected-safe with positive mean, d7 ranking failure(not coverage failure)로 진술 boundary가 수치 단위로 굳었고, Stage D/E는 `postprocess.py` 미구현이라는 *의도된 scope 제약*으로 정직히 처리하면 limitation으로 자리잡는다. 9.0+ 진입은 §F cross-paper 표 1행과 §C 민감도 sweep 2개, 즉 학습 부담이 없는 ~1.5일 작업이면 충분하다.

---

## 9. 평가 근거 (Evidence Base)

### 9.1 방법·기여 문서

- `PREDECODER_METHOD_DESCRIPTION.md` — 본 평가의 방법 정의 source
- `PREDECODER_D3_D5_SUCCESS_STRUCTURE.md`
- `PREDECODER_ARCHITECTURE_SPEC_V1.md`
- `RESEARCH_PLAN_PREDECODER_MAIN.md`
- `GRADUATION_THESIS_DRAFT.md`, `GRADUATION_THESIS_KOREAN_CORE_DRAFT.md`

### 9.2 결과·증거 문서

- `PREDECODER_FINAL_RESULT_TABLES.md` — 메인 표 source
- `PREDECODER_CONSOLIDATED_EVIDENCE.md`
- `PREDECODER_BASELINE_COMPARISON.md`
- `PREDECODER_ABLATION_FAILURE_SYNTHESIS.md`
- `PREDECODER_D3_D5_ROBUSTNESS_ANALYSIS.md` (+ `sedp_d3_d5_seed0_7_bootstrap_ci_summary.json`)
- `PREDECODER_D3_D5_PAIRED_STATISTICS.md` (+ `sedp_d3_d5_paired_statistics_summary.json`)
- `PREDECODER_NOISE_FAMILY_ANALYSIS.md`
- `PREDECODER_ORACLE_RECOVERY_DISTRIBUTION.md`
- `PREDECODER_D7_HARMFUL_EDIT_TAXONOMY.md`
- `PREDECODER_D7_TARGETED_BOTTLENECK_ANALYSIS.md`
- `PREDECODER_HYPERPARAMETER_SENSITIVITY.md`

### 9.3 메타-평가 / 트래킹 문서

- `RESEARCH_EVALUATION_ACTION_PLAN.md`
- `PREDECODER_REMAINING_WORK.md`
- `NEXT_SESSION_HANDOFF.md`

### 9.4 Figure

`artifacts/figures/predecoder/`:
- `fig1_predecoder_pipeline.svg`
- `fig2_model_architecture.svg`
- `fig3_main_accuracy_comparison.svg`
- `fig4_d7_oracle_gap_false_positive.svg`
- `fig5_d7_validation_heldout_scatter.svg`
- `fig6_oracle_recovery_distribution.svg`

### 9.5 코드·재현성

- `decoders/syndrome_edit_predecoder.py` (10,212 LOC, 모놀리식)
- `decoders/baseline_pymatching.py`, `decoders/baseline_rectcnn.py`, `decoders/factorized_logical_frame_decoder.py`, `decoders/multiscale_factorized_decoder.py`, `decoders/research_noise_aware_3d.py`
- `circuits.py`, `sample_dataset.py`, `noise_si1000.py`, `noise_willowcore.py`, `config.py`
- `requirements.txt`, `ENVIRONMENT.md`, `AGENTS.md`
- `tests/test_predecoder_regression.py` (15건 / 3 클래스 / 320 LOC, 15/15 pass)
- `main.tex` (535 LOC, predecoder 구조 본문)
- `PREDECODER_REPRODUCIBILITY_PACKAGE.md`
- `PREDECODER_FIGURE_PACKAGE.md`

### 9.6 자동 일관성 검증

- `artifacts/eval/nn/sedp_final_result_consistency_check.json` — Markdown ↔ JSON 37/37 pass
- `tests/test_predecoder_regression.py::SummaryArtifactRegressionTest` — JSON 핵심 값 6건 cross-check pass

### 9.7 구현 갭 (limitation의 코드 근거)

- `noise_willowcore.py:160-165` — Stage D/E `UnsupportedWillowStageError` 차단 가드
- `noise_si1000.py:113-114` — `stage_d_postprocess_pending`, `stage_e_postprocess_pending` 메타데이터 플래그
- `postprocess.py` 부재 — shot-level dynamics 모듈 미작성
