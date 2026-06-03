# Graduation Thesis Draft

## Final Title

> 표면 코드 양자 오류 정정을 위한 전이 정보 기반 신경망 사전 디코더의 설계 및 성능 분석

English title:

> Design and Evaluation of a Transition-Aware Neural Pre-Decoder for
> Surface-Code Quantum Error Correction

Clean Korean core draft:

- `GRADUATION_THESIS_KOREAN_CORE_DRAFT.md`

Use the clean Korean draft as the current submission-format source. The older
Korean placeholder text in this file is partially encoding-corrupted and should
not be copied into the final thesis.

## Abstract Draft

본 연구는 표면 코드 기반 양자 오류 정정에서 기존 PyMatching 디코더의
입력 syndrome을 신경망으로 사전 보정하는 전이 정보 기반 neural
pre-decoder를 설계하고 평가한다. 직접 구성한 노이즈 환경과 syndrome
데이터셋을 바탕으로, 36채널 syndrome/noise volume을 입력으로 받는 3차원
residual 신경망과 local motif candidate selector를 결합하였다. 제안한
모델은 최종 logical-frame 판정을 직접 수행하지 않고, 선택된 local detector
edit 또는 raw no-edit syndrome을 PyMatching에 전달하는 방식으로 기존
matching decoder와 결합된다.

실험 결과, 제안한 selected-mode predecoder는 held-out `stage_c_corr`
환경에서 d3와 d5 표면 코드에 대해 raw PyMatching보다 각각 `+0.007568359`,
`+0.011230469`의 평균 정확도 향상을 보였다. 반면 d7에서는 selected-mode
향상이 `+0.000151536`에 그쳤다. 추가 분석에서 d7의 candidate-oracle
headroom은 여전히 크게 남아 있었으나, learned selector가 좋은 local edit를
안정적으로 선택하지 못하는 것으로 나타났다. 이는 d7 성능 한계가 candidate
coverage 부족이 아니라 selector ranking 및 generalization 문제임을
시사한다. 본 연구는 neural pre-decoding이 소규모 표면 코드에서 PyMatching
성능을 개선할 수 있음을 보이며, 더 큰 distance로 확장할 때 필요한 selector
설계상의 병목을 실험적으로 분석한다.

Keywords:

- quantum error correction
- surface code
- neural pre-decoder
- PyMatching
- syndrome decoding
- selector ranking

## Core Thesis Claim

The thesis should make this claim:

> A transition-aware neural pre-decoder can improve PyMatching on d3 and d5
> surface-code decoding by selecting local syndrome edits before matching, but
> d7 exposes a selector-ranking generalization bottleneck rather than a
> candidate-coverage bottleneck.

Do not claim:

- the method robustly improves d7
- the method replaces PyMatching
- the d7 candidate set is exhausted
- the noise model itself is the main proposed algorithmic contribution

## Recommended Chapter Structure

### 1. Introduction

Purpose:

Explain why quantum error correction is needed, why surface code decoding is
important, and why neural pre-decoding is worth studying.

Required points:

- Noisy quantum hardware requires quantum error correction.
- Surface code is a practical and widely studied QEC code family.
- PyMatching is a strong classical decoder, but neural pre-processing may help
  by simplifying or locally correcting syndrome structure before matching.
- This work studies a neural pre-decoder, not an end-to-end neural replacement
  for PyMatching.
- The research question is whether local neural syndrome edits can improve
  PyMatching and where the approach fails as code distance increases.

Suggested contribution paragraph:

```text
본 연구의 기여는 다음과 같다. 첫째, 직접 구성한 노이즈 환경에서 생성한
surface-code syndrome 데이터를 바탕으로 PyMatching과 결합 가능한 신경망
사전 디코더를 설계하였다. 둘째, 36채널 syndrome/noise volume과 local motif
candidate selector를 이용하여 local detector edit를 선택하는 구조를
제안하였다. 셋째, d3와 d5에서 raw PyMatching 대비 selected-mode 성능
향상을 확인하였다. 넷째, d7에서 candidate-oracle headroom은 존재하지만
learned selector가 이를 회수하지 못하는 현상을 분석하여, 확장 시 병목이
candidate coverage가 아니라 selector ranking/generalization에 있음을
보였다.
```

### 2. Background

Purpose:

Give enough background for a graduation-thesis reader to understand the method.

Sections:

- Quantum error correction and surface code
- Syndrome measurement and detector events
- PyMatching and minimum-weight perfect matching
- Neural decoders and neural pre-decoders
- Difference from direct CNN correction-field predecoding approaches

NVIDIA comparison should be placed here or in related work:

- NVIDIA Ising-Decoding is also a neural pre-decoder plus PyMatching style
  system.
- The present work differs by using explicit local motif candidate generation
  and patch-head candidate ranking.
- This makes candidate-level benefit/harm and oracle-gap analysis possible.

### 3. Dataset and Noise Environment

Purpose:

Make clear that the experiments were run under directly constructed noise
environments and generated syndrome datasets.

Required points:

- Surface-code circuits are generated for multiple distances.
- Noise is injected through configured noise models.
- Datasets contain detector-event syndrome volumes and metadata/noise channels.
- The input representation is a 36-channel syndrome/noise volume.
- The main held-out evaluation family is `stage_c_corr`.
- The thesis should state that the noise environment is part of the
  experimental setup, while the main algorithmic contribution is the
  pre-decoder structure and analysis.

Suggested section title:

> 실험 노이즈 환경 및 syndrome 데이터셋 구성

Use `PREDECODER_NOISE_FAMILY_ANALYSIS.md` for the noise-family result table.
The important thesis point is that d3/d5 improve on held-out `stage_c_corr`,
while d7 shows validation-to-held-out mismatch.

### 4. Proposed Method

Purpose:

Describe the actual model, not an idealized model.

Use `PREDECODER_METHOD_DESCRIPTION.md` as the canonical method source.

Pipeline:

```text
36-channel syndrome/noise volume
  -> SyndromeEditPreDecoder 3D residual trunk
  -> local motif candidate set
  -> patch-head CandidateEditSelector
  -> selected local detector edit or raw no-edit fallback
  -> PyMatching
  -> logical_class4 prediction
```

Required subsections:

- Input representation
- 3D residual predecoder trunk
- Local motif candidate generation
- Patch-head candidate selector
- Candidate-first selected-mode adoption
- PyMatching handoff

Important wording:

The model predicts a local syndrome edit candidate. It does not directly
predict the final logical class. PyMatching remains the final decoder.

### 5. Experimental Setup

Purpose:

Make the evaluation reproducible and defensible.

Required points:

- Distances: d3, d5, d7
- Baseline: raw no-edit PyMatching
- Compared modes: selected predecoder, candidate branch, target local-edit
  oracle
- Main metric: class4 logical-frame accuracy on held-out `stage_c_corr`
- Report selected delta over raw PyMatching
- Report improved/harmed shot counts where available

Baseline comparison boundary:

Use `PREDECODER_BASELINE_COMPARISON.md` here. The fair main comparison is raw
no-edit PyMatching versus selected predecoder on the same predecoder target
artifacts. Older direct neural decoders should be described as context
baselines, not as the main head-to-head result.

| baseline family | final decoder | thesis role |
| --- | --- | --- |
| raw no-edit PyMatching | PyMatching | fair main baseline |
| selected predecoder | neural local edit plus PyMatching | proposed method |
| FLFD/M3D-FLFD | direct neural `logical_class4` classifier | context/negative baseline |
| RectCNN | direct paper-style CNN classifier | readiness/context baseline |

### 6. Results

Main table:

| distance | raw PyMatching | selected predecoder | candidate branch | oracle | selected delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| d3 | `0.928710938` | `0.936279297` | `0.936279297` | `0.992187500` | `+0.007568359` |
| d5 | `0.888671875` | `0.899902344` | `0.899902344` | `0.978515625` | `+0.011230469` |
| d7 | `0.873046875` | `0.873198411` | `0.871531519` | `0.984375000` | `+0.000151536` |

Interpretation:

- d3 and d5 show positive selected-mode improvement over raw PyMatching.
- d5 is the strongest result.
- d7 selected mode remains essentially raw no-edit.
- The d7 oracle result shows that local-edit headroom remains available.

### 7. D7 Limitation Analysis

Purpose:

Turn d7 into a useful research result rather than an unexplained failure.

Required evidence:

- d7 mean selected delta: `+0.000151536`
- d7 mean actual candidate delta: `-0.001515356`
- d7 mean candidate-oracle delta: `+0.096679688`
- all `58` d7 seeds have positive oracle headroom
- d7 candidate outcomes: positive `6`, neutral `35`, harmful `17`
- among validation-positive candidate seeds, held-out outcomes are harmful
  `13`, neutral `4`, and positive `5`; false-positive ratio `59.09%`
- selected mode blocks all `17/17` harmful candidate seeds by falling back to
  raw no-edit
- simple adoption-grid search checked `183040` policies and found `0` passing
  preserve/recover/block policies
- candidate-compatibility pairwise top-k blocks seed54 but destroys seed2
  candidate delta to `-0.136718750`

Conclusion:

The d7 bottleneck is selector ranking/generalization. It is not candidate
coverage and not a scalar threshold calibration problem. The key failure mode
is validation-positive false positives: candidate branches that look usable on
validation but become harmful on held-out `stage_c_corr`.

## Integrated Core Draft V1

This section is the current paper-ready core draft. It is written in English
to avoid the encoding corruption already present in earlier Korean placeholder
paragraphs. It should be translated or adapted to the required university
format after the technical content is frozen.

### Method Draft

The proposed decoder is a transition-aware neural pre-decoder that works in
front of PyMatching. The model does not replace the matching decoder and does
not directly classify the final logical frame. Instead, it receives a
36-channel syndrome/noise volume, proposes a small local detector-event edit,
and then passes either the edited syndrome or the unchanged raw syndrome to
PyMatching. The final prediction is still the `logical_class4` output produced
after matching.

The implemented inference pipeline is:

```text
36-channel syndrome/noise volume
  -> SyndromeEditPreDecoder 3D residual trunk
  -> local motif candidate set
  -> patch-head CandidateEditSelector
  -> selected local detector edit or raw no-edit fallback
  -> PyMatching
  -> logical_class4 prediction
```

The input tensor uses the same 36 channel semantics across code distances,
while the spatial and temporal volume dimensions scale with distance. In the
successful d3 and d5 experiments, the input shapes are `[36, 4, 4, 4]` and
`[36, 6, 6, 6]`, respectively. The channel set includes detector events,
valid-detector masks, geometry metadata, noise-family indicators, distance and
round statistics, event-fraction summaries, and physical-noise summary
channels.

The neural trunk, `SyndromeEditPreDecoder`, uses a 3D convolutional stem,
three residual 3D convolution blocks, hidden width `24`, dense hidden
dimension `64`, and dropout `0.1`. For the d3/d5 recipe this gives `118834`
parameters. The trunk produces detector-level edit logits, shot-level
needs-edit logits, and pooled shot features. The pooled shot features are
passed to the candidate selector.

Candidate generation is deliberately constrained. The system does not allow
arbitrary high-weight syndrome edits. The successful recipe uses
`local_motif_selector`, keeps at most `16` local motif classes and top `32`
motif candidates, and disables raw policy candidates with
`selector-policy-candidate-mode none`. The identity/no-edit candidate is always
present. This design makes raw PyMatching fallback part of the method rather
than an external rescue path.

For each candidate, the patch-head `CandidateEditSelector` receives the pooled
shot embedding, candidate edit-weight summaries, local motif and pattern
features, detector geometry summaries, local evidence features, local patch
features, and benefit/harm transition features relative to raw PyMatching. The
selector assigns a scalar score to each candidate. The highest-scoring
candidate is not automatically applied; it must also pass the candidate-first
selected-mode safety policy.

The selector target is `benefit_harm`. A candidate is treated as useful only
when applying the local edit and then running PyMatching improves final
`logical_class4` correctness relative to raw no-edit PyMatching. This makes
the target decoder-aware: the model is trained around whether the edit helps
the final matching decoder, not merely around detector-level reconstruction.

Selected-mode adoption is an essential part of the proposed method. If the
candidate branch has insufficient validation evidence or triggers safety
guards, the selected path falls back to raw no-edit PyMatching. This policy is
important because local edits can help some seeds while harming others. In the
final evidence, d3 adopts the local selector for `4/4` seeds, d5 adopts the
local selector for `2/4` seeds and falls back for `2/4`, and d7 adopts the
local selector for only `2/58` seeds.

### Experimental Setup Draft

The experiments evaluate surface-code decoding at distances d3, d5, and d7.
The main held-out evaluation family is `stage_c_corr`. The training and
validation families include `stage_a_si1000` and `stage_b_local`. This setup
is used to test whether selected-mode behavior transfers from constructed
training/validation noise families to a held-out correlated-noise family.

The fair baseline is raw no-edit PyMatching on the same predecoder target
artifacts. The proposed method is the selected predecoder path, which either
applies the selected local edit before PyMatching or falls back to raw no-edit
PyMatching. The candidate branch reports what would happen if the learned
candidate edit were applied without final selected-mode fallback. The target
local-edit oracle reports the best available local-edit target behavior and
measures how much headroom remains in the candidate/edit space.

Older direct neural decoders, including FLFD-small, M3D-FLFD, and RectCNN,
are used only as context baselines. They predict `logical_class4` directly and
do not use PyMatching as the final decoder. Their role is to explain why the
research direction moved away from standalone neural classification and toward
neural pre-decoding plus PyMatching.

### Result Draft

The main held-out `stage_c_corr` result is:

| distance | raw PyMatching | selected predecoder | candidate branch | target local-edit oracle | selected delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| d3 | `0.928710938` | `0.936279297` | `0.936279297` | `0.992187500` | `+0.007568359` |
| d5 | `0.888671875` | `0.899902344` | `0.899902344` | `0.978515625` | `+0.011230469` |
| d7 | `0.873046875` | `0.873198411` | `0.871531519` | `0.984375000` | `+0.000151536` |

The result supports a narrow but clear positive claim. At d3 and d5, the
selected neural pre-decoder improves raw PyMatching on the held-out correlated
noise family. The d3 selected delta is `+0.007568359`; the d5 selected delta is
`+0.011230469`. D5 is the strongest result because the selected-mode policy
falls back to raw no-edit on weak seeds while adopted local edits have a clean
improved/harmed ratio.

The d3/d5 robustness summary over seeds `0..3` reinforces this point. D3 has
`4/0/0` positive/neutral/harmful held-out selected seeds with mean selected
delta `+0.007568359`. D5 has `2/2/0` positive/neutral/harmful held-out
selected seeds with mean selected delta `+0.011230469`. Neither successful
distance has a harmful selected seed in the current seed set.

The selected-mode behavior also shows why the safety policy is part of the
method. At d3, the selected branch improves/harms `104/73` shots. At d5, it
improves/harms `56/10` shots. At d7, the raw candidate branch improves/harms
`161/251` shots, which is unsafe, but selected mode falls back on most seeds
and limits the selected improved/harmed count to `16/7`.

### Ablation And Baseline Draft

The ablation trail supports the final architecture choice. Standalone direct
neural `logical_class4` classifiers did not scale reliably: FLFD-small reached
held-out `stage_c_corr` accuracies of `0.792968750` at d3, `0.761230469` at
d5, and `0.195312500` at d7, all below the relevant PyMatching context.
M3D-FLFD did not fix the collapse, and a stronger d5 multiscale run reached
only `0.077148438`. RectCNN remains only a readiness-scale context baseline,
not a main numerical comparison.

These negative baselines justify the pre-decoder design. The useful role for
the neural model is not to replace PyMatching, but to propose local syndrome
edits that PyMatching can then decode. The final method therefore combines a
3D neural trunk, local motif candidate generation, patch-head benefit/harm
ranking, selected-mode safety, and PyMatching handoff.

### D7 Limitation Draft

D7 should be presented as a controlled scaling limitation, not as a solved
learned-recovery result. The final d7 selected delta is only `+0.000151536`.
However, the target local-edit oracle remains high at `0.984375000`, and the
mean learned candidate-oracle delta is `+0.096679688`. All `58` checked d7
seeds have positive oracle headroom. Therefore, the limitation is not that the
candidate/edit space lacks useful local edits.

The actual learned candidate branch is the problem. Across the 58-seed d7
analysis, candidate outcomes are positive `6`, neutral `35`, and harmful
`17`. The mean actual candidate delta is `-0.001515356`, even though the mean
candidate-oracle delta is strongly positive. This means the selector often
ranks neutral or harmful local edits above edits that would have helped
PyMatching.

The harmful-edit taxonomy identifies the dominant failure type. Among
validation-positive candidate seeds, held-out outcomes are harmful `13`,
neutral `4`, and positive `5`. Thus, a validation-positive candidate branch is
more often harmful than beneficial on held-out `stage_c_corr`, with a
false-positive ratio of `59.09%`. The selected-mode guard blocks all `17/17`
harmful candidate seeds by falling back to raw no-edit, which prevents the
candidate branch from reducing the final selected accuracy.

Several attempted d7 fixes were rejected. A scalar adoption-grid search tested
`183040` simple policies and found `0` policies passing the
preserve/recover/block sentinel gate. Cross-family hard positive-vs-negative
training failed the seed54 false-positive gate. Candidate-compatibility
pairwise top-k blocked seed54 but destroyed the seed2 true-positive branch,
with candidate delta `-0.136718750`. These results show that d7 needs a real
selector-ranking/generalization redesign, not another broad threshold sweep.

### Discussion Draft

The main contribution is a decoder-aware neural pre-decoding framework with
candidate-level interpretability. The method improves PyMatching at d3 and d5
while preserving a clear fallback path to raw PyMatching. The d7 result is
also useful because it explains where the current approach fails: useful local
edits exist, but the learned selector cannot reliably distinguish true
beneficial edits from validation-positive false positives at larger distance.

This claim is intentionally limited. The thesis should not claim that the
method solves d7 or replaces PyMatching. It should claim that neural
pre-decoding can improve a strong matching decoder at smaller distances and
that scaling the approach requires better selector ranking, stronger
generalization, or richer global context.

### 8. Discussion

Purpose:

Explain why the result matters even with limited d7 improvement.

Points:

- Neural pre-decoding can help a strong matching decoder on small distances.
- Safety/adoption policy matters because harmful edits can reduce PyMatching
  accuracy.
- Candidate-level analysis gives more interpretability than direct
  correction-field prediction alone.
- Scaling to d7 requires better candidate ranking, stronger generalization, or
  richer global context.

### 9. Conclusion

Purpose:

Close with a precise, defensible statement.

Suggested conclusion:

```text
본 연구는 표면 코드 양자 오류 정정에서 PyMatching과 결합되는 전이 정보
기반 신경망 사전 디코더를 설계하고 평가하였다. 제안한 모델은 d3와 d5에서
raw PyMatching 대비 selected-mode 성능 향상을 보였으며, 이는 local syndrome
edit를 neural pre-processing으로 선택하는 방식이 기존 matching decoder의
성능을 개선할 수 있음을 보여준다. d7에서는 oracle headroom이 존재함에도
learned selector가 이를 안정적으로 회수하지 못했으며, 이를 통해 더 큰
distance로 확장하기 위해서는 candidate 생성보다 selector
ranking/generalization 개선이 핵심 과제임을 확인하였다.
```

## Figure and Table Plan

Generated figure package:

- `PREDECODER_FIGURE_PACKAGE.md`
- `artifacts/figures/predecoder/`

Recommended figures:

1. Overall decoding pipeline diagram
2. Neural predecoder and candidate selector architecture
3. d3/d5/d7 accuracy comparison bar plot
4. d7 oracle gap illustration

Recommended tables:

1. Dataset/noise environment summary
2. Main d3/d5/d7 result table
3. d3/d5 robustness table from `PREDECODER_D3_D5_ROBUSTNESS_ANALYSIS.md`
4. Noise-family table from `PREDECODER_NOISE_FAMILY_ANALYSIS.md`
5. Baseline comparison table from `PREDECODER_BASELINE_COMPARISON.md`
6. Ablation/failure-path synthesis table from
   `PREDECODER_ABLATION_FAILURE_SYNTHESIS.md`
7. d7 harmful-edit taxonomy table from
   `PREDECODER_D7_HARMFUL_EDIT_TAXONOMY.md`
8. d7 sentinel gate and rejected directions
9. NVIDIA/related neural predecoder structural comparison

## Next Writing Tasks

1. Use `PREDECODER_REMAINING_WORK.md` as the current task-order source.
2. Convert `Integrated Core Draft V1` into the final required
   Korean/university format.
3. Replace the corrupted Korean placeholder abstract/conclusion text with a
   clean final Korean version.
4. Prepare figures and captions: pipeline, architecture, main result, and d7
   oracle-gap/false-positive limitation.
5. Add a reproducibility appendix using
   `PREDECODER_REPRODUCIBILITY_PACKAGE.md`.
6. Run final summary regeneration, final result-table consistency check, and
   PowerShell-compatible `py_compile`.
7. Keep optional d7 experiments out of the main story unless a new objective
   passes the sentinel gate.

## Remaining Work Policy

The remaining work is mostly consolidation, not model discovery.

Required:

- final Korean/university-format prose
- figure and caption preparation
- reproducibility appendix polishing
- final result-table consistency check rerun
- final PowerShell-compatible syntax check

Optional:

- d3/d5 seed `4..7` or confidence intervals for stronger statistics
- one new d7 selector-ranking objective only if it is tested on the sentinel
  gate first

Avoid:

- broad d7 expansion of rejected recipes
- new feature branches before explaining the current selector-ranking failure
- claims that d7 is solved or that d7 candidate coverage is exhausted
