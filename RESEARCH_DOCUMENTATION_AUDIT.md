# Research Documentation Audit

Last checked: 2026-05-27.

## Verdict

The research is documented well enough to finish the thesis/poster from the
current artifacts. The core claim, method, d3/d5 positive results, d7 limitation,
reproducibility package, figure package, and rejected d7 directions all have
dedicated documents.

The main risk is not missing evidence. The main risk is document sprawl: several
large historical planning files contain superseded experiments. Use the
canonical documents below for final writing, and treat older planning logs as
archives unless a specific artifact path is needed.

## Canonical Reading Order

1. `PREDECODER_CLEAN_HANDOFF.md`
   - Fastest current-state summary.
   - Contains the frozen result table, d7 state, rejected directions, and next
     recommended work.
2. `PREDECODER_REMAINING_WORK.md`
   - Current finish plan and scope boundary.
   - Separates required thesis/poster work from optional research extensions.
3. `PREDECODER_FINAL_RESULT_TABLES.md`
   - Main d3/d5/d7 numbers for paper-facing tables.
   - Use this for final result values.
4. `PREDECODER_METHOD_DESCRIPTION.md`
   - Canonical method description.
   - Covers the 36-channel input, 3D residual trunk, local motif candidates,
     selector, selected-mode safety, and PyMatching handoff.
5. `PREDECODER_CONSOLIDATED_EVIDENCE.md`
   - Consolidated evidence across result summaries.
   - Best source for claim boundaries.
6. `main.tex`
   - Active thesis manuscript.
   - Static check passes as of this audit.

## Supporting Analysis Documents

Use these when writing detailed sections:

| topic | document |
| --- | --- |
| d3/d5 success structure | `PREDECODER_D3_D5_SUCCESS_STRUCTURE.md` |
| d3/d5 robustness | `PREDECODER_D3_D5_ROBUSTNESS_ANALYSIS.md` |
| paired statistics | `PREDECODER_D3_D5_PAIRED_STATISTICS.md` |
| noise-stage behavior | `PREDECODER_NOISE_FAMILY_ANALYSIS.md` |
| baseline boundary | `PREDECODER_BASELINE_COMPARISON.md` |
| ablation/failure paths | `PREDECODER_ABLATION_FAILURE_SYNTHESIS.md` |
| d7 selector bottleneck | `PREDECODER_D7_TARGETED_BOTTLENECK_ANALYSIS.md` |
| d7 harmful edits | `PREDECODER_D7_HARMFUL_EDIT_TAXONOMY.md` |
| hyperparameter sensitivity | `PREDECODER_HYPERPARAMETER_SENSITIVITY.md` |
| oracle recovery distribution | `PREDECODER_ORACLE_RECOVERY_DISTRIBUTION.md` |
| reproducibility | `PREDECODER_REPRODUCIBILITY_PACKAGE.md` |
| figures | `PREDECODER_FIGURE_PACKAGE.md` |
| evaluation-report response | `RESEARCH_EVALUATION_ACTION_PLAN.md` |

## Exploratory D7 Selector Documents

These are current but should remain outside the frozen main thesis results
unless deliberately discussed as rejected follow-up work:

| document | status |
| --- | --- |
| `RISK_AWARE_SELECTOR_EXPLORATORY_PLAN.md` | scope and gate for post-presentation d7 selector experiments |
| `RISK_AWARE_SELECTOR_EXPLORATORY_RESULTS.md` | latest risk-aware, risk-guard, harm-guard, and hard-ranking outcome |

Current conclusion:

- fair `risk_guard` seed2 restores the candidate settings but fails selected
  preservation
- harm-logit hard guard does not reliably filter selected harmful candidates
- `risk_guard + positive_negative_hard 0.5/0.5` improves seed2 candidate delta
  to `+0.002929688`, but selected mode remains raw no-edit because validation
  harmed count exceeds the safety cap
- do not expand this path to the d7 sentinel set or seeds `0..57`

## Archive Documents

These are useful historical logs, but not the first source for final writing:

| document | use |
| --- | --- |
| `NEXT_SESSION_HANDOFF.md` | exhaustive chronological handoff; very large |
| `MODEL_SELECTION_D3_D5_D7.md` | historical model-selection log |
| `PROJECT_REBUILD_STATUS.md` | rebuild and experiment history |
| `RESEARCH_PLAN_PREDECODER_MAIN.md` | older research plan log |
| `DECODER_*` and `PREDECODER_ARCHITECTURE_SPEC_V1.md` | early architecture planning |
| `MAIN_TARGET_WORK_SCHEDULE.md` | older schedule history |

## Static Checks

Latest manuscript static check:

```text
python tools/check_main_tex_static.py --main-tex main.tex --out artifacts/eval/nn/main_tex_static_check_summary_latest.json
```

Result:

```text
pass=True
failed_errors=0
failed_warnings=0
num_checks=16
```

Final result consistency check:

```text
artifacts/eval/nn/sedp_final_result_consistency_check.json
```

Recorded result:

```text
pass=True
num_checks=37
num_failed=0
```

## Remaining Documentation Gaps

These are small but should be handled before final submission:

1. `main.tex` does not yet include the latest post-presentation
   `risk_guard + positive_negative_hard` rejected experiment. This is acceptable
   if the experiment stays internal, but add one sentence in the discussion or
   appendix if the thesis claims that selector training/calibration variants
   were tested.
2. `main.tex` should explicitly mention validation-based best epoch selection
   if the thesis discusses learning stabilization or early stopping.
3. Decide final title language. Current `main.tex` uses the Korean title, while
   several presentation discussions preferred the English title. Keep the Korean
   title only if the university format requires it; otherwise update `main.tex`.
4. `NEXT_SESSION_HANDOFF.md`, `MODEL_SELECTION_D3_D5_D7.md`, and
   `PROJECT_REBUILD_STATUS.md` contain superseded paths. Do not use them as the
   primary source for final numbers without checking against
   `PREDECODER_FINAL_RESULT_TABLES.md` and the consolidated JSON.
5. `guide.txt` is empty and can be ignored.

## Current Paper-Facing Claim

Use this as the final claim boundary:

> The transition-aware neural pre-decoder improves selected-mode held-out
> performance over raw PyMatching for d3 and d5. For d7, candidate-oracle
> headroom remains large, but selected-mode gain is negligible because the
> selector cannot reliably rank and calibrate beneficial local edits against
> harmful ones. The d7 result should therefore be presented as a scaling
> limitation of the current candidate-wise selector, not as evidence that useful
> candidates are absent.
