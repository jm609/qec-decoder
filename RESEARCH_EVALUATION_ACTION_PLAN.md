# Research Evaluation Action Plan

This document maps the latest `RESEARCH_EVALUATION_REPORT.md` recommendations
to concrete project work. The current report is the 2026-05-16 rewritten
version and rates the project at `8.7/10`.

## Evaluation Report Verdict

The thesis is defensible as an A-entry graduation thesis if the claim boundary
stays conservative:

- d3: uniformly positive selected-mode result over seed `0..7`.
- d5: positive selected-mode mean and no harmful selected seed, but the
  bootstrap confidence interval touches zero and six seeds are raw no-edit
  fallback.
- d7: controlled scaling limitation, not solved recovery.
- d7 failure cause: selector ranking/generalization, not candidate coverage.
- Stage D/E: excluded from quantitative evaluation because shot-level
  `postprocess.py` dynamics are not implemented and `noise_willowcore.py`
  explicitly blocks these stages.

The report explicitly discourages broad d7 retraining, another threshold-only
sweep, and rushed Stage D/E implementation.

## Required Actions

| priority | report request | action taken |
| ---: | --- | --- |
| 1 | Keep d5 wording conservative | `main.tex` and paired-statistics docs state d5 as positive-mean and selected-safe, not uniformly positive |
| 2 | Show why d5 selected mode is meaningful | `main.tex` now includes a d5 seed-level fallback table showing local-adopted seeds `2,3` and harmful candidate seeds `4,6` blocked by fallback |
| 3 | Emphasize d7 oracle recovery vs selected recovery | `main.tex` states d7 selected recovery `0.14%` vs candidate-oracle recovery `86.84%` and frames d7 as selector ranking/generalization limitation |
| 4 | Check for overly strong significance wording | Paired/sign test summary is included; d5 exact one-sided p-value is `0.250000000`, so strong d5 significance language is avoided |
| 5 | Clarify Stage D/E scope | `main.tex` now states Stage D/E are outside the quantitative scope because `postprocess.py` is absent and `noise_willowcore.py` blocks these stages |

## High-Value Follow-Ups Completed

| report roadmap item | status | artifact |
| --- | --- | --- |
| B method 2: paired/sign test without new training | complete | `PREDECODER_D3_D5_PAIRED_STATISTICS.md` |
| D: lightweight regression tests | complete | `tests/test_predecoder_regression.py` now has 18 tests covering selector helper invariants, summary artifacts, `main.tex` static checks, the six-figure package summary, and Overleaf package manifest |
| I: d5 seed-level fallback table | complete in manuscript | `main.tex`, Table `tab:d5-fallback` |

## Remaining 8.7 To 9.0 Work

| item | value | risk | decision |
| --- | --- | --- | --- |
| F: cross-paper comparison table | high context value; improves method positioning | low, but requires source checking | complete in `main.tex` |
| C: compact sensitivity sweep | improves experimental-design score | moderate runtime | optional after manuscript polish |
| final figure conversion and compile check | required for submission package | blocked locally by missing converter/LaTeX tools | run when tools are available |
| final consistency/test pass | required closeout | low | run at final closeout |

## Actions Deferred

| deferred item | reason |
| --- | --- |
| Stage D/E held-out evaluation | Requires implementing `postprocess.py`, shot-level leakage/DQLR dynamics, DEM consistency checks, dataset regeneration, baseline regeneration, and regression tests; this is a follow-up project, not a near-deadline thesis task |
| Broad d5 seed `8..15` extension | d5 is already defensible as conservative selected-mode recovery; paired tests and fallback table make the limitation explicit |
| d7 additional tuning/sweep | Existing evidence marks this as low-value unless a genuinely new selector-ranking objective first passes a sentinel gate |
| Monolithic file refactor | Valuable for maintainability, but risky near thesis closeout without a larger test suite |

## Current Action Outcome

The report-driven closeout work now consists of:

- `main.tex` with conservative d3/d5/d7 wording, cross-paper context table, d5
  fallback table, and Stage D/E scope clarification.
- `PREDECODER_D3_D5_PAIRED_STATISTICS.md`.
- `artifacts/eval/nn/sedp_d3_d5_paired_statistics_summary.json`.
- `tools/build_d3_d5_paired_statistics_summary.py`.
- `tests/test_predecoder_regression.py`.
- `PREDECODER_REMAINING_WORK.md` and `NEXT_SESSION_HANDOFF.md` as the forward
  planning documents.

## Next Default Task

The next report-aligned task should be final manuscript packaging: figure
conversion/compile verification, school-format metadata, and final consistency
checks. A compact sensitivity sweep remains optional if there is time, but do
not reopen d7 broad tuning before the manuscript package is stable.
