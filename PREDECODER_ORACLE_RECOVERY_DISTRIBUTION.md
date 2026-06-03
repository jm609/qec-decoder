# Oracle Recovery Distribution

This note adds the seed-level oracle recovery distribution requested by the
research evaluation report. It does not introduce a new model or rerun training;
it reuses the canonical d3/d5 seed `0..7` comparisons and the d7 support-guard
candidate-oracle analysis.

Source artifacts:

- `artifacts/eval/nn/sedp_d3_candidatefirst_policy_pairwise_seq_selection_compare_seed0_7.json`
- `artifacts/eval/nn/sedp_d5_candidatefirst_policy_pairwise_selection_compare_seed0_7.json`
- `artifacts/eval/nn/sedp_d7_support_guard_candidate_oracle_analysis_seed0_57.json`
- `artifacts/eval/nn/sedp_oracle_recovery_distribution_summary.json`

Builder:

- `tools/build_oracle_recovery_distribution_summary.py`

Figure:

- `artifacts/figures/predecoder/fig6_oracle_recovery_distribution.svg`

## Definition

Oracle recovery is computed against the fixed target local-edit oracle gap:

```text
oracle recovery = selected delta over raw PyMatching / target local-edit oracle delta
```

The candidate branch recovery is computed the same way, but before
selected-mode adoption. Negative recovery means the candidate branch is worse
than raw PyMatching.

## Selected-Mode Recovery

| distance | seeds | target oracle delta | mean selected recovery | min / median / max | positive/neutral/harmful seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| d3 | `8` | `+0.063476562` | `10.38%` | `3.08% / 10.77% / 16.92%` | `8/0/0` |
| d5 | `8` | `+0.089843750` | `6.25%` | `0.00% / 0.00% / 26.09%` | `2/6/0` |
| d7 | `58` | `+0.111328125` | `0.14%` | `0.00% / 0.00% / 4.39%` | `2/56/0` |

Interpretation:

- d3 is not a single lucky seed result: every checked seed recovers a positive
  fraction of the available target local-edit oracle gap.
- d5 is sparse but selected-safe: only two seeds recover oracle gap, while six
  seeds fall back to zero recovery rather than becoming harmful.
- d7 selected mode recovers almost none of the available target oracle gap.

## Candidate-Branch Recovery

| distance | mean candidate recovery | min / median / max | positive/neutral/harmful seeds |
| --- | ---: | ---: | ---: |
| d3 | `10.38%` | `3.08% / 10.77% / 16.92%` | `8/0/0` |
| d5 | `3.53%` | `-13.04% / 0.00% / 26.09%` | `2/4/2` |
| d7 | `-1.36%` | `-17.54% / 0.00% / 4.39%` | `6/35/17` |

Interpretation:

- d5 candidate-only behavior is not safe: seed `4` and seed `6` have negative
  oracle recovery. Selected-mode adoption blocks both by choosing raw no-edit.
- d7 candidate-only recovery is negative on average, despite large oracle
  headroom.

## D7 Candidate-Oracle Headroom

For d7, the candidate-oracle recovery distribution is also available from the
support-guard oracle analysis:

| metric | value |
| --- | ---: |
| mean candidate-oracle recovery | `86.84%` |
| min candidate-oracle recovery | `64.04%` |
| max candidate-oracle recovery | `114.04%` |
| selected mean recovery | `0.14%` |

This is the central d7 evidence: candidate-level oracle headroom is high, but
the selected branch recovers almost none of it. The limitation is therefore not
that local-edit candidates are absent; it is that the learned selector does not
rank safe useful candidates reliably at d7.

## Thesis Claim Update

Use this result to make the claim more precise:

> The predecoder recovers a modest but positive fraction of the target
> local-edit oracle gap on d3 and selected d5 seeds. At d7, high candidate-oracle
> headroom remains, but selected-mode recovery is nearly zero, confirming a
> selector-ranking/generalization bottleneck.

Do not claim that the method closes most of the oracle gap. The measured
selected recovery remains small: `10.38%` on d3, `6.25%` on d5, and `0.14%` on
d7.
