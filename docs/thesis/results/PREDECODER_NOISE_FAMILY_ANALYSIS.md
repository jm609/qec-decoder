# Noise-Family Analysis

This document records how the predecoder behaves across the constructed noise
families used in the experiments.

Source artifact:

- `artifacts/eval/nn/sedp_noise_family_analysis_summary.json`

Builder:

- `tools/build_noise_family_analysis_summary.py`

Related robustness artifact:

- `artifacts/eval/nn/sedp_d3_d5_robustness_summary.json`
- `PREDECODER_D3_D5_ROBUSTNESS_ANALYSIS.md`

## Family Roles

| family | role in current experiments |
| --- | --- |
| `stage_a_si1000` | training/validation noise family |
| `stage_b_local` | training/validation noise family |
| `stage_c_corr` | held-out correlated-noise evaluation family |

The main paper claim should be made on held-out `stage_c_corr`. The
`stage_a_si1000` and `stage_b_local` rows explain how selected-mode adoption
behaved before the held-out evaluation.

## D3/D5 Noise-Family Summary

| distance | family | role | mean selected delta | positive/neutral/harmful seeds |
| --- | --- | --- | ---: | ---: |
| d3 | `stage_a_si1000` | train/validation | `+0.021103896` | `7/1/0` |
| d3 | `stage_b_local` | train/validation | `+0.042207792` | `8/0/0` |
| d3 | `stage_c_corr` | held-out | `+0.006591797` | `8/0/0` |
| d5 | `stage_a_si1000` | train/validation | `+0.003246753` | `1/6/1` |
| d5 | `stage_b_local` | train/validation | `-0.000000000` | `1/6/1` |
| d5 | `stage_c_corr` | held-out | `+0.005615234` | `2/6/0` |

Interpretation:

- d3 is the cleanest cross-family success: both validation families and the
  held-out correlated family show positive selected-mode behavior.
- d5 is noisier on validation-family slices, but selected-mode adoption still
  produces a non-harmful held-out result because weak seeds fall back to raw
  no-edit PyMatching.
- The successful d3/d5 claim should therefore emphasize selected-mode safety:
  the method is useful because it can adopt local edits selectively instead of
  forcing edits on every seed or family.

## D7 Contrast

D7 is included as a contrast rather than a success result.

| metric | value |
| --- | ---: |
| checked seeds | `58` |
| mean validation delta | `+0.001904432` |
| validation positive/neutral/harmful seeds | `22/36/0` |
| mean held-out selected delta | `+0.000151536` |
| held-out selected positive/neutral/harmful seeds | `2/56/0` |
| mean held-out candidate delta | `-0.001515356` |
| held-out candidate positive/neutral/harmful seeds | `6/35/17` |
| validation-positive but held-out candidate harmful | `13` |
| validation-positive and held-out candidate positive | `5` |

Interpretation:

- D7 validation evidence is not enough for broad adoption.
- Even when validation delta is positive, held-out candidate behavior is more
  often harmful than positive in the observed comparison (`13` harmful vs `5`
  positive among validation-positive seeds).
- This supports the previous d7 conclusion: the blocker is
  selector-ranking/generalization, not merely candidate coverage or threshold
  calibration.

## Paper Use

This analysis should be used in the thesis to support two points:

1. The constructed noise environment is part of the experiment, not an
   afterthought. The model is evaluated across different noise families and a
   held-out correlated-noise family.
2. The d3/d5 success and d7 limitation are consistent with a generalization
   story: selected-mode safety transfers well enough at d3/d5, while d7
   exposes validation-to-held-out mismatch.

Do not overclaim that the model is uniformly robust to all noise families. The
correct claim is narrower:

> Under the constructed noise-family setup, the selected predecoder improves
> held-out `stage_c_corr` performance at d3 and d5, while d7 shows that
> validation-family evidence does not reliably generalize to larger-distance
> candidate selection.
