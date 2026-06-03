# Predecoder Baseline Comparison

This document fixes how baseline comparisons should be presented in the thesis.

Source artifact:

- `artifacts/eval/nn/sedp_baseline_comparison_summary.json`

Builder:

- `tools/build_baseline_comparison_summary.py`

Related main-result artifact:

- `artifacts/eval/nn/sedp_predecoder_consolidated_evidence_summary.json`

## Comparison Boundary

There are two different comparison roles.

The fair main comparison is:

```text
raw no-edit syndrome -> PyMatching
selected local edit or raw no-edit -> PyMatching
```

These two rows are evaluated on the same predecoder target artifacts, so this
is the comparison that supports the final claim.

The older direct neural decoders are context baselines. They predict
`logical_class4` directly and do not use PyMatching as the final decoder. They
are useful because they explain why the project moved from standalone neural
classification to neural pre-decoding, but their numbers should not be
overclaimed as a strict head-to-head comparison against the final predecoder
artifacts.

## Main Same-Artifact Comparison

All rows use held-out `stage_c_corr`.

| distance | raw PyMatching | selected predecoder | candidate branch | oracle | selected delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| d3 | `0.928710938` | `0.936279297` | `0.936279297` | `0.992187500` | `+0.007568359` |
| d5 | `0.888671875` | `0.899902344` | `0.899902344` | `0.978515625` | `+0.011230469` |
| d7 | `0.873046875` | `0.873198411` | `0.871531519` | `0.984375000` | `+0.000151536` |

Interpretation:

- d3 and d5 are the successful selected-mode results.
- d7 should be presented as a scaling limitation, not a solved learned
  recovery result.
- Raw PyMatching is the primary baseline for the final result table.

## Direct Neural Context Baselines

These are direct `logical_class4` neural classifiers on the earlier class4
2k distance ladder. They are not the final model family.

| model | distance | held-out `stage_c_corr` accuracy | macro-F1 | role |
| --- | --- | ---: | ---: | --- |
| FLFD-small | d3 | `0.792968750` | `0.448712193` | direct neural baseline |
| FLFD-small | d5 | `0.761230469` | `0.216107569` | direct neural baseline |
| FLFD-small | d7 | `0.195312500` | `0.081699346` | direct neural baseline |
| M3D-FLFD | d3 | `0.731933594` | `0.255574756` | multiscale direct baseline |
| M3D-FLFD | d5 | `0.761230469` | `0.216107569` | multiscale direct baseline |
| M3D-FLFD stronger | d5 | `0.077148438` | `0.035811423` | stronger multiscale ablation |

Interpretation:

- The direct neural classifier path does not scale cleanly with distance.
- Multiscale dense-trunk changes did not solve the d5 collapse.
- This motivates the final architecture choice: use the neural model to
  propose local syndrome edits, then leave final decoding to PyMatching.

## PyMatching Refresh Context

The direct class4 refresh manifests give these raw PyMatching held-out
`stage_c_corr` accuracies:

| distance | manifest | PyMatching accuracy |
| --- | --- | ---: |
| d3 | `artifacts/datasets/dev_class4_2k/manifest.json` | `0.925292969` |
| d5 | `artifacts/datasets/dev_class4_d5_2k/manifest.json` | `0.899902344` |
| d7 | `artifacts/datasets/dev_class4_d7_2k/manifest.json` | `0.874511719` |

These are useful for historical context and for explaining why PyMatching is a
strong baseline. The final predecoder deltas should still be reported against
the raw PyMatching values on the same predecoder target artifacts.

## RectCNN Readiness Baseline

`decoders/baseline_rectcnn.py` remains a paper-style CNN architecture baseline.
The available RectCNN artifact is only a readiness-scale d3 evaluation:

| artifact | shots | accuracy | macro-F1 | note |
| --- | ---: | ---: | ---: | --- |
| `artifacts/eval/nn/class4_rectcnn_stagea_eval_stagec.json` | `24` | `0.041666667` | `0.020000000` | readiness only |

This should not be used as a main numerical result.

## Thesis Use

The thesis should use this comparison as follows:

1. Present raw PyMatching as the fair baseline for the final result table.
2. Present FLFD/M3D/RectCNN as context showing why a standalone neural decoder
   was not the final direction.
3. Emphasize that the proposed model is a neural predecoder plus PyMatching,
   not an end-to-end neural replacement for PyMatching.
