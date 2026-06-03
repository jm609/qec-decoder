# Baseline And Noise Readiness

This file records whether the current comparison baselines and noise
environments are ready enough to support new decoder work.

## Ready Now

- Classical comparison baseline: `decoders/baseline_pymatching.py`
- Neural comparison backend: `decoders/research_noise_aware_3d.py`
- Lightweight neural sanity baseline: `decoders/baseline_rectcnn.py`
- Noise families available through `sample_dataset.py`:
  - `ideal`
  - `stage_a_si1000`
  - `stage_b_local`
  - `stage_c_corr`

## Verified In This Audit

- Built a fresh dual-axis smoke dataset covering `ideal`, `stage_a_si1000`,
  `stage_b_local`, and `stage_c_corr`:
  - `artifacts/datasets/dev_readiness_dual_axis/dual_axis_manifest.json`
- Ran aligned PyMatching comparison on that dual-axis dataset:
  - `artifacts/eval/pymatching/readiness_dual_axis/dual_axis_pymatching_summary.json`
- Ran aligned neural comparison on that same dual-axis dataset:
  - `artifacts/eval/nn/readiness_dual_axis/dual_axis_experiment_summary.json`
- Ran a direct RectCNN smoke train on Stage A:
  - `artifacts/eval/nn/readiness_rectcnn_stagea_train.json`
- Built a fresh per-shot class4 smoke dataset covering `ideal`,
  `stage_a_si1000`, `stage_b_local`, and `stage_c_corr`:
  - `artifacts/datasets/dev_class4_readiness/manifest.json`
- Ran PyMatching directly on that class4 manifest and confirmed class4-aware
  reporting:
  - `artifacts/eval/pymatching/class4_readiness_manifest.json`
- Ran RectCNN train/eval directly on `logical_class4`:
  - `artifacts/eval/nn/class4_rectcnn_stagea_train.json`
  - `artifacts/eval/nn/class4_rectcnn_stagea_eval_stagec.json`
- Ran the noise-aware 3-D experiment directly on `logical_class4`:
  - `artifacts/eval/nn/class4_noiseaware_manifest/experiment_summary.json`

## What This Means

- The active comparison path is operational for axis-wise experiments.
- The active comparison path is also operational for the first Bell-pair-based
  per-shot `logical_class4` experiments.
- The project can generate fresh datasets and run both classical and neural
  comparisons on `ideal/A/B/C` without touching legacy code.
- The current rebuilt stack is organized enough to start class4-decoder work
  without further baseline/noise infrastructure cleanup.

## Not Ready Yet

- `stage_d` and `stage_e` noise are still missing.
- `variant="xzzx"` is still scaffold reuse, not a real Willow-native schedule.
- Final recovery-based comparison is not implemented yet; current comparison is
  still label/frame oriented.
- The current class4 path depends on the Bell-pair reference-qubit readout,
  not on a native same-shot logical-frame scaffold.

## Active Mainline For Comparison Work

1. `sample_dataset.py`
2. `tools/build_dual_axis_manifest.py`
3. `tools/run_dual_axis_pymatching.py`
4. `tools/run_dual_axis_experiment.py`
5. `decoders/baseline_pymatching.py`
6. `decoders/research_noise_aware_3d.py`
7. `decoders/baseline_rectcnn.py`
