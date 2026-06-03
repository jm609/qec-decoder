# Environment Record

This file records the environment used for the final predecoder evidence and
summary generation. It is meant to complement `requirements.txt`; it does not
claim bit-identical reproducibility across machines.

Recorded on: 2026-05-16

## Platform

| item | value |
| --- | --- |
| OS | Windows 10.0.26200 |
| Python | `3.10.20` |
| CUDA visible to PyTorch | `None` |
| `torch.cuda.is_available()` | `False` |

The reported runs were produced on CPU-only PyTorch in this environment.

## Core Package Versions

| package | version | role |
| --- | ---: | --- |
| `numpy` | `2.2.6` | tensor/data arrays |
| `torch` | `2.10.0` | neural predecoder training/evaluation |
| `stim` | `1.15.0` | circuit/noise sampling and detector error models |
| `PyMatching` | `2.3.1` | matching decoder baseline and final decoder |
| `sinter` | `1.15.0` | Stim ecosystem utility dependency |
| `scipy` | `1.15.3` | scientific utility dependency |

Optional visualization/demo packages:

| package | version | role |
| --- | ---: | --- |
| `matplotlib` | `3.10.8` | `test.py` visualization |
| `networkx` | `3.4.2` | `test.py` graph visualization |
| `qiskit` | `2.3.0` | `test.py` coupling-map visualization |

## Install Command

For a fresh virtual environment:

```bash
python -m pip install -r requirements.txt
```

The project does not require network access to regenerate the paper-facing
summary JSON/SVG files once the existing `artifacts/` directory is present.

## Reproducibility Notes

- The final reported summary artifacts are generated from existing JSON
  artifacts under `artifacts/eval/nn/`.
- `requirements.txt` records package versions from the current working
  environment. It is intentionally minimal and does not include every transient
  dependency emitted by `pip freeze`.
- The training scripts set explicit seeds, but this repository does not enforce
  bit-identical PyTorch determinism across hardware or BLAS backends.
- For final paper consistency, run:

```bash
python tools/build_predecoder_consolidation_summary.py
python tools/build_d3_d5_robustness_summary.py
python tools/build_d3_d5_seed_expansion_ci_summary.py
python tools/build_oracle_recovery_distribution_summary.py
python tools/build_d7_validation_heldout_scatter.py
python tools/build_predecoder_figure_package.py
python tools/build_final_result_consistency_summary.py
```

Expected final result:

```text
pass: true
num_checks: 37
num_failed: 0
```
