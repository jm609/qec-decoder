# QEC Decoder Research Scaffold

This repository contains a surface-code quantum error-correction experiment
scaffold built around Stim circuits, staged noise models, PyMatching baselines,
and neural decoder/pre-decoder experiments.

Current research focus:

> Design and Evaluation of a Transition-Aware Neural Pre-Decoder for
> Surface-Code Quantum Error Correction

The active mainline is a PyMatching-assisted neural pre-decoder: it proposes
small local syndrome edits, evaluates the downstream logical-frame transition,
and falls back to raw PyMatching when selected edits are not justified.

## Main Files

- `config.py`: experiment configuration
- `circuits.py`: ideal rotated surface-code memory circuits and detector metadata
- `sample_dataset.py`: dataset generation and logical-target schema owner
- `noise_si1000.py`, `noise_willowcore.py`: staged noise models
- `decoders/baseline_pymatching.py`: classical PyMatching baseline
- `decoders/syndrome_edit_predecoder.py`: active neural pre-decoder
- `decoders/baseline_rectcnn.py`, `decoders/research_noise_aware_3d.py`: neural baselines
- `main.tex`, `main_en.tex`: thesis manuscript drafts

Legacy neural decoder prototypes are archived under `legacy_archive/`.

## Validation

From the repository root:

```powershell
python -m py_compile circuits.py config.py sample_dataset.py
python -m unittest discover -s tests -v
python tools\check_main_tex_static.py
```

Some regression tests check generated local artifacts when they are present and
skip those checks on a clean clone.

## Artifact Policy

Generated datasets, checkpoints, evaluation summaries, figures, PDFs, and
Overleaf packages belong under `artifacts/` or as local generated files. They
are intentionally ignored by Git unless explicitly needed for a release.
