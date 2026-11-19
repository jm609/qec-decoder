# Surface-Code Pre-Decoding Research

Stim circuits, PyMatching baselines, and neural syndrome-edit experiments for
surface-code logical-frame inference.

> Candidate Quality and Logical-Transition Selection in
> Neural Pre-Decoding for Surface Codes

The pre-decoder proposes local detector edits and compares the original and
edited PyMatching predictions on the same shot. The study separates correct
candidate coverage from repairs and harms caused by an actual selection policy.

## Current research

- [English manuscript](main_d7_en.tex), independently written from the evidence.
- [Korean manuscript](main_d7_ko.tex), reporting the same study.
- [Research index](docs/thesis/README.md), with protocols, results, and limits.
- [Overleaf guide](docs/thesis/OVERLEAF_COMPILE_GUIDE.md), with packaging commands.

Corrected-input validation found accuracy gains of +2.260 percentage points at
distance 3 and +0.022 at distance 5. All eight distance-7 family gates returned
the original decoder.

At distance 7 and a budget of 32 nonzero candidates, learned generation covered
88.70% of original errors with a correct candidate versus 77.63% for the tested
local rule. That oracle advantage did not establish a gain for the evaluated
selection policies. The frozen margin's independent strong-ZZ difference was
−0.035 percentage points, with a 97.5% map-level interval of [−0.082, +0.013].
Independent Pauli-bias follow-ups also left additional gains inconclusive.

An input audit found dataset-level label aggregates in an old context channel.
That slot is now zeroed without changing tensor width. Current manuscript
results use retrained models. Legacy checkpoints require retraining;
evaluating them with the zeroed slot is a sensitivity check.

`main.tex`, `main_en.tex`, and `main.bib` are earlier manuscript versions.
Use `main_d7_en.tex` or `main_d7_ko.tex` for the current study.

## Run locally

Use Anaconda `fp` or an equivalent environment with the pinned dependencies.
Run commands from the repository root.

```powershell
conda activate fp
python -m pip install -r requirements.txt
python sample_dataset.py --out-root artifacts/datasets/dev --distance 5 --rounds 10 --basis z --shots 1000
python inspect_samples.py --manifest artifacts/datasets/dev/manifest.json
python decoders/baseline_pymatching.py --manifest artifacts/datasets/dev/manifest.json
```

Optional legacy PDF rendering scripts also require ReportLab. The current TeX
manuscripts do not require it.

## Repository layout

| Location | Purpose |
| --- | --- |
| `config.py`, `circuits.py` | Settings and ideal circuits |
| `noise_si1000.py`, `noise_willowcore.py` | Noise injection and data-qubit Pauli bias |
| `sample_dataset.py`, `logical_*.py` | Sampling and logical-target conventions |
| `decoders/` | PyMatching and neural decoder implementations |
| `tools/` | Experiments, diagnostics, statistics, figures, and packaging |
| `tests/` | Source, regression, and optional artifact checks |
| `docs/thesis/` | Current research protocols and result reports |
| `qec_below_threshold_review/` | Separate Willow review and public-data reanalysis |
| `qec_detector_rtl/` | Separate measurement-record-to-detector RTL follow-up |

The RTL work has local Python and Icarus validation. Server VCS/Verdi checks
and synthesis remain pending. It does not establish end-to-end real-time QEC.

## Validation

```powershell
python -m compileall -q circuits.py config.py sample_dataset.py decoders tools tests qec_detector_rtl qec_below_threshold_review
python -m unittest discover -s tests -v
```

Artifact-dependent checks skip when generated data are absent. RTL and shell
integration checks require their local tools. Windows sandbox restrictions
may prevent Icarus or Git Bash from starting. Server-shell tests use local
stubs and do not connect to a server.

## Data and generated outputs

Datasets, predictions, checkpoints, timing logs, waveforms, and ZIP packages
belong under `artifacts/` and are excluded from Git. The five frozen manuscript
PDF figures are an explicit publication exception and allow manuscript
packaging on a fresh clone.

Reports retain links to local run artifacts for provenance. Full experiment
reproduction requires the corresponding datasets and trained checkpoints, or
rerunning the documented development and evaluation procedures. A clone does
not contain the original model weights or training datasets.

Local thesis handoffs and planning archives, downloaded papers, presentation
binaries, credentials, and scratch files remain outside the published source.
