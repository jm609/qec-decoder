# Repository Guidelines

## Project Structure & Module Organization
Core experiment code lives at the repository root. Use `config.py` for experiment configuration, `circuits.py` for ideal circuit generation, `noise_si1000.py` and `noise_willowcore.py` for noise injection, and `sample_dataset.py` for dataset creation. Decoder implementations are under `decoders/` (`baseline_pymatching.py`, `baseline_nn.py`, `baseline_tracknn.py`, `baseline_trackformer.py`, plus shared helpers in `track_common.py`). Inspection utilities live in `inspect_samples.py` and `tools/`. Generated datasets, checkpoints, and evaluation JSON files belong in `artifacts/` and should not be treated as hand-edited source files.

## Build, Test, and Development Commands
Run scripts directly with Python from the repo root.

```bash
python sample_dataset.py --out-root artifacts/datasets/dev --distance 5 --rounds 10 --basis z --shots 1000
python inspect_samples.py --manifest artifacts/datasets/dev/manifest.json
python decoders/baseline_pymatching.py --manifest artifacts/datasets/dev/manifest.json
python -m py_compile circuits.py config.py sample_dataset.py decoders/*.py tools/*.py
```

Use `py_compile` as the minimum syntax check before submitting changes. Prefer adding small reproducible CLI examples to PR descriptions when changing dataset or decoder behavior.

## Coding Style & Naming Conventions
Use 4-space indentation and keep code Pythonic and explicit. Prefer type hints, frozen dataclasses for structured payloads, and `snake_case` for functions, variables, and file names. Keep CLI entry points behind `if __name__ == "__main__":`. Reuse shared utilities instead of duplicating tensor/layout logic across decoders. There is no configured formatter in this repo, so keep diffs small and style-consistent with surrounding code.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Validate changes with:

- `python -m py_compile ...` for syntax
- targeted script runs for the modified path
- inspection of emitted `metadata.json` or eval JSON when behavior changes

If you add tests, place them in a new `tests/` directory and name files `test_*.py`.

## Commit & Pull Request Guidelines
Recent commit messages are short summary phrases such as `minor change`, `nn baseline`, and `data extraction change`. Keep commits concise but more specific than the current history when possible, for example: `track_common: deduplicate track tensor builders`. PRs should include the purpose, changed scripts/modules, exact validation commands run, and any affected artifact paths or schema changes.

## Data & Artifact Handling
Do not overwrite published outputs in `artifacts/` without calling it out. When changing dataset schema, update downstream readers and note compatibility impact in the PR.
