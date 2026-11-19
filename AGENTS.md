# Repository Guidelines

## Project Structure & Module Organization
Start thesis work with `README.md` and the published research index in `docs/thesis/README.md`. When available, `docs/thesis/PREDECODER_CLEAN_HANDOFF.md` is a local handoff, not a required checkout file. The separate Willow review lives in `qec_below_threshold_review/`, and detector RTL work lives in `qec_detector_rtl/`.

Core experiment code lives at the repository root. Use `config.py` for experiment configuration, `circuits.py` for ideal circuit generation, `noise_si1000.py` and `noise_willowcore.py` for noise injection, and `sample_dataset.py` for dataset creation. Active decoder implementations are under `decoders/` (`baseline_pymatching.py`, `baseline_rectcnn.py`, `research_noise_aware_3d.py`). Archived legacy decoders live under `legacy_archive/`. Inspection utilities live in `inspect_samples.py` and `tools/`. Generated datasets, checkpoints, and evaluation JSON files belong in `artifacts/` and should not be treated as hand-edited source files.

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
The `tests/` suite covers source invariants and optional local-artifact regressions. Validate changes with:

- `python -m py_compile ...` for syntax
- `python -m unittest discover -s tests -v` for regression checks
- targeted script runs for the modified path
- inspection of emitted `metadata.json` or eval JSON when behavior changes

Place new tests in `tests/` and name files `test_*.py`. Artifact-dependent checks may skip on a clean clone. RTL and shell checks also require their local simulator or shell.

## Commit & Pull Request Guidelines
Recent commit messages are short summary phrases such as `minor change`, `nn baseline`, and `data extraction change`. Keep commits concise but more specific than the current history when possible, for example: `track_common: deduplicate track tensor builders`. PRs should include the purpose, changed scripts/modules, exact validation commands run, and any affected artifact paths or schema changes.

## Data & Artifact Handling
Do not overwrite published outputs in `artifacts/` without calling it out. When changing dataset schema, update downstream readers and note compatibility impact in the PR.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
