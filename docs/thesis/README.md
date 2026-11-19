# Research index

> Candidate Quality and Logical-Transition Selection in
> Neural Pre-Decoding for Surface Codes

The current [English](../../main_d7_en.tex) and
[Korean](../../main_d7_ko.tex) manuscripts supersede the earlier root drafts.
The English manuscript is independently composed from the experimental
evidence. See the [Overleaf guide](OVERLEAF_COMPILE_GUIDE.md) for packaging.

## Main findings

| Experiment | Finding | Interpretation |
| --- | --- | --- |
| Corrected-input validation | d3 +2.260 pp, d5 +0.022 pp, d7 0.000 pp | All eight d7 family gates returned raw PyMatching |
| d7 top-1, strong ZZ | 10,501 repairs and 25,126 harms | Harmful logical transitions dominate the observed loss |
| Candidate control, budget 32 | Coverage 88.70% learned vs 77.63% local, difference +11.07 pp, 95% interval [+7.63, +14.51] | Learned candidates outperform the tested rule in oracle coverage |
| Frozen margin, independent strong ZZ | −0.035 pp, 97.5% interval [−0.082, +0.013] | Additional accuracy is inconclusive |
| Gate before search | Mean batch-time saving 119.1 ms; raw-relative gain +0.028 pp, 95% interval [−0.148, +0.205] | Time condition passed, joint adoption rule failed |
| Existing margin at Pauli bias 20 | +0.0041 pp, 95% interval [−0.0859, +0.0940] | Additional accuracy is inconclusive |
| Biased-data head retraining | +0.0183 pp, 95% interval [−0.0702, +0.1068] | Retraining the linear head did not establish added accuracy |

Differences are in percentage points. Positive values mean lower logical
error rate. Candidate coverage uses truth and is not an inference policy.
Intervals describe map variation conditional on the fixed models. Pauli bias
preserves total probability only in affected data-qubit single-qubit channels.

## Methods and research integrity

- [Method description](PREDECODER_METHOD_DESCRIPTION.md)
- [Input audit](PREDECODER_RESEARCH_INTEGRITY_AUDIT.md)
- [Independent revalidation](PREDECODER_INDEPENDENT_REVALIDATION.md)
- [Noise-shift experiment plan](PREDECODER_NOISE_SHIFT_EXPERIMENT_PLAN.md)
- [Manuscript evidence guide](PREDECODER_PAPER_EVIDENCE.md)

The old input exposed dataset-level label aggregates. That channel is now
zeroed and the current manuscript reports retrained models. Earlier reports
remain a dated research record. Preliminary positive estimates do not
supersede the final frozen independent comparisons.

## Protocols and results

| Stage | Protocol | Result |
| --- | --- | --- |
| Failure diagnosis | [Plan](PREDECODER_NOISE_SHIFT_EXPERIMENT_PLAN.md) | [Transitions](results/PREDECODER_D7_LOGICAL_TRANSITION_DIAGNOSTIC.md) |
| Confidence selection | [Protocol](PREDECODER_D7_CONFIDENCE_DEVELOPMENT_PROTOCOL.md) | [Results](results/PREDECODER_D7_CONFIDENCE_DEVELOPMENT_RESULTS.md) |
| Frozen evaluation | [Protocol](PREDECODER_D7_FINAL_EVALUATION_PROTOCOL.md) | [Results](results/PREDECODER_D7_FINAL_EVALUATION_RESULTS.md) |
| Candidate control | [Protocol](PREDECODER_D7_CANDIDATE_GENERATOR_PROTOCOL.md) | [Results](results/PREDECODER_D7_CANDIDATE_GENERATOR_RESULTS.md) |
| Strong-ZZ selection model | [Protocol](PREDECODER_D7_ADOPTION_PROTOCOL.md) | [Results](results/PREDECODER_D7_ADOPTION_RESULTS.md) |
| Gate before search | [Protocol](PREDECODER_D7_PRE_GATE_PROTOCOL.md) | [Results](results/PREDECODER_D7_PRE_GATE_RESULTS.md) |
| Pauli-bias pilot | [Protocol](PREDECODER_D7_PAULI_BIAS_PROTOCOL.md) | [Results](results/PREDECODER_D7_PAULI_BIAS_PILOT_RESULTS.md) |
| Frozen biased margin | [Protocol](PREDECODER_D7_PAULI_BIAS_INDEPENDENT_PROTOCOL.md) | [Results](results/PREDECODER_D7_PAULI_BIAS_INDEPENDENT_RESULTS.md) |
| Biased head retraining | [Protocol](PREDECODER_D7_PAULI_BIAS_ADOPTION_PROTOCOL.md) | [Results](results/PREDECODER_D7_PAULI_BIAS_ADOPTION_RESULTS.md) |

Additional diagnostics are in `results/`. Linked prediction arrays,
checkpoints, and run JSONs are local generated artifacts excluded from Git.
The source and protocols support regeneration, but a clone does not contain
the original training datasets or model weights.

## Data separation

The candidate panel, maps 788200–788223, was reused as development data for the
later selection model and gate. Their independent panels are 788300–788323 and
788400–788423, respectively.

The biased margin uses independent maps 788600–788623. Later retraining uses
training maps 788700–788731, development maps 788740–788747, and independent
maps 788800–788823. Old and retrained policies share the shots in the last panel.

## Related workspaces

- [Willow review and public-data reanalysis](../../qec_below_threshold_review/README.md)
- [Measurement-record-to-detector RTL](../../qec_detector_rtl/README.md)

These projects have separate questions and validation scopes. Detector RTL
results do not establish an accuracy gain for the neural pre-decoder.
