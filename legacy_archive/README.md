This directory contains archived legacy decoder code that is no longer part of
the rebuilt mainline.

Contents
- `decoders/baseline_nn.py`
- `decoders/baseline_tracknn.py`
- `decoders/baseline_trackformer.py`
- `decoders/track_common.py`
- `baseline_trackformer_eventcentric_v3_fast.py`

Status
- Kept only for historical comparison and ablation reference.
- Not used by the current mainline dataset, baseline, or experiment runners.
- Not expected to stay path-stable while the rebuilt decoder stack evolves.
