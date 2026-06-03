from __future__ import annotations

"""
multiscale_factorized_decoder.py

Multi-scale successor to the first factorized logical-frame decoder.
"""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
import argparse
import copy
import sys

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency at runtime
    torch = None
    nn = None
    F = None

try:
    import factorized_logical_frame_decoder as base
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import factorized_logical_frame_decoder as base


DECODER_NAME = "multiscale_factorized_decoder"
SCHEMA_VERSION_TRAIN = "multiscale_factorized_decoder.train.v1"
SCHEMA_VERSION_EVAL = "multiscale_factorized_decoder.eval.v1"
SCHEMA_VERSION_EXPERIMENT = "multiscale_factorized_decoder.experiment.v1"


class MultiScaleFactorizedLogicalFrameDecoder(nn.Module):
    def __init__(
        self,
        *,
        signal_channels: int,
        context_channels: int,
        hidden_channels: int,
        num_blocks: int,
        dense_hidden_dim: int,
        dropout: float,
        context_hidden_dim: int,
    ) -> None:
        super().__init__()
        if nn is None:
            raise RuntimeError("PyTorch is required to build multiscale_factorized_decoder")
        self.signal_channels = int(signal_channels)
        self.context_channels = int(context_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_blocks = int(num_blocks)
        self.valid_mask_channel_index = 1

        hidden_mid = int(hidden_channels * 2)
        hidden_deep = int(hidden_channels * 4)
        blocks_scale0 = max(1, int(num_blocks))
        blocks_scale1 = max(1, int(num_blocks // 2))
        blocks_scale2 = 1
        self.scale_block_counts = (blocks_scale0, blocks_scale1, blocks_scale2)

        self.stem = nn.Conv3d(self.signal_channels, hidden_channels, kernel_size=3, padding=1)
        self.scale0_blocks = nn.ModuleList(
            [base.FiLMResidualBlock(hidden_channels, dropout=dropout) for _ in range(blocks_scale0)]
        )

        self.down1 = nn.Conv3d(hidden_channels, hidden_mid, kernel_size=3, stride=2, padding=1)
        self.scale1_blocks = nn.ModuleList(
            [base.FiLMResidualBlock(hidden_mid, dropout=dropout) for _ in range(blocks_scale1)]
        )

        self.down2 = nn.Conv3d(hidden_mid, hidden_deep, kernel_size=3, stride=2, padding=1)
        self.scale2_blocks = nn.ModuleList(
            [base.FiLMResidualBlock(hidden_deep, dropout=dropout) for _ in range(blocks_scale2)]
        )

        self.context_mlp = nn.Sequential(
            nn.Linear(self.context_channels, context_hidden_dim),
            nn.ReLU(),
            nn.Linear(context_hidden_dim, context_hidden_dim),
            nn.ReLU(),
        )
        self.scale0_film = nn.Linear(context_hidden_dim, blocks_scale0 * hidden_channels * 2)
        self.scale1_film = nn.Linear(context_hidden_dim, blocks_scale1 * hidden_mid * 2)
        self.scale2_film = nn.Linear(context_hidden_dim, blocks_scale2 * hidden_deep * 2)

        self.scale0_proj = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1)
        self.scale1_proj = nn.Conv3d(hidden_mid, hidden_channels, kernel_size=1)
        self.scale2_proj = nn.Conv3d(hidden_deep, hidden_channels, kernel_size=1)
        self.fuse = nn.Conv3d(hidden_channels * 3, hidden_channels, kernel_size=1)

        groups = 4 if hidden_channels % 4 == 0 else 1
        self.head_norm = nn.GroupNorm(groups, hidden_channels)
        self.head_relu = nn.ReLU()
        self.head_drop = nn.Dropout(dropout)
        self.shared_fc = nn.Linear(hidden_channels, dense_hidden_dim)
        self.x_head = nn.Linear(dense_hidden_dim, 1)
        self.z_head = nn.Linear(dense_hidden_dim, 1)
        self.non_identity_head = nn.Linear(dense_hidden_dim, 1)
        self.error_head = nn.Linear(dense_hidden_dim, 1)
        self.residual_head = nn.Linear(dense_hidden_dim, 4)

    def _masked_mean(self, h: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = valid_mask.to(dtype=h.dtype)
        denom = weights.sum(dim=(2, 3, 4)).clamp_min(1.0)
        return (h * weights).sum(dim=(2, 3, 4)) / denom

    def _apply_block_stack(
        self,
        h: torch.Tensor,
        blocks: nn.ModuleList,
        film_params: torch.Tensor,
        channels: int,
    ) -> torch.Tensor:
        film = film_params.view(h.shape[0], len(blocks), 2, channels)
        for block_index, block in enumerate(blocks):
            h = block(h, gamma=film[:, block_index, 0, :], beta=film[:, block_index, 1, :])
        return h

    def _resize_to(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if list(h.shape[2:]) == list(target.shape[2:]):
            return h
        return F.interpolate(h, size=target.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        signal = x[:, : self.signal_channels, :, :, :]
        context_vec = x[:, self.signal_channels :, 0, 0, 0]
        valid_mask = signal[:, self.valid_mask_channel_index : self.valid_mask_channel_index + 1, :, :, :]

        context = self.context_mlp(context_vec)

        h0 = self.stem(signal)
        h0 = self._apply_block_stack(
            h0,
            self.scale0_blocks,
            self.scale0_film(context),
            self.hidden_channels,
        )

        h1 = self.down1(h0)
        h1 = self._apply_block_stack(
            h1,
            self.scale1_blocks,
            self.scale1_film(context),
            self.hidden_channels * 2,
        )

        h2 = self.down2(h1)
        h2 = self._apply_block_stack(
            h2,
            self.scale2_blocks,
            self.scale2_film(context),
            self.hidden_channels * 4,
        )

        p0 = self.scale0_proj(h0)
        p1 = self._resize_to(self.scale1_proj(h1), h0)
        p2 = self._resize_to(self.scale2_proj(h2), h0)
        fused = self.fuse(torch.cat([p0, p1, p2], dim=1))
        fused = self.head_relu(self.head_norm(fused))
        pooled = self._masked_mean(fused, valid_mask)
        shared = self.head_drop(self.head_relu(self.shared_fc(pooled)))

        x_logits = self.x_head(shared).squeeze(1)
        z_logits = self.z_head(shared).squeeze(1)
        non_identity_logits = self.non_identity_head(shared).squeeze(1)
        error_logits = self.error_head(shared).squeeze(1)
        residual_logits = self.residual_head(shared)
        zeros = torch.zeros_like(x_logits)
        base_logits = torch.stack([zeros, x_logits, z_logits, x_logits + z_logits], dim=1)
        class4_logits = base_logits + residual_logits
        return {
            "x_logits": x_logits,
            "z_logits": z_logits,
            "non_identity_logits": non_identity_logits,
            "error_logits": error_logits,
            "base_class4_logits": base_logits,
            "residual_logits": residual_logits,
            "class4_logits": class4_logits,
        }


@contextmanager
def _patched_base() -> Any:
    original_model = base.FactorizedLogicalFrameDecoder
    original_train_schema = base.SCHEMA_VERSION_TRAIN
    original_eval_schema = base.SCHEMA_VERSION_EVAL
    original_experiment_schema = base.SCHEMA_VERSION_EXPERIMENT
    base.FactorizedLogicalFrameDecoder = MultiScaleFactorizedLogicalFrameDecoder
    base.SCHEMA_VERSION_TRAIN = SCHEMA_VERSION_TRAIN
    base.SCHEMA_VERSION_EVAL = SCHEMA_VERSION_EVAL
    base.SCHEMA_VERSION_EXPERIMENT = SCHEMA_VERSION_EXPERIMENT
    try:
        yield
    finally:
        base.FactorizedLogicalFrameDecoder = original_model
        base.SCHEMA_VERSION_TRAIN = original_train_schema
        base.SCHEMA_VERSION_EVAL = original_eval_schema
        base.SCHEMA_VERSION_EXPERIMENT = original_experiment_schema


def _patch_checkpoint_file(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint["schema_version"] = SCHEMA_VERSION_TRAIN
    checkpoint["decoder"] = DECODER_NAME
    model_hparams = dict(checkpoint.get("model_hparams", {}))
    model_hparams["architecture"] = DECODER_NAME
    checkpoint["model_hparams"] = model_hparams
    torch.save(checkpoint, path)


def _patch_json_file(path: Path, *, schema_version: str) -> None:
    payload = base.common._read_json(path)
    payload["schema_version"] = schema_version
    payload["decoder"] = DECODER_NAME
    if isinstance(payload.get("model"), dict):
        payload["model"]["architecture"] = DECODER_NAME
    if isinstance(payload.get("checkpoint"), dict):
        payload["checkpoint"]["decoder"] = DECODER_NAME
    if isinstance(payload.get("training"), dict) and isinstance(payload["training"].get("model_hparams"), dict):
        payload["training"]["model_hparams"]["architecture"] = DECODER_NAME
    base.common._write_json(path, payload)


def _patch_experiment_outputs(out_dir: Path) -> None:
    checkpoint_path = out_dir / "checkpoint.pt"
    train_json_path = out_dir / "train.json"
    experiment_summary_path = out_dir / "experiment_summary.json"
    if checkpoint_path.exists():
        _patch_checkpoint_file(checkpoint_path)
    if train_json_path.exists():
        _patch_json_file(train_json_path, schema_version=SCHEMA_VERSION_TRAIN)
    for eval_json_path in sorted(out_dir.glob("eval__*.json")):
        _patch_json_file(eval_json_path, schema_version=SCHEMA_VERSION_EVAL)
    if experiment_summary_path.exists():
        payload = base.common._read_json(experiment_summary_path)
        payload["schema_version"] = SCHEMA_VERSION_EXPERIMENT
        payload["decoder"] = DECODER_NAME
        base.common._write_json(experiment_summary_path, payload)


def train_family_dir(**kwargs: Any) -> base.common.TrainResult:
    checkpoint_out = Path(kwargs["checkpoint_out"])
    train_json_out = kwargs.get("train_json_out")
    train_json_out = Path(train_json_out) if train_json_out is not None else None
    with _patched_base():
        result = base.train_family_dir(**kwargs)
    _patch_checkpoint_file(checkpoint_out)
    if train_json_out is not None and train_json_out.exists():
        _patch_json_file(train_json_out, schema_version=SCHEMA_VERSION_TRAIN)
    return replace(
        result,
        schema_version=SCHEMA_VERSION_TRAIN,
        decoder=DECODER_NAME,
        model={**result.model, "architecture": DECODER_NAME},
    )


def evaluate_checkpoint_on_family(**kwargs: Any) -> base.common.EvalResult:
    eval_json_out = kwargs.get("eval_json_out")
    eval_json_out = Path(eval_json_out) if eval_json_out is not None else None
    with _patched_base():
        result = base.evaluate_checkpoint_on_family(**kwargs)
    if eval_json_out is not None and eval_json_out.exists():
        _patch_json_file(eval_json_out, schema_version=SCHEMA_VERSION_EVAL)
    return replace(
        result,
        schema_version=SCHEMA_VERSION_EVAL,
        decoder=DECODER_NAME,
        model={**result.model, "architecture": DECODER_NAME},
        checkpoint={**result.checkpoint, "decoder": DECODER_NAME},
    )


def run_manifest_experiment(**kwargs: Any) -> dict[str, Any]:
    out_dir = Path(kwargs["out_dir"])
    with _patched_base():
        summary = base.run_manifest_experiment(**kwargs)
    _patch_experiment_outputs(out_dir)
    summary = copy.deepcopy(summary)
    summary["schema_version"] = SCHEMA_VERSION_EXPERIMENT
    summary["decoder"] = DECODER_NAME
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate the multi-scale factorized logical-frame decoder.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    base.common._add_common_input_args(train_parser)
    train_parser.add_argument("--checkpoint-out", type=Path, required=True)
    train_parser.add_argument("--train-json-out", type=Path, default=None)
    train_parser.add_argument("--fill-value", type=float, default=-0.5)
    train_parser.add_argument("--max-shots", type=int, default=None)
    train_parser.add_argument("--train-ratio", type=float, default=0.8)
    train_parser.add_argument("--val-ratio", type=float, default=0.1)
    train_parser.add_argument("--test-ratio", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=12345)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--hidden-channels", type=int, default=24)
    train_parser.add_argument("--num-blocks", type=int, default=3)
    train_parser.add_argument("--dense-hidden-dim", type=int, default=64)
    train_parser.add_argument("--context-hidden-dim", type=int, default=64)
    train_parser.add_argument("--main-axis-loss-weight", type=float, default=0.25)
    train_parser.add_argument("--non-identity-loss-weight", type=float, default=0.0)
    train_parser.add_argument("--confidence-loss-weight", type=float, default=0.0)
    train_parser.add_argument("--aux-loss-weight", type=float, default=0.5)
    train_parser.add_argument("--imbalance-mode", type=str, choices=base.IMBALANCE_MODE_CHOICES, default=base.IMBALANCE_MODE_TEMPERED)
    train_parser.add_argument("--main-class4-loss", type=str, choices=base.MAIN_CLASS4_LOSS_CHOICES, default=base.MAIN_CLASS4_LOSS_CROSS_ENTROPY)
    train_parser.add_argument("--focal-gamma", type=float, default=base.DEFAULT_FOCAL_GAMMA)
    train_parser.add_argument("--aux-dual-axis-manifest", type=Path, default=None)
    train_parser.add_argument("--device", type=str, default="auto")

    eval_parser = subparsers.add_parser("eval")
    base.common._add_common_input_args(eval_parser)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--eval-json-out", type=Path, default=None)
    eval_parser.add_argument("--max-shots", type=int, default=None)
    eval_parser.add_argument("--batch-size", type=int, default=128)
    eval_parser.add_argument("--device", type=str, default="auto")

    experiment_parser = subparsers.add_parser("experiment")
    experiment_parser.add_argument("--manifest", type=Path, required=True)
    experiment_parser.add_argument("--train-families", nargs="+", required=True)
    experiment_parser.add_argument("--eval-families", nargs="+", default=None)
    experiment_parser.add_argument("--out-dir", type=Path, required=True)
    experiment_parser.add_argument("--fill-value", type=float, default=-0.5)
    experiment_parser.add_argument("--max-shots", type=int, default=None)
    experiment_parser.add_argument("--train-ratio", type=float, default=0.8)
    experiment_parser.add_argument("--val-ratio", type=float, default=0.1)
    experiment_parser.add_argument("--test-ratio", type=float, default=0.1)
    experiment_parser.add_argument("--seed", type=int, default=12345)
    experiment_parser.add_argument("--epochs", type=int, default=20)
    experiment_parser.add_argument("--batch-size", type=int, default=128)
    experiment_parser.add_argument("--lr", type=float, default=1e-3)
    experiment_parser.add_argument("--weight-decay", type=float, default=1e-4)
    experiment_parser.add_argument("--dropout", type=float, default=0.1)
    experiment_parser.add_argument("--hidden-channels", type=int, default=24)
    experiment_parser.add_argument("--num-blocks", type=int, default=3)
    experiment_parser.add_argument("--dense-hidden-dim", type=int, default=64)
    experiment_parser.add_argument("--context-hidden-dim", type=int, default=64)
    experiment_parser.add_argument("--main-axis-loss-weight", type=float, default=0.25)
    experiment_parser.add_argument("--non-identity-loss-weight", type=float, default=0.0)
    experiment_parser.add_argument("--confidence-loss-weight", type=float, default=0.0)
    experiment_parser.add_argument("--aux-loss-weight", type=float, default=0.5)
    experiment_parser.add_argument("--imbalance-mode", type=str, choices=base.IMBALANCE_MODE_CHOICES, default=base.IMBALANCE_MODE_TEMPERED)
    experiment_parser.add_argument("--main-class4-loss", type=str, choices=base.MAIN_CLASS4_LOSS_CHOICES, default=base.MAIN_CLASS4_LOSS_CROSS_ENTROPY)
    experiment_parser.add_argument("--focal-gamma", type=float, default=base.DEFAULT_FOCAL_GAMMA)
    experiment_parser.add_argument("--aux-dual-axis-manifest", type=Path, default=None)
    experiment_parser.add_argument("--aux-train-families", nargs="+", default=None)
    experiment_parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        result = train_family_dir(
            family_dir=args.family_dir,
            manifest=args.manifest,
            family=args.family,
            checkpoint_out=args.checkpoint_out,
            train_json_out=args.train_json_out,
            fill_value=args.fill_value,
            max_shots=args.max_shots,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
            dense_hidden_dim=args.dense_hidden_dim,
            context_hidden_dim=args.context_hidden_dim,
            main_axis_loss_weight=args.main_axis_loss_weight,
            non_identity_loss_weight=args.non_identity_loss_weight,
            confidence_loss_weight=args.confidence_loss_weight,
            aux_loss_weight=args.aux_loss_weight,
            imbalance_mode=args.imbalance_mode,
            main_class4_loss=args.main_class4_loss,
            focal_gamma=args.focal_gamma,
            aux_dual_axis_manifest=args.aux_dual_axis_manifest,
            device_arg=args.device,
        )
        print(base.json.dumps(result.to_dict(), indent=2))
        return
    if args.mode == "eval":
        result = evaluate_checkpoint_on_family(
            family_dir=args.family_dir,
            manifest=args.manifest,
            family=args.family,
            checkpoint_path=args.checkpoint,
            eval_json_out=args.eval_json_out,
            max_shots=args.max_shots,
            batch_size=args.batch_size,
            device_arg=args.device,
        )
        print(base.json.dumps(result.to_dict(), indent=2))
        return
    summary = run_manifest_experiment(
        manifest=args.manifest,
        train_families=args.train_families,
        eval_families=args.eval_families,
        out_dir=args.out_dir,
        fill_value=args.fill_value,
        max_shots=args.max_shots,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        dense_hidden_dim=args.dense_hidden_dim,
        context_hidden_dim=args.context_hidden_dim,
        main_axis_loss_weight=args.main_axis_loss_weight,
        non_identity_loss_weight=args.non_identity_loss_weight,
        confidence_loss_weight=args.confidence_loss_weight,
        aux_loss_weight=args.aux_loss_weight,
        imbalance_mode=args.imbalance_mode,
        main_class4_loss=args.main_class4_loss,
        focal_gamma=args.focal_gamma,
        aux_dual_axis_manifest=args.aux_dual_axis_manifest,
        aux_train_families=args.aux_train_families,
        device_arg=args.device,
    )
    print(base.json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
