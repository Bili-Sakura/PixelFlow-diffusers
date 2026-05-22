#!/usr/bin/env python3
"""Convert a legacy PixelFlow checkpoint to a self-contained diffusers Hub folder."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml

LIB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LIB_ROOT.parent.parent
SRC_BUNDLE = REPO_ROOT / "src" / "diffusers" / "PixelFlow"
DEFAULT_TEXT_ENCODER = "google/flan-t5-xl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PixelFlow checkpoint to diffusers-style directory.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to legacy model.pt checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--output", type=str, required=True, help="Output variant directory")
    parser.add_argument("--variant-name", type=str, default="", help="Optional variant label for metadata")
    parser.add_argument("--resolution", type=int, default=0, help="Training / inference resolution (0 = auto)")
    parser.add_argument(
        "--text-encoder",
        type=str,
        default=DEFAULT_TEXT_ENCODER,
        help="Hugging Face repo for T5 text encoder (text-to-image only)",
    )
    parser.add_argument(
        "--skip-text-encoder",
        action="store_true",
        help="Do not export text encoder/tokenizer (text-to-image only)",
    )
    return parser


def _copy_bundle(out_dir: Path) -> None:
    if not SRC_BUNDLE.exists():
        raise FileNotFoundError(f"Hub bundle not found: {SRC_BUNDLE}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copy2(SRC_BUNDLE / "pipeline.py", out_dir / "pipeline.py")

    for folder in ("transformer", "scheduler"):
        shutil.copytree(SRC_BUNDLE / folder, out_dir / folder)


def _is_text_to_image(model_cfg: dict) -> bool:
    return model_cfg.get("num_classes", 0) == 0 and model_cfg.get("cross_attention_dim") is not None


def _default_resolution(model_cfg: dict) -> int:
    return 1024 if _is_text_to_image(model_cfg) else 256


def _save_text_encoder(out_dir: Path, text_encoder_name: str) -> None:
    from transformers import T5EncoderModel, T5Tokenizer

    print(f"Exporting text encoder from {text_encoder_name} ...")
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_name)
    tokenizer = T5Tokenizer.from_pretrained(text_encoder_name)
    text_encoder.save_pretrained(str(out_dir / "text_encoder"))
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output)
    _copy_bundle(out_dir)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]["params"]
    sched_cfg = config["scheduler"]
    resolution = args.resolution or _default_resolution(model_cfg)
    is_t2i = _is_text_to_image(model_cfg)

    src_root = LIB_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from diffusers.models.transformers.transformer_pixelflow import PixelFlowTransformer2DModel
    from diffusers.pipelines.pixelflow.pipeline_pixelflow import PixelFlowPipeline
    from diffusers.schedulers.scheduling_pixelflow import PixelFlowScheduler

    transformer = PixelFlowTransformer2DModel(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        num_attention_heads=model_cfg["num_attention_heads"],
        attention_head_dim=model_cfg["attention_head_dim"],
        depth=model_cfg["depth"],
        patch_size=model_cfg["patch_size"],
        dropout=model_cfg.get("dropout", 0.0),
        cross_attention_dim=model_cfg.get("cross_attention_dim"),
        attention_bias=model_cfg.get("attention_bias", True),
        num_classes=model_cfg.get("num_classes", 0),
        sample_size=resolution,
        init_weights=False,
    )

    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    transformer.model.load_state_dict(state_dict, strict=True)

    scheduler = PixelFlowScheduler(
        num_train_timesteps=sched_cfg["num_train_timesteps"],
        num_stages=sched_cfg["num_stages"],
        gamma=-1 / 3,
    )

    pipeline = PixelFlowPipeline(transformer=transformer, scheduler=scheduler)
    pipeline.save_pretrained(str(out_dir))

    # Hub folders ship a self-contained pipeline with dynamic component loading.
    shutil.copy2(SRC_BUNDLE / "pipeline.py", out_dir / "pipeline.py")
    shutil.copytree(SRC_BUNDLE / "transformer", out_dir / "transformer", dirs_exist_ok=True)
    shutil.copytree(SRC_BUNDLE / "scheduler", out_dir / "scheduler", dirs_exist_ok=True)

    model_index = {
        "_class_name": "PixelFlowPipeline",
        "_diffusers_version": "0.36.0",
        "scheduler": ["scheduling_pixelflow", "PixelFlowScheduler"],
        "transformer": ["transformer_pixelflow", "PixelFlowTransformer2DModel"],
    }
    if is_t2i and not args.skip_text_encoder:
        model_index["text_encoder"] = ["transformers", "T5EncoderModel"]
        model_index["tokenizer"] = ["transformers", "T5Tokenizer"]
        _save_text_encoder(out_dir, args.text_encoder)

    (out_dir / "model_index.json").write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "source_config": str(Path(args.config).resolve()),
        "variant": args.variant_name or out_dir.name,
        "resolution": resolution,
        "task": "text-to-image" if is_t2i else "class-to-image",
        "model_params": model_cfg,
        "scheduler_params": sched_cfg,
    }
    if is_t2i:
        metadata["text_encoder"] = args.text_encoder
    (out_dir / "conversion_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Saved converted PixelFlow pipeline to {out_dir}")


if __name__ == "__main__":
    main()
