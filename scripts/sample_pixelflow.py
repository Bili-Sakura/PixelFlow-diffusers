#!/usr/bin/env python3
"""Sample images from a converted PixelFlow diffusers Hub folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample images from a PixelFlow diffusers pipeline.")
    parser.add_argument("--model", type=str, required=True, help="Path to converted variant directory")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    parser.add_argument("--class-label", type=int, default=207, help="ImageNet class id (class-to-image)")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt (text-to-image)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10, help="Inference steps per cascade stage")
    parser.add_argument("--cfg", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--shift", type=float, default=1.0, help="Noise shift for scheduler")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = Path(args.model).resolve()
    sys.path.insert(0, str(model_path))

    metadata_path = model_path / "conversion_metadata.json"
    model_index_path = model_path / "model_index.json"
    is_t2i = False
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        is_t2i = metadata.get("task") == "text-to-image"
    elif model_index_path.exists():
        model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
        is_t2i = model_index.get("_class_name") == "PixelFlowT2IPipeline"

    if is_t2i:
        from pipeline import PixelFlowT2IPipeline as PipelineCls
    else:
        from pipeline import PixelFlowPipeline as PipelineCls

    pipe = PipelineCls.from_pretrained(str(model_path)).to(args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    num_stages = pipe.scheduler.num_stages
    call_kwargs = {
        "num_inference_steps": [args.steps] * num_stages,
        "guidance_scale": args.cfg,
        "shift": args.shift,
        "generator": generator,
        "output_type": "pil",
    }

    if is_t2i:
        call_kwargs["prompt"] = args.prompt or "A golden retriever playing in a sunny garden"
    else:
        call_kwargs["class_labels"] = [args.class_label]

    output = pipe(**call_kwargs)
    image = output.images[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved image to: {output_path}")


if __name__ == "__main__":
    main()
