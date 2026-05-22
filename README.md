# PixelFlow Diffusers Refactor

Diffusers-style PixelFlow implementation (ADM-style layout):

| Path | Purpose |
| --- | --- |
| `src/diffusers/models/transformers/` | `PixelFlowTransformer2DModel`, `PixelFlowModel` |
| `src/diffusers/schedulers/` | `PixelFlowScheduler` |
| `src/diffusers/pipelines/pixelflow/` | `PixelFlowPipeline`, `PixelFlowPipelineOutput` |
| `scripts/convert_pixelflow_to_diffusers.py` | Convert legacy checkpoints |
| `scripts/sample_pixelflow.py` | Sample from converted Hub folders |

Hub bundles built from this lib live at [`src/diffusers/PixelFlow/`](../../src/diffusers/PixelFlow/). Converted checkpoints: [`models/BiliSakura/PixelFlow-diffusers/`](../../models/BiliSakura/PixelFlow-diffusers/).

## Install

```bash
pip install -e libs/PixelFlow-diffusers
```

This installs the conversion and sampling CLI scripts. Inference uses self-contained Hub folders produced by the converter.

## Convert a legacy checkpoint

Class-to-image (256×256):

```bash
python scripts/convert_pixelflow_to_diffusers.py \
  --checkpoint models/raw/PixelFlow/c2i/model.pt \
  --config models/raw/PixelFlow/c2i/config.yaml \
  --output models/BiliSakura/PixelFlow-diffusers/PixelFlow-256
```

Text-to-image (1024×1024, T5 loaded at runtime unless exported):

```bash
python scripts/convert_pixelflow_to_diffusers.py \
  --checkpoint models/raw/PixelFlow/t2i/model.pt \
  --config models/raw/PixelFlow/t2i/config.yaml \
  --output models/BiliSakura/PixelFlow-diffusers/PixelFlow-T2I \
  --skip-text-encoder
```

To bundle `google/flan-t5-xl` into the output folder, omit `--skip-text-encoder`.

## Sample images

Class-to-image:

```bash
python scripts/sample_pixelflow.py \
  --model models/BiliSakura/PixelFlow-diffusers/PixelFlow-256 \
  --output demo.png \
  --class-label 207 \
  --steps 10 \
  --cfg 4.0
```

Text-to-image:

```bash
python scripts/sample_pixelflow.py \
  --model models/BiliSakura/PixelFlow-diffusers/PixelFlow-T2I \
  --output demo.png \
  --prompt "A golden retriever playing in a sunny garden" \
  --steps 10 \
  --cfg 4.0
```

## Load from Python

Development (lib package):

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("libs/PixelFlow-diffusers/src")))
from diffusers.pipelines.pixelflow.pipeline_pixelflow import PixelFlowPipeline

pipe = PixelFlowPipeline.from_pretrained("models/BiliSakura/PixelFlow-diffusers/PixelFlow-256")
pipe.to("cuda")
```

Converted Hub folders (self-contained, no lib install):

```python
import sys
from pathlib import Path

variant = Path("models/BiliSakura/PixelFlow-diffusers/PixelFlow-256")
sys.path.insert(0, str(variant))
from pipeline import PixelFlowPipeline

pipe = PixelFlowPipeline.from_pretrained(".")
pipe.to("cuda")

images = pipe(
    class_labels=207,
    num_inference_steps=[10, 10, 10, 10],
    guidance_scale=4.0,
).images
```

## Notes

- Hub folders are self-contained: each variant ships `pipeline.py`, component code, and weights.
- Text-to-image variants use [`google/flan-t5-xl`](https://huggingface.co/google/flan-t5-xl) unless `text_encoder/` is exported during conversion.
- Upstream training code: [ShoufaChen/PixelFlow](https://github.com/ShoufaChen/PixelFlow).
