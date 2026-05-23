# PixelFlow Diffusers Refactor

Diffusers-style PixelFlow implementation:

| Path | Purpose |
| --- | --- |
| `src/diffusers/models/transformers/` | `PixelFlowTransformer2DModel`, `PixelFlowModel` |
| `src/diffusers/schedulers/` | `PixelFlowScheduler` |
| `src/diffusers/pipelines/pixelflow/pipeline_pixelflow.py` | `PixelFlowPipeline` (class-conditional) |
| `src/diffusers/pipelines/pixelflow/pipeline_pixelflow_t2i.py` | `PixelFlowT2IPipeline` (text-to-image) |
| `scripts/convert_pixelflow_to_diffusers.py` | Convert legacy checkpoints |
| `scripts/sample_pixelflow.py` | Sample from converted Hub folders |

Hub bundles built from this lib live at:

- [`src/diffusers/PixelFlow/`](../../src/diffusers/PixelFlow/) — class-to-image template
- [`src/diffusers/PixelFlow-T2I/`](../../src/diffusers/PixelFlow-T2I/) — text-to-image template

Converted checkpoints: [`models/BiliSakura/PixelFlow-diffusers/`](../../models/BiliSakura/PixelFlow-diffusers/).

Each pipeline file is self-contained (no shared helper module). Hub variant folders are also self-contained at inference time.

## Install

```bash
pip install -e libs/PixelFlow-diffusers
```

## Convert a legacy checkpoint

Class-to-image (256×256):

```bash
python libs/PixelFlow-diffusers/scripts/convert_pixelflow_to_diffusers.py \
  --checkpoint models/raw/PixelFlow/c2i/model.pt \
  --config models/raw/PixelFlow/c2i/config.yaml \
  --output models/BiliSakura/PixelFlow-diffusers/PixelFlow-256
```

Text-to-image (1024×1024):

```bash
python libs/PixelFlow-diffusers/scripts/convert_pixelflow_to_diffusers.py \
  --checkpoint models/raw/PixelFlow/t2i/model.pt \
  --config models/raw/PixelFlow/t2i/config.yaml \
  --output models/BiliSakura/PixelFlow-diffusers/PixelFlow-T2I \
  --skip-text-encoder
```

## Sample images

```bash
python libs/PixelFlow-diffusers/scripts/sample_pixelflow.py \
  --model models/BiliSakura/PixelFlow-diffusers/PixelFlow-256 \
  --output demo.png \
  --class-label 207 \
  --steps 10 \
  --cfg 4.0
```

## Load from Python

Development (lib package):

```python
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

pipe = PixelFlowPipeline.from_pretrained(str(variant))
pipe.to("cuda")
```
