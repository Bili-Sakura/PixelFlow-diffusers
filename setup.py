from pathlib import Path

from setuptools import find_packages, setup

LIB_ROOT = Path(__file__).resolve().parent

setup(
    name="pixelflow-diffusers",
    version="0.1.0",
    description="Diffusers-style PixelFlow implementation",
    long_description=(LIB_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "diffusers>=0.32.0",
        "einops",
        "numpy",
        "PyYAML",
        "safetensors",
        "torch",
        "transformers>=4.48.0",
    ],
    scripts=[
        "scripts/convert_pixelflow_to_diffusers.py",
        "scripts/sample_pixelflow.py",
    ],
)
