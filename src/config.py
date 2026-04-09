"""
config.py – Centralised configuration for the OlmoOCR 2 RAG-preprocessing pipeline.

All tunable parameters and filesystem paths are defined here so that
olmo_ocr_pipeline.py stays clean and easy to adapt.  Every value can be
overridden at runtime via the corresponding environment variable.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

# Absolute path to the project root (parent of src/)
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
OUTPUT_DIR: Path = BASE_DIR / "image_and_markdown"
IMAGES_DIR: Path = OUTPUT_DIR / "images"

# ---------------------------------------------------------------------------
# Input / output files
# ---------------------------------------------------------------------------

# Path to the source PDF document
PDF_PATH: Path = DATA_DIR / "document.pdf"

# Destination Markdown file produced by the OCR pipeline
OUTPUT_MD_PATH: Path = OUTPUT_DIR / "output.md"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# HuggingFace model identifier for olmOCR 2
# allenai/olmOCR-7B-0225-preview is a Qwen2-VL checkpoint fine-tuned for OCR
MODEL_NAME: str = os.environ.get("OLMOCR_MODEL", "allenai/olmOCR-7B-0225-preview")

# Compute device: "cuda" to use GPU, "cpu" to force CPU-only inference,
# or "auto" to let the library decide (recommended when multiple GPUs are
# available).
DEVICE: str = os.environ.get("OLMOCR_DEVICE", "auto")

# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

# DPI used when rasterising PDF pages (higher = better quality, more VRAM)
PDF_RENDER_DPI: int = int(os.environ.get("OLMOCR_DPI", "150"))

# Maximum image width sent to the model (pixels). Pages wider than this are
# downscaled while preserving the aspect ratio.
MAX_IMAGE_WIDTH: int = int(os.environ.get("OLMOCR_MAX_WIDTH", "1024"))

# Maximum image height sent to the model (pixels).
MAX_IMAGE_HEIGHT: int = int(os.environ.get("OLMOCR_MAX_HEIGHT", "1024"))

# ---------------------------------------------------------------------------
# Generation / inference
# ---------------------------------------------------------------------------

# Maximum number of new tokens the model is allowed to generate per page.
MAX_NEW_TOKENS: int = int(os.environ.get("OLMOCR_MAX_TOKENS", "4096"))

# Temperature for sampling (0 = greedy / deterministic).
TEMPERATURE: float = float(os.environ.get("OLMOCR_TEMPERATURE", "0.0"))

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

# Minimum pixel area for an embedded image to be extracted from the PDF.
# Images smaller than this are silently ignored (e.g. decorative bullets).
MIN_IMAGE_AREA: int = int(os.environ.get("OLMOCR_MIN_IMAGE_AREA", "5000"))
