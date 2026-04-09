from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_PATH = DATA_DIR / "mml-book-247-267.pdf"
PDF_IMAGES_DIR = DATA_DIR / "pdf_images"
MARKDOWN_DIR = DATA_DIR / "markdown"
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"
CORPUS_JSON = PROJECT_ROOT / "corpus.json"
SAMPLE_JSON = PROJECT_ROOT / "sample.json"

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "allenai/olmOCR-7B-0225-preview"
PROCESSOR_NAME = "Qwen/Qwen2-VL-7B-Instruct"

# ── Quantization (4-bit NF4 via bitsandbytes) ───────────────────────────────
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"

# ── Image settings ───────────────────────────────────────────────────────────
TARGET_LONGEST_IMAGE_DIM = 1024

# ── Generation settings ─────────────────────────────────────────────────────
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.8

# ── Sample size ──────────────────────────────────────────────────────────────
SAMPLE_PAGES = 10

# ── Output formatting ─────────────────────────────────────────────────────────
# If True, extract all embedded images/diagrams from each PDF page.
EXTRACT_EMBEDDED_IMAGES = True