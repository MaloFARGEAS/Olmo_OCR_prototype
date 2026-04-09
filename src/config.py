from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_PATH = DATA_DIR / "mml-book_10_20.pdf"
PDF_IMAGES_DIR = DATA_DIR / "pdf_images"
MARKDOWN_DIR = DATA_DIR / "markdown"
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"
CORPUS_JSON = PROJECT_ROOT / "corpus.json"
SAMPLE_JSON = PROJECT_ROOT / "sample.json"

# ── LM Studio API ───────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:1234/v1"
API_MODELS = [
    "allenai/olmocr-2-7b",
    "allenai/olmocr-2-7b:2",
]

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