from pathlib import Path
from dataclasses import dataclass

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
<<<<<<< Updated upstream
PDF_PATH = DATA_DIR / "mml-book-247-267.pdf"
=======
PDF_PATH = DATA_DIR / "mml-book_clean.pdf"
>>>>>>> Stashed changes
PDF_IMAGES_DIR = DATA_DIR / "pdf_images"
MARKDOWN_DIR = DATA_DIR / "markdown"
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"
CORPUS_JSON = PROJECT_ROOT / "corpus.json"
SAMPLE_JSON = PROJECT_ROOT / "sample.json"

<<<<<<< Updated upstream
# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "allenai/olmOCR-7B-0225-preview"
PROCESSOR_NAME = "Qwen/Qwen2-VL-7B-Instruct"

# ── Quantization (4-bit NF4 via bitsandbytes) ───────────────────────────────
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
=======
# ── LM Studio API ───────────────────────────────────────────────────────────
# LM Studio runs on Windows; from WSL use the vEthernet (WSL) adapter IP.
# Make sure LM Studio server is set to listen on 0.0.0.0 (not 127.0.0.1).
API_BASE_URL = "http://100.76.121.29:1234/v1"
API_MODEL = "allenai/olmocr-2-7b"

# ── llama.cpp Server (Alternative to LM Studio) ────────────────────────────
# Use llama.cpp with Vulkan GPU acceleration for better performance on AMD GPUs.
# 
# Run llama.cpp server with:
#   ./llama-server \
#     -m /path/to/model.Q4_K_M.gguf \
#     --device Vulkan0 \
#     --main-gpu 0 \
#     -ngl 99 \
#     -c 4096 \
#     --batch-size 512 \
#     --ubatch-size 128 \
#     --parallel 1 \
#     -t 8 \
#     --port 1234 \
#     --host 0.0.0.0 \
#     -ctk q8_0 \
#     -ctv q8_0
#
# For AMD RX 6900 XT:
LLAMACPP_GPU_DEVICE = "Vulkan0"           # First Vulkan-capable GPU device
LLAMACPP_MAIN_GPU = 0                     # Primary GPU index for compute
LLAMACPP_GPU_LAYERS = 99                  # Offload as many layers as fit in VRAM
LLAMACPP_CONTEXT_LENGTH = 3072            # Smaller context reduces VRAM pressure and speeds up single-job runs
LLAMACPP_BATCH_SIZE = 256                 # Lower batch size is more stable on 16GB VRAM
LLAMACPP_UBATCH_SIZE = 64                 # Lower micro-batch size reduces VRAM spikes
LLAMACPP_PARALLEL_REQUESTS = 1            # Single request at a time for stability
LLAMACPP_CPU_THREADS = 8                  # CPU threads (tune per your CPU core count)
LLAMACPP_KV_CACHE_QUANT = "q8_0"          # KV cache quantization
LLAMACPP_PORT = 1234                      # Port for OpenAI-compatible API server
LLAMACPP_HOST = "0.0.0.0"                 # Bind to all interfaces for network access

# ── Workers ──────────────────────────────────────────────────────────────────
# Number of concurrent API requests (keep at 1 for single-GPU setups).
NUM_WORKERS = 1

# ── Retry settings ───────────────────────────────────────────────────────────
# Retries per page when the model crashes or returns an error.
MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds to wait before retrying (gives LM Studio time to reload)
>>>>>>> Stashed changes

# ── Image settings ───────────────────────────────────────────────────────────
TARGET_LONGEST_IMAGE_DIM = 1024

# ── Generation settings ─────────────────────────────────────────────────────
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0

# ── Sample size ──────────────────────────────────────────────────────────────
SAMPLE_PAGES = 10

# ── Output formatting ─────────────────────────────────────────────────────────
# If True, extract all embedded images/diagrams from each PDF page.
EXTRACT_EMBEDDED_IMAGES = True

# ── Server Type ──────────────────────────────────────────────────────────────
# Choose which server backend to use:
# "lm_studio" - LM Studio (cross-platform GUI, easier setup)
# "llamacpp"  - llama.cpp (better GPU utilization, lower latency)
SERVER_TYPE = "lm_studio"  # or "llamacpp"


@dataclass(frozen=True)
class OCRRuntimeConfig:
	"""Runtime settings for OCR runs with sensible defaults from module constants."""

	api_base_url: str = API_BASE_URL
	api_model: str = API_MODEL
	num_workers: int = NUM_WORKERS
	max_retries: int = MAX_RETRIES
	retry_delay: int = RETRY_DELAY
	target_longest_image_dim: int = TARGET_LONGEST_IMAGE_DIM
	max_new_tokens: int = MAX_NEW_TOKENS
	temperature: float = TEMPERATURE
	sample_pages: int = SAMPLE_PAGES


DEFAULT_RUNTIME_CONFIG = OCRRuntimeConfig()


def get_llamacpp_server_command(model_path: str) -> str:
	"""Generate a llama.cpp server launch command with RX 6900 XT Vulkan settings.
	
	Args:
	    model_path: Full path to the GGUF model file.
	
	Returns:
	    Shell command string ready to execute (e.g., in bash).
	"""
	return (
		f"./llama-server "
		f"-m {model_path} "
		f"--device {LLAMACPP_GPU_DEVICE} "
		f"--main-gpu {LLAMACPP_MAIN_GPU} "
		f"-ngl {LLAMACPP_GPU_LAYERS} "
		f"-c {LLAMACPP_CONTEXT_LENGTH} "
		f"--batch-size {LLAMACPP_BATCH_SIZE} "
		f"--ubatch-size {LLAMACPP_UBATCH_SIZE} "
		f"--parallel {LLAMACPP_PARALLEL_REQUESTS} "
		f"-t {LLAMACPP_CPU_THREADS} "
		f"--port {LLAMACPP_PORT} "
		f"--host {LLAMACPP_HOST} "
		f"-ctk {LLAMACPP_KV_CACHE_QUANT} "
		f"-ctv {LLAMACPP_KV_CACHE_QUANT}"
	)