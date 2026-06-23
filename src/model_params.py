"""
Advanced model loading parameters for LM Studio and llama.cpp inference servers.

These mirror the settings from llama.cpp's web UI (Context Length, GPU Offload,
CPU Thread Pool, Batch Sizes, KV Cache, etc.) and are tuned for OLMo OCR 2 7B
on AMD RX 6900 XT with 16GB VRAM.
"""

# ── Context & Memory Settings ───────────────────────────────────────────────
MODEL_CONTEXT_LENGTH = 4096         # Max tokens in context window (8192 max for 7B)
MODEL_KEEP_IN_MEMORY = False        # Keep model in VRAM between inferences (uses more VRAM)

# ── GPU Offload Settings ────────────────────────────────────────────────────
MODEL_N_GPU_LAYERS = 99             # Number of GPU layers (99 = "offload all that fit")
MODEL_OFFLOAD_KV_CACHE_GPU = True   # Move KV cache to GPU memory (faster inference)
MODEL_FLASH_ATTENTION = False       # Use Flash Attention v2 if GPU supports (experimental)

# ── Batch Size Settings ─────────────────────────────────────────────────────
MODEL_BATCH_SIZE = 512              # Total tokens per batch during inference
MODEL_UBATCH_SIZE = 128             # Micro-batch size (reduces VRAM spikes and stalls)

# ── Thread & Concurrency Settings ───────────────────────────────────────────
MODEL_N_THREADS = 8                 # CPU threads for compute (tune per your CPU cores)
MODEL_MAX_CONCURRENCY = 1           # Max concurrent requests (1 = process one at a time)

# ── Quantization Settings ───────────────────────────────────────────────────
MODEL_KV_CACHE_QUANT = "q8_0"       # KV cache quantization (q8_0, q4_0, or disabled)


def get_model_params_dict() -> dict:
    """Return all model loading parameters as a dictionary for UI/config."""
    return {
        "context_length": MODEL_CONTEXT_LENGTH,
        "batch_size": MODEL_BATCH_SIZE,
        "ubatch_size": MODEL_UBATCH_SIZE,
        "gpu_layers": MODEL_N_GPU_LAYERS,
        "threads": MODEL_N_THREADS,
        "max_concurrency": MODEL_MAX_CONCURRENCY,
        "keep_in_memory": MODEL_KEEP_IN_MEMORY,
        "offload_kv_cache_gpu": MODEL_OFFLOAD_KV_CACHE_GPU,
        "flash_attention": MODEL_FLASH_ATTENTION,
        "kv_cache_quant": MODEL_KV_CACHE_QUANT,
    }


def describe_model_param(param_name: str) -> str:
    """Return a description of a model parameter for UI tooltips."""
    descriptions = {
        "context_length": "Maximum tokens in the context window (4096 safe, 8192 max for 7B with 16GB)",
        "batch_size": "Total tokens processed per batch (higher = faster but uses more VRAM)",
        "ubatch_size": "Micro-batch size per kernel call (lower = less VRAM spikes)",
        "gpu_layers": "Layers to offload to GPU (99 = offload all that fit in VRAM)",
        "threads": "CPU threads for compute (tune per your CPU core count)",
        "max_concurrency": "Max concurrent inference requests (1 recommended for stability)",
        "keep_in_memory": "Keep model loaded in VRAM between inferences (faster but uses more VRAM)",
        "offload_kv_cache_gpu": "Store KV cache on GPU for faster inference",
        "flash_attention": "Use Flash Attention v2 optimization (experimental, GPU-dependent)",
        "kv_cache_quant": "Quantization for KV cache (q8_0=good quality, q4_0=lower quality/VRAM)",
    }
    return descriptions.get(param_name, "")
