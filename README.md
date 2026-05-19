# Olmo_OCR_prototype
Python implementation for RAG preprocessing and PDF digitization to markdown using OLMo OCR with OpenAI-compatible inference servers (LM Studio or llama.cpp).

## Features
- OCR pipeline for PDF to per-page markdown.
- Streamlit web app for interactive PDF processing.
- Support for multiple inference backends:
	- LM Studio (cross-platform GUI)
	- llama.cpp (optimized GPU performance)
- Input modes:
	- Upload PDF
	- Local PDF path (server-side)
- Output downloads:
	- Combined markdown file
	- Per-page markdown zip archive

## Requirements
- Python 3.11+
- OpenAI-compatible inference server:
	- **LM Studio** (recommended for beginners), OR
	- **llama.cpp** (recommended for AMD GPUs like RX 6900 XT)
- OLMo OCR model in GGUF format (for llama.cpp) or natively supported (for LM Studio)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Server Setup

### Option A: LM Studio (Cross-platform GUI)

1. Download and install [LM Studio](https://lmstudio.ai/).
2. Load the OLMo OCR 2 7B model.
3. Start the local server (default: `http://127.0.0.1:1234`).
4. If accessing from WSL/container, edit `API_BASE_URL` in `src/config.py` to use the WSL vEthernet IP (e.g., `http://100.76.121.29:1234/v1`).

### Option B: llama.cpp (AMD GPU Optimization)

Ideal for AMD Radeon RX 6900 XT with Vulkan support.

1. **Build or download llama.cpp** with Vulkan support:
	```bash
	git clone https://github.com/ggerganov/llama.cpp.git
	cd llama.cpp
	make -j$(nproc)  # or `make LLAMA_VULKAN=1` if CMake is your build system
	```

2. **Download an OLMo OCR 2 model in GGUF format** (if not already available).

3. **Generate and run the server command**:
	```python
	from src.config import get_llamacpp_server_command
	model_path = "/path/to/olmocr-2-7b.Q4_K_M.gguf"
	cmd = get_llamacpp_server_command(model_path)
	print(cmd)
	# Output: ./llama-server -m /path/to/olmocr-2-7b.Q4_K_M.gguf --device Vulkan0 --main-gpu 0 -ngl 99 ...
	```
	
	Or run directly in bash:
	```bash
	./llama-server \
	  -m /path/to/olmocr-2-7b.Q4_K_M.gguf \
	  --device Vulkan0 \
	  --main-gpu 0 \
	  -ngl 99 \
	  -c 4096 \
	  --batch-size 512 \
	  --ubatch-size 128 \
	  --parallel 1 \
	  -t 8 \
	  --port 1234 \
	  --host 0.0.0.0 \
	  -ctk q8_0 \
	  -ctv q8_0
	```

4. **Update `API_BASE_URL`** in `src/config.py` if needed (default: `http://localhost:1234/v1`).

#### llama.cpp Parameters Explained (RX 6900 XT)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--device Vulkan0` | GPU device | Uses first Vulkan-capable GPU (your RX 6900 XT) |
| `--main-gpu 0` | Primary GPU | Ensures GPU is the main compute device |
| `-ngl 99` | GPU layers | Special value: "offload as many as fit in VRAM" (~100% on 7B models with 16GB VRAM) |
| `-c 4096` | Context length | 4096 tokens; increase if VRAM allows |
| `--batch-size 512` | Batch tokens | Total tokens per inference; tune carefully |
| `--ubatch-size 128` | Micro-batch | Reduces VRAM spikes; avoids stalls |
| `--parallel 1` | Concurrency | Handle one request at a time; queue externally |
| `-t 8` | CPU threads | Adjust per your CPU core count |
| `-ctk q8_0` / `-ctv q8_0` | KV cache quantization | Q8_0 is a good speed/quality tradeoff |

## Run Streamlit App

```bash
uv run streamlit run app.py
# or: source .venv/bin/activate && python -m streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## CLI Pipeline (Existing Behavior)

```bash
python src/olmo_ocr_pipeline.py
```

This generates:
- `data/pdf_images/page_XXX.png`
- `data/markdown/page_XXX.md`
- `corpus.json`
- `sample.json`

## Troubleshooting

### "Cannot reach LM Studio API" or "ModuleNotFoundError: No module named 'olmocr'"
- Check that your venv is activated: `source .venv/bin/activate`
- Run the app with: `python -m streamlit run app.py`
- Verify dependencies: `uv sync` or `pip install -r requirements.txt`

### llama.cpp Performance Issues
- Increase `-ngl` gradually (higher = more GPU layers) if VRAM allows.
- Reduce `--batch-size` if OOM errors occur.
- Ensure `--device Vulkan0` matches your GPU (check `vulkaninfo` if unsure).

### Long PDF Processing
- Streamlit may timeout on very large PDFs. Consider processing smaller page ranges.
- Reduce `NUM_WORKERS` in `src/config.py` if model crashes.
- Increase `RETRY_DELAY` if you see repeated failures.

## Notes
- The app supports real-time progress updates and recovers gracefully from transient errors.
- Output quality depends on model quantization (Q4_K_M is good for 7B models on 16GB VRAM).
- Consider using `--parallel 1` with external request queueing for stability (avoids the "0% inference" stall issue).
