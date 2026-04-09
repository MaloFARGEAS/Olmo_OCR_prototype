# OlmoOCR 2 – RAG Preprocessing Pipeline

A Python pipeline that converts a PDF document into structured Markdown using
**olmOCR 2** (Allen AI's `allenai/olmOCR-7B-0225-preview` vision-language model,
built on Qwen2-VL).  The output is optimised for downstream RAG (Retrieval-
Augmented Generation) ingestion.

---

## Project structure

```
Olmo_OCR_prototype/
├── data/
│   └── document.pdf               # Source PDF to analyse (add your own)
├── image_and_markdown/
│   ├── images/                    # Figures and graphics extracted from the PDF
│   └── output.md                  # Markdown file produced by the OCR pipeline
├── src/
│   ├── config.py                  # All parameters and filesystem paths
│   └── olmo_ocr_pipeline.py       # Main pipeline script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Quick start

### 1. Clone and install dependencies

```bash
git clone https://github.com/MaloFARGEAS/Olmo_OCR_prototype.git
cd Olmo_OCR_prototype
pip install -r requirements.txt
```

> **GPU strongly recommended.**  The Qwen2-VL 7B model requires ~16 GB of VRAM
> in bfloat16.  CPU inference is supported but very slow.

### 2. Add your PDF

Copy the PDF you want to process to `data/document.pdf`:

```bash
cp /path/to/your/file.pdf data/document.pdf
```

### 3. Run the pipeline

```bash
python src/olmo_ocr_pipeline.py
```

The script will:

1. Render each PDF page as a high-resolution image.
2. Run olmOCR 2 on every page and collect the Markdown output.
3. Extract any embedded figures/graphics and save them under
   `image_and_markdown/images/`.
4. Write the final Markdown (with image references) to
   `image_and_markdown/output.md`.

---

## Configuration

All settings live in `src/config.py`.  Every parameter can also be overridden
with an environment variable:

| Parameter | Env variable | Default | Description |
|---|---|---|---|
| `MODEL_NAME` | `OLMOCR_MODEL` | `allenai/olmOCR-7B-0225-preview` | HuggingFace model ID |
| `DEVICE` | `OLMOCR_DEVICE` | `auto` | `"cuda"`, `"cpu"`, or `"auto"` |
| `PDF_RENDER_DPI` | `OLMOCR_DPI` | `150` | DPI for page rasterisation |
| `MAX_IMAGE_WIDTH` | `OLMOCR_MAX_WIDTH` | `1024` | Max page width sent to the model |
| `MAX_IMAGE_HEIGHT` | `OLMOCR_MAX_HEIGHT` | `1024` | Max page height sent to the model |
| `MAX_NEW_TOKENS` | `OLMOCR_MAX_TOKENS` | `4096` | Max tokens generated per page |
| `TEMPERATURE` | `OLMOCR_TEMPERATURE` | `0.0` | Sampling temperature (0 = greedy) |
| `MIN_IMAGE_AREA` | `OLMOCR_MIN_IMAGE_AREA` | `5000` | Minimum area (px²) to extract an image |

Example – use CPU and increase DPI:

```bash
OLMOCR_DEVICE=cpu OLMOCR_DPI=200 python src/olmo_ocr_pipeline.py
```

---

## Using the pipeline programmatically

```python
from pathlib import Path
from src.olmo_ocr_pipeline import run_pipeline

run_pipeline(
    pdf_path=Path("data/my_report.pdf"),
    output_md_path=Path("image_and_markdown/my_report.md"),
    images_dir=Path("image_and_markdown/images"),
)
```

---

## Requirements

- Python 3.10+
- PyTorch 2.2+ (with CUDA for GPU inference)
- transformers 4.46+
- PyMuPDF (for PDF rendering and image extraction)
- Pillow

See `requirements.txt` for the complete list.

---

## Model

olmOCR 2 is released by Allen Institute for AI under the Apache 2.0 licence.
The model card is available at:
<https://huggingface.co/allenai/olmOCR-7B-0225-preview>