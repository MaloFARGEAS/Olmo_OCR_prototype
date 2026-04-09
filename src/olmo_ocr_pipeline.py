"""
OLMo OCR 2 Pipeline
===================
Converts a PDF into page images + markdown using OLMo OCR 2 (7B)
quantized to 4-bit (NF4) to fit on an RTX 4060 8 GB.

Outputs:
    - data/pdf_images/page_XXX.png   (rendered page images)
    - data/markdown/page_XXX.md      (per-page markdown)
    - corpus.json                    (all pages)
    - sample.json                    (first N pages)
"""

import base64
import json
import shutil
import sys
from io import BytesIO
from pathlib import Path
import re

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

# ── olmocr utilities ─────────────────────────────────────────────────────────
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

# ── project config ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_QUANT_TYPE,
    CORPUS_JSON,
    LOAD_IN_4BIT,
    MARKDOWN_DIR,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    PDF_IMAGES_DIR,
    PDF_PATH,
    PROCESSOR_NAME,
    SAMPLE_JSON,
    SAMPLE_PAGES,
    TARGET_LONGEST_IMAGE_DIM,
    TEMPERATURE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 0 — Pre-flight checks
# ═══════════════════════════════════════════════════════════════════════════════

def preflight_checks() -> None:
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA is not available. An NVIDIA GPU with CUDA support is required.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    print(f"GPU : {gpu_name}  ({vram_gb} GB VRAM)")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    if not PDF_PATH.exists():
        sys.exit(f"ERROR: PDF not found at {PDF_PATH}")

    PDF_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage A — PDF → page images
# ═══════════════════════════════════════════════════════════════════════════════

def get_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF using pypdfium2 (bundled with olmocr)."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def render_pages(pdf_path: Path, num_pages: int) -> list[Path]:
    """Render each PDF page to a PNG in pdf_images/."""
    image_paths: list[Path] = []
    print(f"\n{'='*60}")
    print(f"Stage A — Rendering {num_pages} PDF pages to images")
    print(f"{'='*60}")

    poppler_ready = bool(shutil.which("pdfinfo") and shutil.which("pdftoppm"))
    pdf_doc = None

    if not poppler_ready:
        import pypdfium2 as pdfium

        print("  Poppler tools not found (pdfinfo/pdftoppm). Falling back to pypdfium2 renderer.")
        pdf_doc = pdfium.PdfDocument(str(pdf_path))

    for page_num in tqdm(range(1, num_pages + 1), desc="Rendering pages"):
        if poppler_ready:
            try:
                img_b64 = render_pdf_to_base64png(
                    str(pdf_path), page_num, target_longest_image_dim=TARGET_LONGEST_IMAGE_DIM
                )
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(BytesIO(img_bytes))
            except FileNotFoundError:
                import pypdfium2 as pdfium

                print("  Poppler tools became unavailable during rendering; switching to pypdfium2 fallback.")
                poppler_ready = False
                if pdf_doc is None:
                    pdf_doc = pdfium.PdfDocument(str(pdf_path))
                page = pdf_doc[page_num - 1]
                width, height = page.get_size()
                scale = TARGET_LONGEST_IMAGE_DIM / max(width, height)
                bitmap = page.render(scale=scale)
                img = bitmap.to_pil()
        else:
            page = pdf_doc[page_num - 1]
            width, height = page.get_size()
            scale = TARGET_LONGEST_IMAGE_DIM / max(width, height)
            bitmap = page.render(scale=scale)
            img = bitmap.to_pil()

        out_path = PDF_IMAGES_DIR / f"page_{page_num:03d}.png"
        img.save(out_path)
        image_paths.append(out_path)

    if pdf_doc is not None:
        pdf_doc.close()

    print(f"  Saved {len(image_paths)} images to {PDF_IMAGES_DIR}")
    return image_paths


# ═══════════════════════════════════════════════════════════════════════════════
# Stage B — Load model (4-bit quantized)
# ═══════════════════════════════════════════════════════════════════════════════

def load_model():
    """Load OLMo OCR model in 4-bit NF4 quantization."""
    print(f"\n{'='*60}")
    print("Stage B — Loading model (4-bit NF4 quantization)")
    print(f"{'='*60}")

    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print(f"  Model: {MODEL_NAME}")
    print(f"  Quantization: 4-bit {BNB_4BIT_QUANT_TYPE}, compute dtype: {BNB_4BIT_COMPUTE_DTYPE}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)

    vram_used = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  Model loaded — VRAM used: {vram_used:.1f} GB")
    return model, processor


# ═══════════════════════════════════════════════════════════════════════════════
# Stage C — OCR inference (per page)
# ═══════════════════════════════════════════════════════════════════════════════

def ocr_page(
    model,
    processor,
    pdf_path: Path,
    page_num: int,
    image_path: Path,
) -> str:
    """Run OCR on a single page and return the markdown text."""
    # Read the already-rendered image
    img_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    # Build prompt with anchor text from the PDF text layer
    try:
        anchor_text = get_anchor_text(str(pdf_path), page_num, pdf_engine="pdfreport", target_length=4000)
    except Exception:
        anchor_text = ""
    prompt = build_finetuning_prompt(anchor_text) + (
        "\n\nOutput requirements:\n"
        "- Return valid Markdown only.\n"
        "- Preserve math as LaTeX with $...$ for inline and $$...$$ for block equations.\n"
        "- Do not escape LaTeX backslashes unless required by Markdown syntax.\n"
        "- Do not wrap the entire answer in markdown code fences.\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    result = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    del inputs, output, new_tokens
    torch.cuda.empty_cache()

    return result


def normalize_markdown_latex(text: str) -> str:
    """Normalize model output to plain markdown while preserving LaTeX math delimiters."""
    cleaned = text.strip()

    # Some model responses are JSON wrappers with a natural_text payload.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            natural_text = parsed.get("natural_text")
            if isinstance(natural_text, str) and natural_text.strip():
                cleaned = natural_text.strip()
    except json.JSONDecodeError:
        # Fallback for JSON-like payloads with invalid escapes (for example \$).
        match = re.search(r'"natural_text"\s*:\s*"(.*)"\s*}\s*$', cleaned, flags=re.DOTALL)
        if match:
            candidate = match.group(1)
            candidate = candidate.replace(r"\$", "$")
            candidate = candidate.replace(r"\n", "\n")
            candidate = candidate.replace(r'\"', '"')
            if candidate.strip():
                cleaned = candidate.strip()

    # Remove accidental top-level markdown fences added by the model.
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            cleaned = "\n".join(lines[1:-1]).strip()

    # Normalize escaped dollar signs used as math delimiters.
    cleaned = cleaned.replace(r"\$", "$")

    # Turn \(...\) and \[...\] into markdown-compatible math delimiters.
    cleaned = re.sub(r"\\\((.+?)\\\)", r"$\1$", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\\\[(.+?)\\\]", r"$$\n\1\n$$", cleaned, flags=re.DOTALL)

    return cleaned.strip() + "\n"


def run_ocr(model, processor, pdf_path: Path, image_paths: list[Path]) -> list[dict]:
    """Run OCR on all pages and save markdown files."""
    print(f"\n{'='*60}")
    print(f"Stage C — Running OCR on {len(image_paths)} pages")
    print(f"{'='*60}")

    pages: list[dict] = []
    source_name = pdf_path.name
    for idx, img_path in enumerate(tqdm(image_paths, desc="OCR pages"), start=1):
        md_text = ocr_page(model, processor, pdf_path, idx, img_path)
        md_text = normalize_markdown_latex(md_text)

        # Save per-page markdown
        md_path = MARKDOWN_DIR / f"page_{idx:03d}.md"
        md_path.write_text(md_text, encoding="utf-8")

        pages.append(
            {
                "source": source_name,
                "page": idx,
                "char_count": len(md_text),
                "text": md_text,
            }
        )
        tqdm.write(f"  page {idx:03d}: {len(md_text)} chars")

    print(f"  Saved {len(pages)} markdown files to {MARKDOWN_DIR}")
    return pages


# ═══════════════════════════════════════════════════════════════════════════════
# Stage D — Save JSON outputs
# ═══════════════════════════════════════════════════════════════════════════════

def save_json_outputs(pages: list[dict]) -> None:
    print(f"\n{'='*60}")
    print("Stage D — Saving JSON outputs")
    print(f"{'='*60}")

    CORPUS_JSON.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  corpus.json: {len(pages)} pages -> {CORPUS_JSON}")

    sample = pages[:SAMPLE_PAGES]
    SAMPLE_JSON.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  sample.json: {len(sample)} pages -> {SAMPLE_JSON}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    preflight_checks()

    num_pages = get_page_count(PDF_PATH)
    print(f"  PDF: {PDF_PATH.name} — {num_pages} pages")

    # Stage A — render pages
    image_paths = render_pages(PDF_PATH, num_pages)

    # Stage B — load model
    model, processor = load_model()

    # Stage C — OCR
    pages = run_ocr(model, processor, PDF_PATH, image_paths)

    # Stage D — save outputs
    save_json_outputs(pages)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
