"""
<<<<<<< Updated upstream
OLMo OCR 2 Pipeline
===================
=======
OLMo OCR 2 Pipeline (LM Studio API)
===================================
>>>>>>> Stashed changes
Converts a PDF into page images + markdown using OLMo OCR 2 (7B)
quantized to 4-bit (NF4) to fit on an RTX 4060 8 GB.

Outputs:
    - data/pdf_images/page_XXX.png   (rendered page images)
    - data/markdown/page_XXX.md      (per-page markdown)
    - corpus.json                    (all pages)
    - sample.json                    (first N pages)
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import sys
<<<<<<< Updated upstream
=======
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
>>>>>>> Stashed changes
from io import BytesIO
from pathlib import Path
import re
from typing import Callable

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

# -- olmocr utilities ---------------------------------------------------------
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

# -- project config -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
<<<<<<< Updated upstream
from src.config import (
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_QUANT_TYPE,
    CORPUS_JSON,
    LOAD_IN_4BIT,
    MARKDOWN_DIR,
    MAX_NEW_TOKENS,
    MODEL_NAME,
=======
from src.config import (  # noqa: E402
    CORPUS_JSON,
    DEFAULT_RUNTIME_CONFIG,
    MARKDOWN_DIR,
    OCRRuntimeConfig,
>>>>>>> Stashed changes
    PDF_IMAGES_DIR,
    PDF_PATH,
    PROCESSOR_NAME,
    SAMPLE_JSON,
)

RenderProgressCallback = Callable[[int, int], None]
PageDoneCallback = Callable[[int, str], None]
PageErrorCallback = Callable[[int, str], None]
ShouldStopCallback = Callable[[], bool]


class OCRCancelledError(RuntimeError):
    """Raised when the caller requests cancellation of an OCR run."""


# =============================================================================
# Stage 0 -- Pre-flight checks
# =============================================================================

<<<<<<< Updated upstream
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
=======

def preflight_checks(
    config: OCRRuntimeConfig,
    pdf_path: Path,
    pdf_images_dir: Path,
    markdown_dir: Path,
) -> None:
    if not pdf_path.exists():
        sys.exit(f"ERROR: PDF not found at {pdf_path}")

    # Verify LM Studio API is reachable
    client = OpenAI(base_url=config.api_base_url, api_key="lm-studio")
    try:
        client.models.list()
        print(f"LM Studio API: {config.api_base_url} -- connected")
        print(f"Model: {config.api_model}")
        print(f"Workers: {config.num_workers}")
    except Exception as exc:
        sys.exit(f"ERROR: Cannot reach LM Studio API at {config.api_base_url}\n  {exc}")

    pdf_images_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)

>>>>>>> Stashed changes

# =============================================================================
# Stage A -- PDF -> page images
# =============================================================================


def get_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF using pypdfium2 (bundled with olmocr)."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def render_pages(
    pdf_path: Path,
    num_pages: int,
    pdf_images_dir: Path,
    target_longest_image_dim: int,
    on_render_progress: RenderProgressCallback | None = None,
    should_stop: ShouldStopCallback | None = None,
) -> list[Path]:
    """Render each PDF page to a PNG in the selected image directory."""
    image_paths: list[Path] = []
    print(f"\n{'='*60}")
    print(f"Stage A -- Rendering {num_pages} PDF pages to images")
    print(f"{'='*60}")

    poppler_ready = bool(shutil.which("pdfinfo") and shutil.which("pdftoppm"))
    pdf_doc = None

    if not poppler_ready:
        import pypdfium2 as pdfium

        print("  Poppler tools not found (pdfinfo/pdftoppm). Falling back to pypdfium2 renderer.")
        pdf_doc = pdfium.PdfDocument(str(pdf_path))

    for page_num in tqdm(range(1, num_pages + 1), desc="Rendering pages"):
        if should_stop is not None and should_stop():
            raise OCRCancelledError("OCR run cancelled during PDF rendering")

        if poppler_ready:
            try:
                img_b64 = render_pdf_to_base64png(
                    str(pdf_path), page_num, target_longest_image_dim=target_longest_image_dim
                )
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(BytesIO(img_bytes))
            except FileNotFoundError:
                import pypdfium2 as pdfium

                print("  Poppler tools became unavailable during rendering; switching to pypdfium2 fallback.")
                poppler_ready = False
                if pdf_doc is None:
                    pdf_doc = pdfium.PdfDocument(str(pdf_path))
                assert pdf_doc is not None
                page = pdf_doc[page_num - 1]
                width, height = page.get_size()
                scale = target_longest_image_dim / max(width, height)
                bitmap = page.render(scale=scale)
                img = bitmap.to_pil()
        else:
            assert pdf_doc is not None
            page = pdf_doc[page_num - 1]
            width, height = page.get_size()
            scale = target_longest_image_dim / max(width, height)
            bitmap = page.render(scale=scale)
            img = bitmap.to_pil()

        out_path = pdf_images_dir / f"page_{page_num:03d}.png"
        img.save(out_path)
        image_paths.append(out_path)
        if on_render_progress is not None:
            on_render_progress(page_num, num_pages)

    if pdf_doc is not None:
        pdf_doc.close()

    print(f"  Saved {len(image_paths)} images to {pdf_images_dir}")
    return image_paths


<<<<<<< Updated upstream
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
=======
# =============================================================================
# Stage B -- Create API client
# =============================================================================


def create_client(config: OCRRuntimeConfig = DEFAULT_RUNTIME_CONFIG, verbose: bool = True) -> OpenAI:
    """Create an OpenAI-compatible client pointing at LM Studio."""
    if verbose:
        print(f"\n{'='*60}")
        print("Stage B -- Connecting to LM Studio API")
        print(f"{'='*60}")
        print(f"  Endpoint: {config.api_base_url}")
        print(f"  Model:    {config.api_model}")
        print(f"  Workers:  {config.num_workers}")

    client = OpenAI(base_url=config.api_base_url, api_key="lm-studio")
    return client

>>>>>>> Stashed changes

# =============================================================================
# Stage C -- OCR inference (per page)
# =============================================================================


def ocr_page(
<<<<<<< Updated upstream
    model,
    processor,
=======
    client: OpenAI,
    config: OCRRuntimeConfig,
>>>>>>> Stashed changes
    pdf_path: Path,
    page_num: int,
    image_path: Path,
) -> str:
<<<<<<< Updated upstream
    """Run OCR on a single page and return the markdown text."""
    # Read the already-rendered image
=======
    """Run OCR on a single page via the LM Studio API and return markdown text."""
>>>>>>> Stashed changes
    img_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    # Build prompt with anchor text from the PDF text layer.
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

<<<<<<< Updated upstream
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
=======
    response = client.chat.completions.create(
        model=config.api_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            }
        ],
        max_tokens=config.max_new_tokens,
        temperature=config.temperature,
>>>>>>> Stashed changes
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
        )

<<<<<<< Updated upstream
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    result = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    del inputs, output, new_tokens
    torch.cuda.empty_cache()

    return result
=======
    if not response.choices:
        print(f"  [WARN] Empty response for page {page_num}: {response}")
        return ""

    return response.choices[0].message.content or ""


def ocr_page_with_retry(
    client: OpenAI,
    config: OCRRuntimeConfig,
    pdf_path: Path,
    page_num: int,
    image_path: Path,
    should_stop: ShouldStopCallback | None = None,
) -> str:
    """Call ocr_page with retries and backoff on failure."""
    for attempt in range(1, config.max_retries + 1):
        if should_stop is not None and should_stop():
            raise OCRCancelledError(f"OCR run cancelled before page {page_num}")

        try:
            return ocr_page(client, config, pdf_path, page_num, image_path)
        except Exception as exc:
            if should_stop is not None and should_stop():
                raise OCRCancelledError(f"OCR run cancelled while processing page {page_num}") from exc

            if attempt == config.max_retries:
                print(f"  [ERROR] Page {page_num} failed after {config.max_retries} attempts: {exc}")
                return ""
            wait = config.retry_delay * attempt
            print(f"  [RETRY] Page {page_num} attempt {attempt}/{config.max_retries} failed: {exc}")
            print(f"           Waiting {wait}s for LM Studio to recover...")
            time.sleep(wait)

    return ""
>>>>>>> Stashed changes


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


<<<<<<< Updated upstream
def run_ocr(model, processor, pdf_path: Path, image_paths: list[Path]) -> list[dict]:
    """Run OCR on all pages and save markdown files."""
    print(f"\n{'='*60}")
    print(f"Stage C — Running OCR on {len(image_paths)} pages")
    print(f"{'='*60}")

    pages: list[dict] = []
=======
def run_ocr(
    client: OpenAI,
    config: OCRRuntimeConfig,
    pdf_path: Path,
    image_paths: list[Path],
    markdown_dir: Path,
    write_markdown_files: bool = True,
    resume: bool = True,
    on_page_done: PageDoneCallback | None = None,
    on_page_error: PageErrorCallback | None = None,
    should_stop: ShouldStopCallback | None = None,
) -> tuple[list[dict], list[int]]:
    """Run OCR on all pages and optionally write markdown page files."""
    print(f"\n{'='*60}")
    print(f"Stage C -- Running OCR on {len(image_paths)} pages ({config.num_workers} workers)")
    print(f"{'='*60}")

    results: dict[int, str] = {}
    failed_pages: list[int] = []
>>>>>>> Stashed changes
    source_name = pdf_path.name
    for idx, img_path in enumerate(tqdm(image_paths, desc="OCR pages"), start=1):
        md_text = ocr_page(model, processor, pdf_path, idx, img_path)
        md_text = normalize_markdown_latex(md_text)

        # Save per-page markdown
        md_path = MARKDOWN_DIR / f"page_{idx:03d}.md"
        md_path.write_text(md_text, encoding="utf-8")

<<<<<<< Updated upstream
=======
    if write_markdown_files:
        markdown_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: load already-processed pages from markdown files.
    skipped = 0
    if write_markdown_files and resume:
        for i, _img_path in enumerate(image_paths):
            idx = i + 1
            md_path = markdown_dir / f"page_{idx:03d}.md"
            if md_path.exists() and md_path.stat().st_size > 0:
                results[idx] = md_path.read_text(encoding="utf-8")
                skipped += 1
                if on_page_done is not None:
                    on_page_done(idx, results[idx])

    if skipped:
        print(f"  Resuming: {skipped} pages already done, {len(image_paths) - skipped} remaining")

    todo = [(i + 1, img_path) for i, img_path in enumerate(image_paths) if (i + 1) not in results]

    def _process_page(idx: int, img_path: Path) -> tuple[int, str]:
        if should_stop is not None and should_stop():
            raise OCRCancelledError(f"OCR run cancelled before page {idx}")

        md_text = ocr_page_with_retry(client, config, pdf_path, idx, img_path, should_stop=should_stop)
        return idx, md_text

    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = {executor.submit(_process_page, idx, img_path): idx for idx, img_path in todo}

        with tqdm(total=len(todo), desc="OCR pages", initial=0) as pbar:
            for future in as_completed(futures):
                if should_stop is not None and should_stop():
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise OCRCancelledError("OCR run cancelled during page processing")

                idx = futures[future]
                try:
                    md_text = future.result()[1]
                except OCRCancelledError:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as exc:
                    md_text = ""
                    error_message = f"Page {idx} failed: {exc}"
                    print(f"  [ERROR] {error_message}")
                    if on_page_error is not None:
                        on_page_error(idx, str(exc))

                md_text = normalize_markdown_latex(md_text)
                if not md_text.strip():
                    failed_pages.append(idx)
                    if on_page_error is not None:
                        on_page_error(idx, "Empty response")

                if write_markdown_files:
                    md_path = markdown_dir / f"page_{idx:03d}.md"
                    md_path.write_text(md_text, encoding="utf-8")

                results[idx] = md_text
                if on_page_done is not None:
                    on_page_done(idx, md_text)

                pbar.update(1)
                tqdm.write(f"  page {idx:03d}: {len(md_text)} chars")

    pages: list[dict] = []
    for idx in sorted(results.keys()):
        md_text = results[idx]
>>>>>>> Stashed changes
        pages.append(
            {
                "source": source_name,
                "page": idx,
                "char_count": len(md_text),
                "text": md_text,
            }
        )
        tqdm.write(f"  page {idx:03d}: {len(md_text)} chars")

    if write_markdown_files:
        print(f"  Saved {len(pages)} markdown files to {markdown_dir}")

    return pages, sorted(set(failed_pages))


# =============================================================================
# Stage D -- Save JSON outputs
# =============================================================================


def save_json_outputs(
    pages: list[dict],
    corpus_json_path: Path = CORPUS_JSON,
    sample_json_path: Path = SAMPLE_JSON,
    sample_pages: int = DEFAULT_RUNTIME_CONFIG.sample_pages,
) -> None:
    print(f"\n{'='*60}")
    print("Stage D -- Saving JSON outputs")
    print(f"{'='*60}")

    corpus_json_path.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  corpus.json: {len(pages)} pages -> {corpus_json_path}")

    sample = pages[:sample_pages]
    sample_json_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  sample.json: {len(sample)} pages -> {sample_json_path}")


# =============================================================================
# Convenience helpers for app integrations
# =============================================================================


def combine_markdown_pages(pages: list[dict]) -> str:
    """Return a single markdown document from per-page OCR outputs."""
    chunks: list[str] = []
    for page in sorted(pages, key=lambda item: item["page"]):
        chunks.append(f"<!-- Page {page['page']} -->\n")
        chunks.append(page["text"].rstrip() + "\n")
        chunks.append("\n")
    return "".join(chunks)


<<<<<<< Updated upstream
    # Stage B — load model
    model, processor = load_model()

    # Stage C — OCR
    pages = run_ocr(model, processor, PDF_PATH, image_paths)
=======
def build_per_page_zip_bytes(pages: list[dict]) -> bytes:
    """Build a zip archive with page_XXX.md files from OCR results."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for page in sorted(pages, key=lambda item: item["page"]):
            filename = f"page_{page['page']:03d}.md"
            zf.writestr(filename, page["text"])
    buffer.seek(0)
    return buffer.getvalue()

>>>>>>> Stashed changes

def process_pdf_to_markdown(
    pdf_path: Path,
    config: OCRRuntimeConfig = DEFAULT_RUNTIME_CONFIG,
    pdf_images_dir: Path = PDF_IMAGES_DIR,
    markdown_dir: Path = MARKDOWN_DIR,
    write_markdown_files: bool = True,
    resume: bool = False,
    on_render_progress: RenderProgressCallback | None = None,
    on_page_done: PageDoneCallback | None = None,
    on_page_error: PageErrorCallback | None = None,
    should_stop: ShouldStopCallback | None = None,
) -> dict:
    """Run the complete OCR flow and return a structured result for UI/CLI callers."""
    start_time = time.time()

    preflight_checks(config, pdf_path, pdf_images_dir, markdown_dir)
    num_pages = get_page_count(pdf_path)
    print(f"  PDF: {pdf_path.name} -- {num_pages} pages")

    image_paths = render_pages(
        pdf_path=pdf_path,
        num_pages=num_pages,
        pdf_images_dir=pdf_images_dir,
        target_longest_image_dim=config.target_longest_image_dim,
        on_render_progress=on_render_progress,
        should_stop=should_stop,
    )

    if should_stop is not None and should_stop():
        raise OCRCancelledError("OCR run cancelled before inference started")

    client = create_client(config)

    pages, failed_pages = run_ocr(
        client=client,
        config=config,
        pdf_path=pdf_path,
        image_paths=image_paths,
        markdown_dir=markdown_dir,
        write_markdown_files=write_markdown_files,
        resume=resume,
        on_page_done=on_page_done,
        on_page_error=on_page_error,
        should_stop=should_stop,
    )

    duration_seconds = round(time.time() - start_time, 2)
    return {
        "pages": pages,
        "failed_pages": failed_pages,
        "num_pages": num_pages,
        "duration_seconds": duration_seconds,
        "image_paths": image_paths,
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    config = DEFAULT_RUNTIME_CONFIG
    result = process_pdf_to_markdown(
        pdf_path=PDF_PATH,
        config=config,
        pdf_images_dir=PDF_IMAGES_DIR,
        markdown_dir=MARKDOWN_DIR,
        write_markdown_files=True,
        resume=True,
    )

    save_json_outputs(
        pages=result["pages"],
        corpus_json_path=CORPUS_JSON,
        sample_json_path=SAMPLE_JSON,
        sample_pages=config.sample_pages,
    )

    print(f"\n{'='*60}")
    print("Done!")
    print(f"Failed pages: {len(result['failed_pages'])}")
    print(f"Duration: {result['duration_seconds']}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
