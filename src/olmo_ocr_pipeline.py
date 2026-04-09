"""
olmo_ocr_pipeline.py - Main RAG-preprocessing pipeline using olmOCR 2.

High-level flow
---------------
1. Render every page of the source PDF as a PIL image (via PyMuPDF).
2. Send each page image to the olmOCR 2 vision-language model and collect
   the Markdown text it produces.
3. Extract any figures/graphics embedded inside the PDF and save them as
   individual PNG files under image_and_markdown/images/.
4. Write the concatenated Markdown to image_and_markdown/output.md, with
   inline references to the extracted images.

Dependencies (see requirements.txt)
-------------------------------------
    torch, transformers, qwen-vl-utils, pymupdf, Pillow
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

# Allow running the script directly from the src/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# Step 1 - PDF -> page images
# ===========================================================================

def render_pdf_pages(pdf_path: Path) -> List[Image.Image]:
    """
    Render each page of pdf_path to a PIL RGB Image using PyMuPDF.

    Parameters
    ----------
    pdf_path : Path
        Absolute path to the PDF file.

    Returns
    -------
    list[PIL.Image.Image]
        One RGB image per page, resized to fit the configured max dimensions.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required to render PDF pages. "
            "Install it with:  pip install pymupdf"
        ) from exc

    log.info("Rendering PDF: %s", pdf_path)
    doc = fitz.open(str(pdf_path))
    pages: List[Image.Image] = []
    zoom = config.PDF_RENDER_DPI / 72.0  # 72 DPI is the PDF baseline
    matrix = fitz.Matrix(zoom, zoom)

    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img = _resize_image(img)
        pages.append(img)
        log.info(
            "  Rendered page %d/%d (%dx%d px)",
            page_num, len(doc), img.width, img.height,
        )

    doc.close()
    return pages


def _resize_image(img: Image.Image) -> Image.Image:
    """Down-scale img so it fits within the configured maximum dimensions."""
    max_w, max_h = config.MAX_IMAGE_WIDTH, config.MAX_IMAGE_HEIGHT
    if img.width <= max_w and img.height <= max_h:
        return img
    ratio = min(max_w / img.width, max_h / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size, Image.LANCZOS)


# ===========================================================================
# Step 2 - Load the olmOCR 2 model
# ===========================================================================

def load_model() -> Tuple:
    """
    Load the olmOCR 2 model and its processor from HuggingFace Hub.

    olmOCR 2 is built on Qwen2-VL.  The model weights are downloaded
    automatically on the first run.

    Returns
    -------
    tuple[AutoModelForVision2Seq, AutoProcessor]
    """
    try:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required. "
            "Install them with:  pip install torch transformers"
        ) from exc

    log.info("Loading olmOCR 2 model: %s", config.MODEL_NAME)

    device_map = config.DEVICE if config.DEVICE != "auto" else "auto"

    processor = AutoProcessor.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    log.info("Model loaded successfully.")
    return model, processor


# ===========================================================================
# Step 3 - OCR: page image -> Markdown text
# ===========================================================================

def build_ocr_prompt() -> str:
    """
    Return the system/user prompt used to ask the model to produce Markdown.

    The prompt is the standard olmOCR 2 instruction: convert the page image
    to a clean, faithful Markdown representation, preserving headings, lists,
    tables and maths.
    """
    return (
        "Convert this document page image to Markdown. "
        "Preserve all text, headings, lists, tables, equations, and code blocks. "
        "Do not add any commentary. Output only the Markdown text."
    )


def ocr_page(
    page_image: Image.Image,
    model,
    processor,
    page_number: int,
) -> str:
    """
    Run olmOCR 2 on a single page image and return the Markdown string.

    Parameters
    ----------
    page_image : PIL.Image.Image
        RGB image of one PDF page.
    model :
        Loaded AutoModelForVision2Seq instance.
    processor :
        Corresponding AutoProcessor instance.
    page_number : int
        1-based page number (used only for logging).

    Returns
    -------
    str
        Markdown text produced by the model.
    """
    import torch

    log.info("  OCR page %d ...", page_number)

    prompt_text = build_ocr_prompt()

    # Build the multimodal chat messages expected by Qwen2-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": page_image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Apply the chat template
    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Prepare tensors
    inputs = processor(
        text=[text_input],
        images=[page_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    gen_kwargs = {"max_new_tokens": config.MAX_NEW_TOKENS}
    if config.TEMPERATURE > 0:
        gen_kwargs["temperature"] = config.TEMPERATURE
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only the newly generated tokens (exclude the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    new_token_ids = output_ids[0][prompt_len:]
    markdown = processor.decode(new_token_ids, skip_special_tokens=True).strip()
    return markdown


# ===========================================================================
# Step 4 - Extract embedded images from the PDF
# ===========================================================================

def extract_pdf_images(pdf_path: Path, images_dir: Path) -> List[Path]:
    """
    Extract figures and graphics embedded in the PDF and save them as PNG.

    Only images whose pixel area exceeds config.MIN_IMAGE_AREA are saved
    (smaller bitmaps are typically decorative bullets or artefacts).

    Parameters
    ----------
    pdf_path : Path
        Absolute path to the source PDF.
    images_dir : Path
        Directory where extracted PNG files are written.

    Returns
    -------
    list[Path]
        Sorted list of paths to the saved PNG files.
    """
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for image extraction. "
            "Install it with:  pip install pymupdf"
        ) from exc

    images_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    doc = fitz.open(str(pdf_path))
    img_index = 0

    for page_num, page in enumerate(doc, start=1):
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            if width * height < config.MIN_IMAGE_AREA:
                continue  # Skip tiny / decorative images

            img_index += 1
            ext = base_image.get("ext", "png")
            out_name = f"page{page_num:03d}_img{img_index:03d}.{ext}"
            out_path = images_dir / out_name

            out_path.write_bytes(image_bytes)
            saved.append(out_path)
            log.info(
                "  Extracted image: %s  (%dx%d)", out_name, width, height
            )

    doc.close()
    log.info("Total images extracted: %d", len(saved))
    return sorted(saved)


# ===========================================================================
# Step 5 - Assemble and write the final Markdown file
# ===========================================================================

def write_markdown(
    page_markdowns: List[str],
    extracted_images: List[Path],
    output_path: Path,
) -> None:
    """
    Concatenate per-page Markdown and append an image gallery section,
    then write the result to output_path.

    Parameters
    ----------
    page_markdowns : list[str]
        OCR Markdown text for each page in order.
    extracted_images : list[Path]
        Paths to extracted image files (absolute).
    output_path : Path
        Destination .md file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    # --- Per-page content ---
    for i, md in enumerate(page_markdowns, start=1):
        lines.append(f"<!-- Page {i} -->")
        lines.append(md)
        lines.append("")  # blank line between pages

    # --- Extracted images gallery ---
    if extracted_images:
        lines.append("---")
        lines.append("")
        lines.append("## Extracted Figures")
        lines.append("")
        for img_path in extracted_images:
            # Make the path relative to the output Markdown file location
            try:
                rel = img_path.relative_to(output_path.parent)
            except ValueError:
                rel = img_path
            alt = img_path.stem.replace("_", " ")
            lines.append(f"![{alt}]({rel})")
            lines.append("")

    full_text = "\n".join(lines)
    output_path.write_text(full_text, encoding="utf-8")
    log.info("Markdown written to: %s", output_path)


# ===========================================================================
# Main entry point
# ===========================================================================

def run_pipeline(
    pdf_path: Optional[Path] = None,
    output_md_path: Optional[Path] = None,
    images_dir: Optional[Path] = None,
) -> None:
    """
    Execute the full OlmoOCR 2 preprocessing pipeline.

    Parameters
    ----------
    pdf_path : Path, optional
        Source PDF.  Defaults to config.PDF_PATH.
    output_md_path : Path, optional
        Destination Markdown file.  Defaults to config.OUTPUT_MD_PATH.
    images_dir : Path, optional
        Directory for extracted images.  Defaults to config.IMAGES_DIR.
    """
    pdf_path = pdf_path or config.PDF_PATH
    output_md_path = output_md_path or config.OUTPUT_MD_PATH
    images_dir = images_dir or config.IMAGES_DIR

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Source PDF not found: {pdf_path}\n"
            "Place your PDF at data/document.pdf or pass a custom path."
        )

    log.info("=== OlmoOCR 2 Pipeline Start ===")
    log.info("PDF         : %s", pdf_path)
    log.info("Output MD   : %s", output_md_path)
    log.info("Images dir  : %s", images_dir)

    # -- 1. Render pages --
    pages = render_pdf_pages(pdf_path)

    # -- 2. Load model --
    model, processor = load_model()

    # -- 3. OCR each page --
    page_markdowns: List[str] = []
    for i, page_img in enumerate(pages, start=1):
        md = ocr_page(page_img, model, processor, page_number=i)
        page_markdowns.append(md)

    # -- 4. Extract embedded images --
    extracted_images = extract_pdf_images(pdf_path, images_dir)

    # -- 5. Write Markdown --
    write_markdown(page_markdowns, extracted_images, output_md_path)

    log.info("=== OlmoOCR 2 Pipeline Complete ===")


if __name__ == "__main__":
    run_pipeline()
