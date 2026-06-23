from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import DEFAULT_RUNTIME_CONFIG, OCRRuntimeConfig, SERVER_TYPE, LLAMACPP_PORT, LLAMACPP_HOST

PIPELINE_IMPORT_ERROR: str | None = None
build_per_page_zip_bytes: Any = None
combine_markdown_pages: Any = None
get_page_count: Any = None
process_pdf_to_markdown: Any = None

try:
    from src.olmo_ocr_pipeline import (
        build_per_page_zip_bytes,
        combine_markdown_pages,
        get_page_count,
        process_pdf_to_markdown,
    )
except ModuleNotFoundError as exc:
    PIPELINE_IMPORT_ERROR = str(exc)

st.set_page_config(page_title="OLMo OCR -> Markdown", layout="wide")


@st.cache_data(show_spinner=False)
def _check_local_pdf(path: str) -> tuple[bool, str]:
    p = Path(path).expanduser()
    if not p.exists():
        return False, "File does not exist"
    if not p.is_file():
        return False, "Path is not a file"
    if p.suffix.lower() != ".pdf":
        return False, "File is not a PDF"
    return True, "OK"


st.title("OLMo OCR PDF Digitizer")
st.caption(f"Convert PDFs to Markdown via {SERVER_TYPE.title()} OpenAI-compatible API")

if PIPELINE_IMPORT_ERROR:
    st.error(
        "Missing Python dependency while loading OCR pipeline. "
        f"Import error: {PIPELINE_IMPORT_ERROR}"
    )
    st.info(
        "**Setup steps:**\n"
        "1. Activate venv: `source .venv/bin/activate`\n"
        "2. Install/sync deps: `uv sync` or `pip install -r requirements.txt`\n"
        "3. Run app: `python -m streamlit run app.py`"
    )
    st.stop()

# Server info callout
with st.expander("⚙️ Server Configuration", expanded=False):
    if SERVER_TYPE == "llamacpp":
        st.info(
            f"**llama.cpp server** configured for OpenAI API on `{LLAMACPP_HOST}:{LLAMACPP_PORT}/v1`\n\n"
            "To generate a launch command, run:\n"
            "```python\nfrom src.config import get_llamacpp_server_command\nprint(get_llamacpp_server_command('/path/to/model.gguf'))\n```"
        )
    else:
        st.info(f"**LM Studio** configured. Ensure server is running on `{DEFAULT_RUNTIME_CONFIG.api_base_url}`.")

with st.sidebar:
    st.header(f"🔌 {SERVER_TYPE.title()} API Configuration")
    api_base_url = st.text_input("API base URL", value=DEFAULT_RUNTIME_CONFIG.api_base_url)
    api_model = st.text_input("Model", value=DEFAULT_RUNTIME_CONFIG.api_model)

    st.header("OCR Settings")
    num_workers = st.slider("Workers", min_value=1, max_value=16, value=DEFAULT_RUNTIME_CONFIG.num_workers)
    max_retries = st.slider("Max retries", min_value=1, max_value=10, value=DEFAULT_RUNTIME_CONFIG.max_retries)
    retry_delay = st.slider("Retry delay (seconds)", min_value=1, max_value=60, value=DEFAULT_RUNTIME_CONFIG.retry_delay)
    max_new_tokens = st.slider("Max new tokens", min_value=512, max_value=8192, value=DEFAULT_RUNTIME_CONFIG.max_new_tokens, step=256)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=float(DEFAULT_RUNTIME_CONFIG.temperature), step=0.1)
    target_longest_image_dim = st.slider(
        "Target longest image dimension",
        min_value=512,
        max_value=2048,
        value=DEFAULT_RUNTIME_CONFIG.target_longest_image_dim,
        step=128,
    )

runtime_config = OCRRuntimeConfig(
    api_base_url=api_base_url,
    api_model=api_model,
    num_workers=num_workers,
    max_retries=max_retries,
    retry_delay=retry_delay,
    target_longest_image_dim=target_longest_image_dim,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    sample_pages=DEFAULT_RUNTIME_CONFIG.sample_pages,
)

mode = st.radio("Input mode", ["Upload PDF", "Local PDF path"], horizontal=True)

pdf_name = None
pdf_path: Path | None = None

if mode == "Upload PDF":
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None:
        pdf_name = uploaded.name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            tmp_pdf.write(uploaded.getvalue())
            pdf_path = Path(tmp_pdf.name)
else:
    local_pdf = st.text_input("Absolute path to local PDF", value="")
    if local_pdf.strip():
        ok, message = _check_local_pdf(local_pdf)
        if not ok:
            st.error(message)
        else:
            pdf_path = Path(local_pdf).expanduser()
            pdf_name = pdf_path.name
            st.success(f"Ready: {pdf_path}")

if pdf_path is not None:
    try:
        total_pages = get_page_count(pdf_path)
        st.info(f"Detected {total_pages} pages in {pdf_name}")
    except Exception as exc:
        st.warning(f"Could not read page count yet: {exc}")

run_clicked = st.button("Start OCR", type="primary", disabled=pdf_path is None)

if run_clicked and pdf_path is not None:
    if "ocr_runs" not in st.session_state:
        st.session_state["ocr_runs"] = []

    with tempfile.TemporaryDirectory(prefix="olmo_streamlit_") as run_dir:
        run_root = Path(run_dir)
        images_dir = run_root / "pdf_images"
        markdown_dir = run_root / "markdown"

        render_progress_bar = st.progress(0)
        render_status = st.empty()

        ocr_progress_bar = st.progress(0)
        ocr_status = st.empty()

        errors_box = st.empty()
        error_messages: list[str] = []

        render_total = {"count": 1}
        ocr_total = {"count": 1}
        processed_pages = {"count": 0}

        def on_render_progress(done: int, total: int) -> None:
            render_total["count"] = max(total, 1)
            render_progress_bar.progress(done / render_total["count"])
            render_status.text(f"Rendering pages: {done}/{total}")

        def on_page_done(page_num: int, _text: str) -> None:
            processed_pages["count"] += 1
            progress = processed_pages["count"] / max(ocr_total["count"], 1)
            ocr_progress_bar.progress(min(progress, 1.0))
            ocr_status.text(f"OCR pages processed: {processed_pages['count']}/{ocr_total['count']}")

        def on_page_error(page_num: int, message: str) -> None:
            error_messages.append(f"Page {page_num}: {message}")
            errors_box.warning("\n".join(error_messages[-5:]))

        started = time.time()

        try:
            # Determine total pages once so OCR progress has stable denominator.
            ocr_total["count"] = get_page_count(pdf_path)

            result = process_pdf_to_markdown(
                pdf_path=pdf_path,
                config=runtime_config,
                pdf_images_dir=images_dir,
                markdown_dir=markdown_dir,
                write_markdown_files=True,
                resume=False,
                on_render_progress=on_render_progress,
                on_page_done=on_page_done,
                on_page_error=on_page_error,
            )
        except Exception as exc:
            st.error(f"OCR run failed: {exc}")
            result = None

        if result is not None:
            pages = result["pages"]
            combined_markdown = combine_markdown_pages(pages)
            per_page_zip = build_per_page_zip_bytes(pages)
            elapsed = time.time() - started

            st.session_state["ocr_runs"].append(
                {
                    "pdf_name": pdf_name,
                    "result": result,
                    "combined_markdown": combined_markdown,
                    "per_page_zip": per_page_zip,
                    "elapsed": elapsed,
                    "errors": error_messages,
                }
            )

if st.session_state.get("ocr_runs"):
    latest = st.session_state["ocr_runs"][-1]
    result = latest["result"]
    pages = result["pages"]

    st.subheader("Run Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDF", latest["pdf_name"])
    c2.metric("Pages", str(result["num_pages"]))
    c3.metric("Failed pages", str(len(result["failed_pages"])))
    c4.metric("Duration (s)", f"{latest['elapsed']:.1f}")

    st.subheader("Downloads")
    download_name = Path(latest["pdf_name"]).stem if latest["pdf_name"] else "document"
    st.download_button(
        "Download combined markdown",
        data=latest["combined_markdown"].encode("utf-8"),
        file_name=f"{download_name}.md",
        mime="text/markdown",
    )
    st.download_button(
        "Download per-page markdown zip",
        data=latest["per_page_zip"],
        file_name=f"{download_name}_pages.zip",
        mime="application/zip",
    )

    st.subheader("Markdown Preview")
    page_numbers = [page["page"] for page in pages]
    selected_page = st.selectbox("Choose page", options=page_numbers, index=0)
    selected_text = next(page["text"] for page in pages if page["page"] == selected_page)

    tab1, tab2 = st.tabs(["Per-page", "Combined"])
    with tab1:
        st.markdown(selected_text)
        with st.expander("Raw markdown"):
            st.code(selected_text, language="markdown")

    with tab2:
        st.markdown(latest["combined_markdown"])
        with st.expander("Raw combined markdown"):
            st.code(latest["combined_markdown"], language="markdown")

    if latest["errors"]:
        st.subheader("Recent Errors")
        st.text("\n".join(latest["errors"][-20:]))
