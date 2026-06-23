from __future__ import annotations

import json
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import DEFAULT_RUNTIME_CONFIG, OCRRuntimeConfig, SERVER_TYPE, LLAMACPP_PORT, LLAMACPP_HOST

PIPELINE_IMPORT_ERROR: str | None = None
build_per_page_zip_bytes: Any = None
combine_markdown_pages: Any = None
get_page_count: Any = None
process_pdf_to_markdown: Any = None
OCRCancelledError: Any = None

try:
    from src.olmo_ocr_pipeline import (
        OCRCancelledError,
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


def _init_job_state() -> dict[str, Any]:
    if "ocr_job" not in st.session_state:
        st.session_state["ocr_job"] = {
            "status": "idle",
            "message": "",
            "pdf_name": None,
            "render_done": 0,
            "render_total": 1,
            "ocr_done": 0,
            "ocr_total": 1,
            "errors": [],
            "result": None,
            "combined_markdown": None,
            "per_page_zip": None,
            "elapsed": None,
            "stop_event": None,
            "thread": None,
            "run_dir": None,
            "unload_message": "",
        }
    return st.session_state["ocr_job"]


def _reset_job_state(job: dict[str, Any]) -> None:
    stop_event = job.get("stop_event")
    thread = job.get("thread")
    run_dir = job.get("run_dir")
    job.clear()
    job.update(
        {
            "status": "idle",
            "message": "",
            "pdf_name": None,
            "render_done": 0,
            "render_total": 1,
            "ocr_done": 0,
            "ocr_total": 1,
            "errors": [],
            "result": None,
            "combined_markdown": None,
            "per_page_zip": None,
            "elapsed": None,
            "stop_event": stop_event,
            "thread": thread,
            "run_dir": run_dir,
            "unload_message": "",
        }
    )


def _build_lmstudio_rest_url(api_base_url: str, path: str) -> str:
    parsed = urllib.parse.urlparse(api_base_url)
    base_path = parsed.path.rstrip("/")
    if base_path.endswith("/v1"):
        base_path = base_path[:-3]
    rest_path = f"{base_path}/api/v1{path}"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, rest_path, "", "", ""))


def _http_json(url: str, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=10) as response:
        raw = response.read().decode("utf-8")
    return {} if not raw else json.loads(raw)


def _resolve_loaded_instance_id(api_base_url: str, api_model: str) -> str | None:
    url = _build_lmstudio_rest_url(api_base_url, "/models")
    payload = _http_json(url, "GET")
    models = payload.get("models", [])

    for model in models:
        if model.get("key") != api_model:
            continue
        loaded_instances = model.get("loaded_instances", [])
        if loaded_instances:
            return loaded_instances[0].get("id")

    for model in models:
        loaded_instances = model.get("loaded_instances", [])
        if loaded_instances and model.get("selected_variant") == api_model:
            return loaded_instances[0].get("id")

    return None


def _unload_active_model(api_base_url: str, api_model: str) -> str:
    try:
        instance_id = _resolve_loaded_instance_id(api_base_url, api_model)
        if not instance_id:
            return "No loaded model instance found to unload."

        url = _build_lmstudio_rest_url(api_base_url, "/models/unload")
        _http_json(url, "POST", {"instance_id": instance_id})
        return f"Unloaded model instance: {instance_id}"
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        return f"Unload failed: HTTP {exc.code} {details}".strip()
    except Exception as exc:
        return f"Unload failed: {exc}"


def _run_ocr_job(
    job: dict[str, Any],
    pdf_path: Path,
    runtime_config: OCRRuntimeConfig,
) -> None:
    run_root = Path(tempfile.mkdtemp(prefix="olmo_streamlit_"))
    images_dir = run_root / "pdf_images"
    markdown_dir = run_root / "markdown"
    stop_event = job["stop_event"]

    job["run_dir"] = str(run_root)
    job["message"] = "Preparing OCR run..."
    started = time.time()

    def should_stop() -> bool:
        return bool(stop_event and stop_event.is_set())

    def on_render_progress(done: int, total: int) -> None:
        job["render_done"] = done
        job["render_total"] = max(total, 1)
        job["message"] = f"Rendering pages: {done}/{total}"

    def on_page_done(_page_num: int, _text: str) -> None:
        job["ocr_done"] += 1
        job["message"] = f"OCR pages processed: {job['ocr_done']}/{job['ocr_total']}"

    def on_page_error(page_num: int, message: str) -> None:
        job["errors"].append(f"Page {page_num}: {message}")

    try:
        job["ocr_total"] = get_page_count(pdf_path)
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
            should_stop=should_stop,
        )

        if should_stop():
            raise OCRCancelledError("OCR run stopped by user")

        pages = result["pages"]
        job["result"] = result
        job["combined_markdown"] = combine_markdown_pages(pages)
        job["per_page_zip"] = build_per_page_zip_bytes(pages)
        job["elapsed"] = time.time() - started
        job["status"] = "completed"
        job["message"] = "OCR run completed."
    except OCRCancelledError:
        job["elapsed"] = time.time() - started
        job["status"] = "stopped"
        job["message"] = "OCR run stopped."
    except Exception as exc:
        job["elapsed"] = time.time() - started
        job["status"] = "failed"
        job["message"] = f"OCR run failed: {exc}"
        job["errors"].append(str(exc))
    finally:
        job["thread"] = None


job_state = _init_job_state()


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

    # --- OCR Inference Settings ---
    st.header("📊 Inference Settings")
    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=512,
        max_value=8192,
        value=DEFAULT_RUNTIME_CONFIG.max_new_tokens,
        step=256,
        help="Maximum tokens to generate per page OCR result"
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(DEFAULT_RUNTIME_CONFIG.temperature),
        step=0.1,
        help="Model randomness (0=deterministic, 1.0=balanced, >1=very random)"
    )

    # --- OCR Pipeline Settings ---
    st.header("⚙️ OCR Pipeline")
    st.info("Single-job mode: one OCR request at a time for stability on the RX 6900 XT.")
    max_retries = st.slider(
        "Max retries per page",
        min_value=1,
        max_value=10,
        value=DEFAULT_RUNTIME_CONFIG.max_retries
    )
    retry_delay = st.slider(
        "Retry delay (seconds)",
        min_value=1,
        max_value=60,
        value=DEFAULT_RUNTIME_CONFIG.retry_delay
    )

    # --- Image Settings ---
    st.header("🖼️ Image Processing")
    target_longest_image_dim = st.slider(
        "Target longest image dimension",
        min_value=512,
        max_value=2048,
        value=DEFAULT_RUNTIME_CONFIG.target_longest_image_dim,
        step=128,
        help="Resize PDF pages to this max dimension (smaller = faster but lower quality)"
    )


runtime_config = OCRRuntimeConfig(
    api_base_url=api_base_url,
    api_model=api_model,
    num_workers=DEFAULT_RUNTIME_CONFIG.num_workers,
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

is_running = job_state["status"] in {"running", "stopping"}

action_col1, action_col2, action_col3 = st.columns([1, 1, 2])
run_clicked = action_col1.button("Start OCR", type="primary", disabled=pdf_path is None or is_running)
stop_clicked = action_col2.button("Stop OCR", disabled=not is_running)
refresh_clicked = action_col3.button("Refresh status", disabled=not is_running)

if stop_clicked and is_running:
    if job_state.get("stop_event") is not None:
        job_state["stop_event"].set()
    job_state["status"] = "stopping"
    job_state["message"] = "Stopping OCR and unloading model..."
    job_state["unload_message"] = _unload_active_model(runtime_config.api_base_url, runtime_config.api_model)
    st.rerun()

if refresh_clicked:
    st.rerun()

if run_clicked and pdf_path is not None:
    _reset_job_state(job_state)
    job_state["status"] = "running"
    job_state["pdf_name"] = pdf_name
    job_state["stop_event"] = threading.Event()
    worker = threading.Thread(
        target=_run_ocr_job,
        kwargs={
            "job": job_state,
            "pdf_path": pdf_path,
            "runtime_config": runtime_config,
        },
        daemon=True,
    )
    job_state["thread"] = worker
    worker.start()
    st.rerun()

if job_state["status"] != "idle":
    render_progress = job_state["render_done"] / max(job_state["render_total"], 1)
    ocr_progress = job_state["ocr_done"] / max(job_state["ocr_total"], 1)

    st.subheader("Current Run")
    st.write(job_state["message"] or f"Status: {job_state['status']}")
    st.progress(min(render_progress, 1.0), text=f"Rendering pages: {job_state['render_done']}/{job_state['render_total']}")
    st.progress(min(ocr_progress, 1.0), text=f"OCR pages processed: {job_state['ocr_done']}/{job_state['ocr_total']}")

    if job_state["unload_message"]:
        st.info(job_state["unload_message"])

    if job_state["status"] == "failed":
        st.error(job_state["message"])
    elif job_state["status"] == "stopped":
        st.warning(job_state["message"])
    elif job_state["status"] == "completed":
        st.success(job_state["message"])

if job_state.get("result") is not None:
    result = job_state["result"]
    pages = result["pages"]

    st.subheader("Run Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDF", job_state["pdf_name"])
    c2.metric("Pages", str(result["num_pages"]))
    c3.metric("Failed pages", str(len(result["failed_pages"])))
    c4.metric("Duration (s)", f"{job_state['elapsed']:.1f}")

    st.subheader("Downloads")
    download_name = Path(job_state["pdf_name"]).stem if job_state["pdf_name"] else "document"
    st.download_button(
        "Download combined markdown",
        data=job_state["combined_markdown"].encode("utf-8"),
        file_name=f"{download_name}.md",
        mime="text/markdown",
    )
    st.download_button(
        "Download per-page markdown zip",
        data=job_state["per_page_zip"],
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
        st.markdown(job_state["combined_markdown"])
        with st.expander("Raw combined markdown"):
            st.code(job_state["combined_markdown"], language="markdown")

    if job_state["errors"]:
        st.subheader("Recent Errors")
        st.text("\n".join(job_state["errors"][-20:]))

if job_state["status"] in {"running", "stopping"}:
    time.sleep(2)
    st.rerun()
