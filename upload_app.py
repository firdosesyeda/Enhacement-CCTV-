"""Local browser page for uploading and preparing one CCTV recording."""

import io
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE = PROJECT_DIR / "enhance_pipeline.py"
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "ts", "m4v"]

st.set_page_config(page_title="CCTV Frame Prep", page_icon="📹")
st.title("CCTV recording → YOLO frames")
st.write(
    "Upload one recording. The tool keeps the original unchanged and prepares "
    "sampled original and gently enhanced frames for review and labeling."
)

uploaded = st.file_uploader("Choose a CCTV recording", type=VIDEO_TYPES)
interval = st.number_input(
    "Sample one real frame every (seconds)", min_value=0.1, max_value=60.0,
    value=1.0, step=0.1,
)
denoise = st.slider("Denoising strength", min_value=0, max_value=10, value=3)
use_clahe = st.checkbox("Adjust local contrast (CLAHE)", value=True)

if uploaded and st.button("Process recording", type="primary"):
    started = time.perf_counter()
    with st.status("Starting video processing…", expanded=True) as progress:
        with tempfile.TemporaryDirectory(prefix="cctv_frame_prep_") as temp_name:
            temp_dir = Path(temp_name)
            input_dir = temp_dir / "input"
            output_dir = temp_dir / "output"
            input_dir.mkdir()
            safe_name = Path(uploaded.name).name
            input_file = input_dir / safe_name
            input_file.write_bytes(uploaded.getvalue())

            command = [
                sys.executable, str(PIPELINE), "--input", str(input_file),
                "--output", str(output_dir), "--every-n-seconds", str(interval),
                "--denoise-strength", str(denoise),
            ]
            if not use_clahe:
                command.append("--no-clahe")
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            while process.poll() is None:
                seconds = int(time.perf_counter() - started)
                progress.update(label=f"Processing recording… elapsed {seconds // 60}m {seconds % 60}s")
                time.sleep(1)
            output_text, _ = process.communicate()
            elapsed = time.perf_counter() - started
            st.session_state["process_elapsed"] = elapsed
            st.session_state["pipeline_log"] = output_text

            if process.returncode == 0:
                archive = io.BytesIO()
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for path in output_dir.rglob("*"):
                        if path.is_file():
                            zf.write(path, path.relative_to(output_dir))
                st.session_state["prepared_zip"] = archive.getvalue()
                st.session_state["prepared_name"] = f"{Path(safe_name).stem}_prepared.zip"
                st.session_state["process_success"] = True
                progress.update(label="Processing complete", state="complete")
            else:
                st.session_state["prepared_zip"] = None
                st.session_state["process_success"] = False
                progress.update(label="Processing failed", state="error")

if "pipeline_log" in st.session_state:
    if st.session_state.get("process_success"):
        duration = st.session_state.get("process_elapsed", 0)
        st.success(
            f"Preparation finished in {int(duration // 60)}m {int(duration % 60)}s. "
            "Download the ZIP, then review the original and enhanced frames."
        )
        st.download_button(
            "Download prepared frames and log",
            data=st.session_state["prepared_zip"],
            file_name=st.session_state["prepared_name"],
            mime="application/zip",
        )
    else:
        st.error("Processing did not finish. See the message below.")
    with st.expander("Processing details"):
        st.code(st.session_state["pipeline_log"] or "No output was returned.")

st.caption(
    "The enhanced images are candidates for annotation, not guaranteed to improve accuracy. "
    "The tool does not label objects or train YOLO."
)
