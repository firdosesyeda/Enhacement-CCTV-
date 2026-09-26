"""Local browser page for uploading and preparing one CCTV recording."""

import io
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE = PROJECT_DIR / "enhance_pipeline.py"
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "ts", "m4v"]
PERSON_MODEL = "yolo26n.pt"


def detect_people(video_path, output_path, confidence, progress, started):
    """Run pretrained YOLO in small batches on every frame and write boxes."""
    from ultralytics import YOLO

    model = YOLO(PERSON_MODEL)  # downloads pretrained weights on first use
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("YOLO could not open the uploaded video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("The video does not have valid FPS or image-size metadata.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the annotated MP4 video.")

    done = 0
    try:
        while True:
            batch_frames = []
            for _ in range(4):
                ok, frame = cap.read()
                if not ok:
                    break
                batch_frames.append(frame)
            if not batch_frames:
                break
            results = model.predict(
                source=batch_frames, classes=[0], conf=confidence, verbose=False
            )
            for result in results:
                writer.write(result.plot())
                done += 1
            if total_frames and (done % 40 < 4 or done == total_frames):
                fraction = min(done / total_frames, 1.0)
                elapsed = int(time.perf_counter() - started)
                progress.progress(
                    fraction,
                    text=(f"YOLO on enhanced video: {done:,} / {total_frames:,} frames · "
                          f"elapsed {elapsed // 60}m {elapsed % 60}s"),
                )
    finally:
        cap.release()
        writer.release()
    return done

st.set_page_config(page_title="CCTV Frame Prep", page_icon="📹")
st.title("CCTV recording → YOLO frames")
st.write(
    "Upload one recording. The tool creates a full enhanced video, runs YOLO person detection "
    "on that enhanced video, and prepares sampled frames for review and labeling."
)

uploaded = st.file_uploader("Choose a CCTV recording", type=VIDEO_TYPES)
interval = st.number_input(
    "Sample one real frame every (seconds)", min_value=0.1, max_value=60.0,
    value=1.0, step=0.1,
)
denoise = st.slider(
    "Optional denoising strength (slower when above 0)",
    min_value=0, max_value=10, value=0,
    help="Leave at 0 for faster processing. Increase it only when the recording has visible grain/noise.",
)
use_clahe = st.checkbox("Adjust local contrast (CLAHE)", value=True)
run_yolo = st.checkbox("Create video with YOLO person detections", value=True)
confidence = st.slider(
    "Minimum person-detection confidence", min_value=0.10, max_value=0.90,
    value=0.25, step=0.05,
    help="Lower values may find more people but can add false detections.",
)

if uploaded and st.button("Process recording", type="primary"):
    started = time.perf_counter()
    st.session_state["detection_message"] = ""
    st.session_state["detection_error"] = ""
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
                "--denoise-strength", str(denoise), "--create-enhanced-video",
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
            st.session_state["pipeline_log"] = output_text

            if process.returncode == 0:
                if run_yolo:
                    detection_path = output_dir / "person_detection" / f"{Path(safe_name).stem}_person_detected.mp4"
                    enhanced_video = output_dir / "enhanced_videos" / f"{Path(safe_name).stem}_enhanced.mp4"
                    try:
                        progress.update(label="Loading YOLO model and detecting people in the enhanced video…")
                        count = detect_people(
                            enhanced_video, detection_path, confidence, st.progress(0), started
                        )
                        st.session_state["detection_message"] = (
                            f"YOLO checked {count:,} frames from the enhanced video."
                        )
                        st.session_state["detection_error"] = ""
                    except Exception as exc:
                        st.session_state["detection_message"] = ""
                        st.session_state["detection_error"] = str(exc)
                archive = io.BytesIO()
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for path in output_dir.rglob("*"):
                        if path.is_file():
                            zf.write(path, path.relative_to(output_dir))
                st.session_state["prepared_zip"] = archive.getvalue()
                st.session_state["prepared_name"] = f"{Path(safe_name).stem}_prepared.zip"
                st.session_state["process_success"] = True
                st.session_state["process_elapsed"] = time.perf_counter() - started
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
            "Download the ZIP for the enhanced video, person-detection video, frames, and log."
        )
        if st.session_state.get("detection_message"):
            st.info(st.session_state["detection_message"])
        if st.session_state.get("detection_error"):
            st.warning(f"YOLO video was not created: {st.session_state['detection_error']}")
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
